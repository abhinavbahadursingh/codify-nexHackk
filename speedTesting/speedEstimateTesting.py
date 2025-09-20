"""
vehicle_speed_estimator.py

Requirements:
  - Python 3.8+
  - ultralytics
  - supervision
  - opencv-python(-headless)
  - numpy
  - tqdm

Adjust SOURCE and TARGET to match your camera and real-world ROI. TARGET units are in meters.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import argparse
import csv
import time
import os

# -----------------------------
# Utility: Perspective transform
# -----------------------------
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        source: 4x2 array of pixel coordinates in the image (A,B,C,D)
        target: 4x2 array of coordinates in real-world units (meters) representing where those points map to
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        points: Nx2 array of (x,y) image coordinates
        returns Nx2 array of transformed coordinates in target space (same units as TARGET)
        """
        if points is None or len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)


# -----------------------------
# Speed estimator
# -----------------------------
def compute_speed_for_tracker(deque_y_coords, fps):
    """
    Given a deque of Y coordinates (or any 1D coordinate in meters) ordered newest->oldest,
    compute speed in km/h using difference over stored time window.
    This uses absolute difference between newest and oldest and time = len/window / fps.
    """
    if len(deque_y_coords) < 2:
        return None
    y_new = deque_y_coords[0]   # newest (most recent appended at left - we'll keep pattern)
    y_old = deque_y_coords[-1]  # oldest
    distance_m = abs(y_new - y_old)   # in meters
    time_s = len(deque_y_coords) / fps
    if time_s == 0:
        return None
    speed_m_s = distance_m / time_s
    speed_kmh = speed_m_s * 3.6
    return speed_kmh


# -----------------------------
# Main pipeline
# -----------------------------
def main(
    video_path = r"E:\codify_hackarena\realTimeTracking\data\3.mp4",
    model_path=YOLO("yolo11n.pt"),
    output_path=r"E:\codify_hackarena\realTimeTracking\data\3.mp4",
    display=True,
    save_csv=True,
    min_conf=0.3,
    speed_fps_window_fraction=1.0,  # window length in seconds (1.0 -> ~1 sec)
):
    # -----------------------------
    # Configure ViewTransformer: ADAPT THESE TO YOUR CAMERA
    # -----------------------------
    # These are example SOURCE pixel coordinates (A,B,C,D) from the tutorial.
    # You MUST set these for your camera: pick 4 corners of a rectangular patch on road whose true
    # size (in meters) you know. Example target is 24m x 249m (from tutorial).
    SOURCE = np.array([
        [1252, 787], 
        [2298, 803], 
        [5039, 2159], 
        [-550, 2159]
    ], dtype=np.float32)

    TARGET = np.array([
        [0, 0],
        [24, 0],
        [24, 249],
        [0, 249],
    ], dtype=np.float32)
    # NOTE: TARGET units are in meters. Here target rectangle is width=24m, length=249m.
    # Update SOURCE and TARGET to your data.

    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # -----------------------------
    # Load YOLO model
    # -----------------------------
    model = YOLO(model_path)   # change to path / name of your model
    # If using Roboflow / remote inference, you can adapt model loading accordingly.

    # -----------------------------
    # Initialize BYTETrack via supervision
    # -----------------------------
    byte_track = sv.ByteTrack()  # uses default params; tune as needed

    # -----------------------------
    # Video IO
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_info = sv.VideoInfo.from_video_path(video_path)

    out_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Data structures for speed calc
    # coordinates: dict tracker_id -> deque of y coordinates (or chosen axis) in meters
    # We'll store position along the road axis (y in transformed space). Which axis corresponds to forward depends on TARGET orientation - we follow tutorial using y.
    window_len = int(max(2, round(fps * max(0.5, speed_fps_window_fraction))))  # keep at least 2 entries
    coordinates = defaultdict(lambda: deque(maxlen=window_len))

    # csv saving
    if save_csv:
        csv_file = open("speeds.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "frame", "tracker_id", "speed_kmh"])

    # annotators
    # bbox_annotator = sv.BoundingBoxAnnotator()
    # label_annotator = sv.LabelAnnotator()
    # line_annotator = sv.LineAnnotator()

    # helper for colored-pill background on text
    def draw_label(img, text, xy):
        # small helper - draw filled rect then put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = int(xy[0]), int(xy[1]) - 10
        cv2.rectangle(img, (x, y - 2), (x + w + 4, y + h + 4), (0,0,0), -1)
        cv2.putText(img, text, (x + 2, y + h - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run detection (ultralytics)
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)


        # Update tracker
        detections = byte_track.update_with_detections(detections=detections)

        # Get anchor points (bottom-center) for each detection
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)  # Nx2 (x,y) in image coords

        # Transform to real-world coordinates (meters)
        if len(points) > 0:
            points_world = view_transformer.transform_points(points).astype(np.float32)
        else:
            points_world = np.zeros((0,2), dtype=np.float32)

        # store transformed coordinate for each tracker id
        # detections.tracker_id order corresponds to points order
        for tid, pt in zip(detections.tracker_id, points_world):
            # Here we pick the Y coordinate in transformed space (assuming opponent axis)
            # If your TARGET has length along X, choose x accordingly. Check orientation on your setup.
            _, y_world = pt
            # We'll append newest at left, to compute [newest ... oldest] easier
            # But deque appends right; to keep compatibility, appendleft then read [0] as newest.
            coordinates[tid].appendleft(y_world)

        # compute speed and draw boxes/labels
        annotated_frame = frame.copy()
        for i, box in enumerate(detections.xyxy):  # each box: [x1,y1,x2,y2]
            x1, y1, x2, y2 = box
            tid = detections.tracker_id[i]
            score = detections.confidence[i] if detections.confidence is not None else None

            label = f"ID:{tid}"
            speed = None
            if len(coordinates[tid]) >= max(2, int(window_len/2)):
                speed = compute_speed_for_tracker(coordinates[tid], fps)
            # draw bbox
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            # compose label
            if speed is not None:
                txt = f"ID:{tid} {speed:.1f} km/h"
            else:
                txt = f"ID:{tid} -- km/h"
            draw_label(annotated_frame, txt, (int(x1), int(y1)))

            # save csv
            if save_csv and speed is not None:
                csv_writer.writerow([time.time(), frame_idx, tid, f"{speed:.3f}"])

        # overlay a small legend
        cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}  FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # write output frame
        if out_writer is not None:
            out_writer.write(annotated_frame)

        if display:
            cv2.imshow("speed_estimation", annotated_frame)
            # waitKey small so UI responsive; pressing 'q' quits
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pbar.update(1)

    pbar.close()
    cap.release()
    if out_writer:
        out_writer.release()
    if display:
        cv2.destroyAllWindows()
    if save_csv:
        csv_file.close()
    print("Done. Output:", output_path)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle speed estimation from video")
    parser.add_argument("--video", "-v", required=True, help="Path to input video")
    parser.add_argument("--model", "-m", default="yolov8x.pt", help="YOLO model (ultralytics) path or name")
    parser.add_argument("--out", "-o", default="out_speed.mp4", help="Output annotated video path")
    parser.add_argument("--no-display", action="store_true", help="Do not show window")
    parser.add_argument("--min-conf", type=float, default=0.35, help="Min detection confidence")
    args = parser.parse_args()

    main(video_path=args.video,
         model_path=args.model,
         output_path=args.out,
         display=not args.no_display,
         min_conf=args.min_conf)
