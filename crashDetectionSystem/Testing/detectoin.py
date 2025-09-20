import cv2
from ultralytics import YOLO
from crashToFirebase import crash_to_fire
import numpy as np

# Load YOLO model
model = YOLO("yolo11n.pt")

class_list = model.names
videoPath = r"E:\codify_hackquanta\crashDetectionSystem\test_videos\Untitled video - Made with Clipchamp (1).mp4"
cap = cv2.VideoCapture(videoPath)

# Crash detection parameters
prev_positions = {}
trajectory = {}
speed_history = {}
crash_counter = 0
crash_reported_flag = False

# Thresholds for crash detection
SPEED_THRESHOLD = 10  # Minimum speed change to be considered significant
DIRECTION_CHANGE_THRESHOLD = 30  # Minimum direction change (in degrees) to be considered significant
STATIONARY_THRESHOLD = 20  # Number of frames a vehicle must be stationary to be considered crashed
IOU_THRESHOLD = 0.2  # Intersection over Union threshold for collision detection


def calculate_speed(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_direction(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True)

    if results and results[0].boxes is not None and results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        class_indices = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []

        current_boxes = {int(track_ids[i]): boxes[i] for i in range(len(track_ids))}

        for i in range(len(track_ids)):
            track_id = int(track_ids[i])
            x1, y1, x2, y2 = map(int, boxes[i])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Update trajectory and speed history
            if track_id not in trajectory:
                trajectory[track_id] = []
            trajectory[track_id].append((cx, cy))
            if len(trajectory[track_id]) > 2:
                trajectory[track_id].pop(0)

            if len(trajectory[track_id]) > 1:
                speed = calculate_speed(trajectory[track_id][-1], trajectory[track_id][-2])
                if track_id not in speed_history:
                    speed_history[track_id] = []
                speed_history[track_id].append(speed)
                if len(speed_history[track_id]) > 5:
                    speed_history[track_id].pop(0)

                # Crash detection logic
                if not crash_reported_flag:
                    # 1. Sudden deceleration
                    if len(speed_history[track_id]) > 2:
                        avg_speed = np.mean(speed_history[track_id][:-1])
                        current_speed = speed_history[track_id][-1]
                        if avg_speed > SPEED_THRESHOLD and current_speed < 1:
                            crash_counter += 1
                            crash_id = f"crash-{crash_counter}"
                            print(f"Crash detected (sudden stop)! Crash ID: {crash_id}")
                            crash_to_fire(cx, cy, crash_id)
                            crash_reported_flag = True
                            cv2.putText(frame, "Crash Detected", (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # 2. Collision with another vehicle
                    for other_track_id, other_box in current_boxes.items():
                        if track_id != other_track_id:
                            iou = calculate_iou(boxes[i], other_box)
                            if iou > IOU_THRESHOLD:
                                crash_counter += 1
                                crash_id = f"crash-{crash_counter}"
                                print(f"Crash detected (collision)! Crash ID: {crash_id}")
                                crash_to_fire(cx, cy, crash_id)
                                crash_reported_flag = True
                                cv2.putText(frame, "Crash Detected", (x1, y1 - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                break

            # Display track ID
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLO Object Tracking & Counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()