import cv2
from ultralytics import YOLO
from crashToFirebase import crash_to_fire
import numpy as np

# load yolo model
model = YOLO("yolo11n.pt")

class_list = model.names
videoPath = r"E:\codify_hackarena\vehicle-crash-detector\test_videos\test (16).mp4"
cap = cv2.VideoCapture(videoPath)

# Store previous positions and stationary time of tracked objects
prev_positions = {}
stationary_time = {}
crash_counter = 0
crash_reported_flag = False

# Speed threshold for crash detection (pixels per frame)
SPEED_THRESHOLD = 5

# Time threshold for stationary detection (in frames)
STATIONARY_THRESHOLD = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True)

    if results and results[0].boxes is not None and results[0].boxes.data is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        class_indices = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []

        # Loop through each detected object
        for i in range(len(track_ids)):
            if i >= len(boxes) or i >= len(class_indices):
                continue  # Safety check

            x1, y1, x2, y2 = map(int, boxes[i])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track_id = int(track_ids[i])
            class_name = class_list[class_indices[i]]

            # Calculate speed
            if track_id in prev_positions:
                prev_cx, prev_cy = prev_positions[track_id]
                speed = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                # Check for sudden stop (potential crash)
                if speed < SPEED_THRESHOLD:
                    if track_id in stationary_time:
                        stationary_time[track_id] += 1
                    else:
                        stationary_time[track_id] = 1
                else:
                    stationary_time[track_id] = 0

                # If vehicle is stationary for a while, consider it a crash
                if stationary_time.get(track_id, 0) > STATIONARY_THRESHOLD:
                    if not crash_reported_flag:
                        crash_counter += 1
                        crash_id = f"crash-{crash_counter}"
                        print(f"Crash detected! Crash ID: {crash_id}")
                        crash_to_fire(cx, cy, crash_id)
                        crash_reported_flag = True

                        # Mark the box with "Crash Detected"
                        cv2.putText(frame, "Crash Detected", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Update previous position
            prev_positions[track_id] = (cx, cy)

            # Draw a dot at the center and display the tracking ID and class name
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLO Object Tracking & Counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()