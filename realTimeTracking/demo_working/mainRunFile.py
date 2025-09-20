import threading

import cv2
from ultralytics import YOLO
from collections import defaultdict
import math
import queue
from datetime import datetime

import vehicleData
import speedData
import accident
import gps
# from demo_working import videoPath
# Load YOLO model on GPU
model = YOLO("yolo11n.pt")

class_list = model.names  # List of class names

# Open the video file
# videoUrl=videoPath.get_stream_path()
cap = cv2.VideoCapture(r'E:\codify\realTimeTracking\data\3.mp4')


if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Video stream opened successfully!")
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default FPS if retrieval fails


# Define line positions for counting
line_y_red = 298  # Red line position
line_y_blue = line_y_red + 150  # Blue line position

# Variables to store counting and tracking information
counted_ids_red_to_blue = set()
counted_ids_blue_to_red = set()

# Dictionaries to count objects by class for each direction
count_red_to_blue = defaultdict(int)  # Moving downwards
count_blue_to_red = defaultdict(int)  # Moving upwards

# State dictionaries to track which line was crossed first
crossed_red_first = {}
crossed_blue_first = {}

scale_factor = 0.165  # Meters per pixel (Adjust based on real-world measurement)
frame_time = 1 / fps


prev_positions = {}

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break  # Stop processing if the frame is invalid

    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True)

    # Ensure results are not empty
    if results and results[0].boxes is not None and results[0].boxes.data is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        class_indices = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
        confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []

        # Draw the lines on the frame
        cv2.line(frame, (250, line_y_red), (690, line_y_red), (0, 0, 255), 1)
        cv2.putText(frame, 'Red Line', (190, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.line(frame, (27, line_y_blue), (900, line_y_blue), (255, 0, 0), 1)
        cv2.putText(frame, 'Blue Line', (27, line_y_blue - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # Loop through each detected object
        for i in range(len(track_ids)):
            if i >= len(boxes) or i >= len(class_indices):
                continue  # Safety check

            x1, y1, x2, y2 = map(int, boxes[i])

            cx = (x1 + x2) // 2  # Calculate the center point
            cy = (y1 + y2) // 2

            # Get the class name using the class index
            class_name = class_list[class_indices[i]]

            # Draw a dot at the center and display the tracking ID and class name
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_ids[i]} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # vehicleData.track_vehicle(track_ids[i],cx,cy)
            uploadThread1 = threading.Thread(target=vehicleData.track_vehicle, args=(track_ids[i], cx, cy,class_name))
            uploadThread1.start()
            # tracked_Vehicle_data = {"vehicle_id": track_id, "longitude:":str(lat),"latitude ":str(lon),"entry time ":timestamp}
            # Use the track ID to manage previous positions
            track_id = track_ids[i]
            curr_center = (cx, cy)

            # Calculate speed only if the object is above the red line
            if cy < (line_y_red ):
                if track_id in prev_positions:
                    prev_center = prev_positions[track_id]
                    # Calculate Euclidean distance between current and previous center
                    distance_pixels = math.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                                (curr_center[1] - prev_center[1]) ** 2)
                    # Convert pixels to meters
                    distance_meters = distance_pixels *0.143
                    # Calculate speed (m/s and km/h)
                    speed_mps = distance_meters / frame_time
                    speed_kmph = speed_mps * 3.6
                    # if(speed_kmph==0.0):
                    #     uploadThread2=threading.Thread(target=accident.pushAccident,args=(track_id, cx, cy))
                    #     uploadThread2.start()
                    uploadThread3=threading.Thread(target=speedData.log_speed,args=(track_id,speed_kmph))
                    uploadThread3.start()
                    # Display speed on video near the object's center
                    cv2.putText(frame, f"Speed: {speed_kmph:.2f} km/h", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)
                # Update the previous position for this object
                prev_positions[track_id] = curr_center

            # Calculate speed only if the object is above the blue line
            if cy < (line_y_blue - 20)and(cy>line_y_red):
                if track_id in prev_positions:
                    prev_center = prev_positions[track_id]
                    # Calculate Euclidean distance between current and previous center
                    distance_pixels = math.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                                (curr_center[1] - prev_center[1]) ** 2)
                    # Convert pixels to meters
                    distance_meters = distance_pixels * scale_factor
                    # Calculate speed (m/s and km/h)
                    speed_mps = distance_meters / frame_time
                    speed_kmph = speed_mps * 3.6
                    uploadThread4=threading.Thread(target=speedData.log_speed,args=(track_id,speed_kmph))
                    uploadThread4.start()

                    # Display speed on video near the object's center
                    cv2.putText(frame, f"Speed: {speed_kmph:.2f} km/h", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 1)
                # Update the previous position for this object
                prev_positions[track_id] = curr_center
            # Check if the object crosses the red line
            if line_y_red - 5 <= cy <= line_y_red + 5:
                crossed_red_first[track_ids[i]] = True
                uploadThread5 = threading.Thread(target=vehicleData.track_vehicle, args=(track_ids[i], cx, cy,class_name))
                uploadThread5.start()

            # Check if the object crosses the blue line
            if line_y_blue - 5 <= cy <= line_y_blue + 5:
                crossed_blue_first[track_ids[i]] = True
                uploadThread7 = threading.Thread(target=vehicleData.track_vehicle, args=(track_ids[i], cx, cy,class_name))
                uploadThread7.start()
            # Counting logic for downward direction (red -> blue)
            if track_ids[i] in crossed_red_first and track_ids[i] not in counted_ids_red_to_blue:
                if line_y_blue - 5 <= cy <= line_y_blue + 5:
                    counted_ids_red_to_blue.add(track_ids[i])
                    count_red_to_blue[class_name] += 1

            # Counting logic for upward direction (blue -> red)
            if track_ids[i] in crossed_blue_first and track_ids[i] not in counted_ids_blue_to_red:
                if line_y_red - 5 <= cy <= line_y_red + 5:
                    counted_ids_blue_to_red.add(track_ids[i])
                    count_blue_to_red[class_name] += 1

    # Show the frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)

    # Exit loop if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()