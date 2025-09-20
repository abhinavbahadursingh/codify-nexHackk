
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import csv
from demo_working import vehicleData
from demo_working import speedData
from collections import deque
from demo_working import accident
# Load YOLO model
model = YOLO('yolo11n.pt')
class_list = model.names

# Load video
cap = cv2.VideoCapture(r'C:\Users\tar30\realTimeTracking\Data\Video\testoing.mp4')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
# Tracking variables
vehicle_data = {}
frame_count = 0

# Distance calibration
pixels_per_meter = 20

# Open CSV file for logging
csv_file = open("vehicle_speed_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Frame", "VehicleID", "Class", "Speed_km_h", "Center_X", "Center_Y"])



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    # Process frame with YOLO tracking
    results = model.track(frame, persist=True)
    cv2.line(frame,(0,0),(23,23),(0,0,0),5)
    # If detections exist
    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = class_list[class_id]

                if track_id not in vehicle_data:
                    vehicle_data[track_id] = {'class': class_name, 'first_seen_frame': frame_count,
                                              'last_seen': current_time, 'positions': [(frame_count, cx, cy)],
                                              'speed': 0.0}
                    vehicleData.track_vehicle(track_id, cx,cy)

                else:
                    vehicle_data[track_id]['last_seen'] = current_time
                    vehicle_data[track_id]['positions'].append((frame_count, cx, cy))
                    if len(vehicle_data[track_id]['positions']) > 10:
                        vehicle_data[track_id]['positions'] = vehicle_data[track_id]['positions'][-10:]

                    if len(vehicle_data[track_id]['positions']) >= 2:
                        old_frame, old_x, old_y = vehicle_data[track_id]['positions'][0]
                        new_frame, new_x, new_y = vehicle_data[track_id]['positions'][-1]
                        pixel_distance = np.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)
                        distance_meters = pixel_distance / pixels_per_meter
                        frame_diff = new_frame - old_frame
                        time_diff = frame_diff / fps

                        if time_diff > 0:
                            speed_km_h = (distance_meters / time_diff) * 3.6
                            alpha = 0.3
                            old_speed = vehicle_data[track_id]['speed']
                            new_speed = alpha * speed_km_h + (1 - alpha) * old_speed
                            vehicle_data[track_id]['speed'] = round(new_speed, 2)
                            if round(new_speed, 2)==0.0:
                                        accident.pushAccident(track_id, cx, cy)

                            speedData.log_speed(track_id,round(new_speed,2))
                            speedData.log_speed_to_csv(track_id,round(new_speed,2))


                            speedData.push_avg_speed_to_firebase()  # Push avg speed to Firebase

                csv_writer.writerow([current_time, frame_count, track_id, class_name,
                                     vehicle_data[track_id]['speed'], cx, cy])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                speed = vehicle_data[track_id]['speed']
                cv2.putText(frame, f"Speed: {speed:.2f} KM/H", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    vehicles_to_remove = [v_id for v_id, data in vehicle_data.items()
                          if current_time - data['last_seen'] > 5]
    for v_id in vehicles_to_remove:
        vehicle_data.pop(v_id)


    vehicle_count = len(vehicle_data)
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # # Update Graph Data
    # time_series.append(frame_count)
    # vehicle_count_series.append(vehicle_count)
    # avg_speed = np.mean([v['speed'] for v in vehicle_data.values()]) if vehicle_data else 0
    # speed_series.append(avg_speed)
    # # Real-time graph update
    # ax1.clear()
    # ax1.plot(time_series, vehicle_count_series, color='b', label='Vehicles')
    # ax1.set_ylabel("Number of Vehicles")
    # ax1.legend()
    # ax2.clear()
    # ax2.plot(time_series, speed_series, color='r', label='Speed (km/h)')
    # ax2.set_xlabel("Frames")
    # ax2.set_ylabel("Average Speed (km/h)")
    # ax2.legend()
    # plt.pause(0.01)
    # Display the frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)

    if cv2.waitKey(2) & 0xFF == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
plt.ioff()
plt.show()