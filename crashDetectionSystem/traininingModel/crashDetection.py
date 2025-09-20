from ultralytics import YOLO
import cv2

from crashToFirebase import crash_to_fire

model = YOLO(r"E:\makeForLkoTrafficCrash\Models\best.pt")
class_list = model.names  # maps class index to class name

cap = cv2.VideoCapture(r"E:\makeForLkoTrafficCrash\test_videos\test (6).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, save=False, verbose=False)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes
    detected = False
    if boxes is not None and boxes.cls is not None:
        class_indices = boxes.cls.cpu().numpy()
        for cls_id in class_indices:
            class_name = class_list[int(cls_id)].lower()
            # if class_name == "car_car_accident":  # Replace with your crash class name if different
            #     crash_to_fire(12,12,9)
    cv2.imshow("Crash Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
