import cv2
import queue
import threading
import firebase_admin
from firebase_admin import credentials, db
import time
import sys
sys.path.insert(1, r'/demo_working/firebase_auth.py')
import firebase_admin
from firebase_admin import credentials, db

# Queue for storing object data
data_queue = queue.Queue()
cred = credentials.Certificate(
    r"C:\Users\tar30\realTimeTracking\fireBase\traffic-monitoring-f490d-myfirebase-adminsdk-fbsvc-f24fd8d3d3.json")  # Update with your correct path
if not firebase_admin._apps:  # Prevents re-initialization error
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://traffic-monitoring-f490d-default-rtdb.firebaseio.com/'  # Replace with your Firebase URL
    })

def object_tracking():
    cap = cv2.VideoCapture(r"C:\Users\tar30\realTimeTracking\Data\Video\realTimeTesting.mp4")  # Capture video from webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated object detection (Replace this with your object tracking code)
        detected_object = {"id": time.time(), "label": "car", "confidence": 0.9}

        # Append data to queue
        data_queue.put(detected_object)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def upload_to_firebase():
    while True:
        if not data_queue.empty():
            data = data_queue.get()
            ref = db.reference("/object_tracking")
            ref.push(data)  # Upload to Firebase
            print(f"Uploaded: {data}")
        time.sleep(1)  # Adjust based on needs


# Start object tracking in the main thread
tracking_thread = threading.Thread(target=object_tracking)
upload_thread = threading.Thread(target=upload_to_firebase, daemon=True)  # Daemon thread for background upload

tracking_thread.start()
upload_thread.start()

tracking_thread.join()  # Wait for tracking to finish
