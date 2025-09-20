import sys
import threading
import time
import random
from datetime import datetime

sys.path.insert(1, r'/demo_working/firebase_auth.py')
import firebase_admin
from firebase_admin import credentials, db
import numpy as np


# ---------------- FIREBASE INIT ----------------
def initialize_firebase():
    """Initialize Firebase and return a database reference"""
    cred = credentials.Certificate(
        r"E:\codify_hackarena\traffic-monitoring-f490d-firebase-adminsdk-fbsvc-29d3ed2ce7.json"
    )  # Update with your correct path
    if not firebase_admin._apps:  # Prevents re-initialization error
        firebase_admin.initialize_app(
            cred,
            {
                "databaseURL": "https://traffic-monitoring-f490d-default-rtdb.firebaseio.com/"
            },
        )
    return db.reference("traffic_data")  # Returns database reference


# ---------------- TRANSFORM ----------------
H = np.array([
    [-0.027557, -0.050103, 42.587],
    [-2.9606e-17, -0.52608, 215.69],
    [0, -0.0031216, 1]
])

def pixel_to_gps(cx, cy):
    pixel_coords = np.array([cx, cy, 1])
    world_coords = np.dot(H, pixel_coords)
    world_coords /= world_coords[2]

    latitude_ref = 28.634556
    longitude_ref = 77.448029
    scale_factor = 0.00001

    lat = latitude_ref + (world_coords[1] * scale_factor)
    lon = longitude_ref + (world_coords[0] * scale_factor)

    return lat, lon


# ---------------- CRASH UPLOAD ----------------
def crash_to_fire(cx, cy, track_id):
    ref = initialize_firebase()
    lat, lon = pixel_to_gps(cx, cy)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    accident_ref = ref.child("crash_vehicle").child(str(track_id))
    accident_ref.set({
        "latitude": str(lat),
        "vehicle_id": track_id,
        "longitude": str(lon),
        "timestamp": timestamp
    })
    print(f"[SENT] Vehicle {track_id} @ ({lat}, {lon}) at {timestamp}")


# ---------------- THREAD FUNCTION ----------------
def send_dummy_data():
    track_id = 1
    while True:
        # Generate random pixel coords
        cx = random.randint(100, 800)
        cy = random.randint(100, 600)

        crash_to_fire(cx, cy, track_id)

        track_id += 1
        time.sleep(5)  # every 5 seconds


# ---------------- MAIN ----------------
if __name__ == "__main__":
    t = threading.Thread(target=send_dummy_data, daemon=True)
    t.start()

    # Keep main thread alive
    while True:
        time.sleep(1)
