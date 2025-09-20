import sys
from datetime import datetime
import os

# Add root directory to path to find blockchain module
# This assumes blockchain.py is in the root of the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
from blockchain import add_to_blockchain

sys.path.insert(1, r'/demo_working/firebase_auth.py')
import firebase_admin
from firebase_admin import credentials, db
import numpy as np

def initialize_firebase():
    """Initialize Firebase and return a database reference"""
    cred = credentials.Certificate(os.path.join(root_dir, "traffic-monitoring-f490d-firebase-adminsdk-fbsvc-29d3ed2ce7.json"))
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://traffic-monitoring-f490d-default-rtdb.firebaseio.com/'
        })
    return db.reference("traffic_data")

# Transformation matrix from camera calibration
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

def crash_to_fire(cx, cy, track_id):
    ref = initialize_firebase()

    lat, lon = pixel_to_gps(cx, cy)
    timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    accident_ref = ref.child("crash_vehicle").child(str(track_id))
    data_to_log = {
        "latitude": str(lat),
        "vehicle_id": track_id,
        "longitude": str(lon),
        "timestamp": timestamp
    }
    accident_ref.set(data_to_log)

    # Add the data to our blockchain
    add_to_blockchain('crash', data_to_log)