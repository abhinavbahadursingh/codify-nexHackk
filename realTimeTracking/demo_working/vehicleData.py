from firebase_auth import initialize_firebase
from gps import pixel_to_gps
from datetime import datetime
import sys
import os

# Add root directory to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
from blockchain import add_to_blockchain

ref = initialize_firebase()
tracked_vehicles = set()

def track_vehicle(track_id, cx, cy, name):
    lat, lon = pixel_to_gps(cx, cy)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    track_id = int(track_id)
    timestamp = str(timestamp)

    data_to_log = {"vehicle_id": track_id, "longitude": str(lon), "latitude": str(lat), "entry_time": timestamp, "type": name}

    # Log to Firebase
    vehicle_ref = ref.child("vehicle_Data").child("total_vehicle").child(str(track_id))
    vehicle_ref.set(data_to_log)

    type_ref = ref.child("vehicle_Data").child("type_vehicle").child(str(name)).child(str(track_id))
    type_ref.set(data_to_log)

    # Add to blockchain
    add_to_blockchain('vehicle_tracking', data_to_log)