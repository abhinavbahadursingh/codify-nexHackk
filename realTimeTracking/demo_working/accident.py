from firebase_auth import initialize_firebase
from gps import pixel_to_gps
from datetime import datetime
import time
import sys
import os

# Add root directory to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
from blockchain import add_to_blockchain

ref = initialize_firebase()

def pushAccident(track_id, cx, cy):
    lat, lon = pixel_to_gps(cx, cy)
    timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    track_id = int(float(track_id))

    accident_ref = ref.child("vehicle_Breakdown").child(str(track_id))
    data_to_log = {
        "latitude": str(lat),
        "vehicle_id": track_id,
        "longitude": str(lon),
        "timestamp": timestamp
    }
    accident_ref.set(data_to_log)

    # Add to blockchain
    add_to_blockchain('vehicle_breakdown', data_to_log)