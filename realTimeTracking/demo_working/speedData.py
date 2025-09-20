import csv
from collections import defaultdict
from firebase_auth import initialize_firebase
import pandas as pd
import time
from datetime import datetime
import sys
import os

# Add root directory to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
from blockchain import add_to_blockchain

ref = initialize_firebase()
tracked_vehicles = set()
csv_file = "traffic_speed_data.csv"

def store_speed(track_id, speed):
    new_entry = pd.DataFrame({
        "track_id": [track_id],
        "timestamp": [datetime.now().strftime("%H:%M:%S")],
        "speed": [speed]
    })
    try:
        new_entry.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def calculate_average_speed():
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("CSV file not found!")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S")
    df.sort_values(by=["track_id", "timestamp"], inplace=True)
    df["interval"] = df["timestamp"].dt.floor("5S")
    avg_speed = df.groupby(["track_id", "interval"])["speed"].mean().reset_index()
    print("\nAverage Speed per 5-second Interval:")
    print(avg_speed)

def log_speed(track_id, speed):
    track_id = int(float(track_id))
    speed_ref = ref.child("speed_Data").child("real_Time_Speed").child(str(track_id))
    store_speed(track_id, speed)
    
    count_ref = speed_ref.child("count")
    count_snapshot = count_ref.get()
    count = 1 if count_snapshot is None else count_snapshot + 1

    data_to_log = {"speed": speed, "timestamp": str(datetime.now())}
    speed_ref.child(f"time{count}").set(data_to_log)
    count_ref.set(count)

    # Add to blockchain
    blockchain_payload = {
        "vehicle_id": track_id,
        "speed": speed,
        "timestamp": data_to_log["timestamp"]
    }
    add_to_blockchain('speed_data', blockchain_payload)