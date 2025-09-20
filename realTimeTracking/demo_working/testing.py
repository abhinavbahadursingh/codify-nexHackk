import csv
import time
from collections import defaultdict

def calculate_and_store_avg_speed():
    """Calculate the average speed for each track_id and store it in avg_speed.csv."""

    speed_data = defaultdict(list)

    # Read the existing speed logs
    try:
        with open("speed_log.csv", mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue  # Skip empty or malformed rows
                track_id, speed, timestamp = row
                speed_data[track_id].append(float(speed))

        # Calculate average speeds
        avg_speeds = []
        for track_id, speeds in speed_data.items():
            avg_speed = sum(speeds) / len(speeds)  # Compute average
            avg_speeds.append([track_id, round(avg_speed, 2), int(time.time())])  # Corrected this line

        # Store in avg_speed.csv
        with open("avg_speed.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["track_id", "avg_speed", "timestamp"])  # Write header
            writer.writerows(avg_speeds)

        print("Average speed calculations stored in avg_speed.csv")

    except FileNotFoundError:
        print("No speed log file found. Make sure speed_log.csv exists.")

# Call the function
calculate_and_store_avg_speed()

if __name__ == '__main__':
    calculate_and_store_avg_speed()