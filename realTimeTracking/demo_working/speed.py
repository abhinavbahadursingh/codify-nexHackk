# Set the known real-world distance between the red and blue lines (e.g., in meters)
distance = 10.0  # Change this value as needed
def calculate_speed(duration):
    # For instance, if the distance between the tracks is in meters and you want speed in km/h:
    distance_meters = 10.0  # Change this to your actual distance between tracks
    if duration > 0:
        speed_m_per_sec = distance_meters / duration
        speed_km_per_hr = speed_m_per_sec * 3.6
        return round(speed_km_per_hr, 2)
    return 0
