import numpy as np

# Transformation matrix from camera calibration
H = np.array([
    [-0.027557, -0.050103, 42.587],
    [-2.9606e-17, -0.52608, 215.69],
    [0, -0.0031216, 1]
])

def pixel_to_gps(cx, cy):
    pixel_coords = np.array([cx, cy, 1])  # Convert to homogeneous coordinates
    world_coords = np.dot(H, pixel_coords)  # Apply transformation
    world_coords /= world_coords[2]  # Normalize

    # Assuming reference point (latitude_ref, longitude_ref) and scale
    latitude_ref = 28.634556# Reference latitude (Example: New Delhi)
    longitude_ref = 77.448029  # Reference longitude

    scale_factor = 0.00001  # Approximate conversion factor (adjust as needed)
    lat = latitude_ref + (world_coords[1] * scale_factor)
    lon = longitude_ref + (world_coords[0] * scale_factor)

    return lat, lon



