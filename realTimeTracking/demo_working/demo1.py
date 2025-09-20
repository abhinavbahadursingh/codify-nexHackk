import cv2
import math

# Open the video file
cap = cv2.VideoCapture(r"/Data/Video/realTimeTesting.mp4")


def process_frame(frame, fps):
        """
        Processes a given video frame to detect moving objects and calculate speed.

        Parameters:
        frame (numpy.ndarray): The current frame from the video.
        fps (int): Frames per second of the video.

        Returns:
        frame (numpy.ndarray): Processed frame with speed overlay.
        """
        scale_factor = 0.00175  # Meters per pixel (Adjust based on real-world measurement)
        frame_time = 1 / fps  # Time per frame in seconds
        fgbg = cv2.createBackgroundSubtractorMOG2()
        prev_center = None

        # Convert to grayscale and apply background subtraction
        fgmask = fgbg.apply(frame)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small objects
                        x, y, w, h = cv2.boundingRect(contour)
                        curr_center = (x + w // 2, y + h // 2)  # Calculate centroid

                        # Draw a circle at the center
                        cv2.circle(frame, curr_center, 5, (0, 255, 0), -1)

                        if prev_center is not None:
                                # Calculate Euclidean distance
                                distance_pixels = math.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                                            (curr_center[1] - prev_center[1]) ** 2)

                                # Convert to meters
                                distance_meters = distance_pixels * scale_factor

                                # Calculate speed (m/s and km/h)
                                speed_mps = distance_meters / frame_time
                                speed_kmph = speed_mps * 3.6

                                # Display speed on video
                                cv2.putText(frame, f"Speed: {speed_kmph:.2f} km/h", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # Update previous center
                        prev_center = curr_center

        return frame
