import cv2

# Load classifier
vehicle_cascade = cv2.CascadeClassifier('cars.xml')

# Load video
video = cv2.VideoCapture('./data/3.mp4')

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        print("End of video stream or failed to read.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Vehicle Detection', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
