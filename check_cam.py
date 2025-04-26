import cv2

camera_name = ! libcamera-hello --list-cameras | grep -Po '(?<=\().+(?=\))'
pipeline = (
    f"libcamerasrc camera-name={camera_name[0]} "
    "! video/x-raw, width=640, height=480, framerate=30/1, format=RGBx "
    "! videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# capture video from the first camera (index 0)
# cap = cv2.VideoCapture(0)

# check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# read a frame from the camera
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from camera.")
    exit()

# save frame to a file
cv2.imwrite("test_frame.jpg", frame)
print("Frame captured and saved as test_frame.jpg")
