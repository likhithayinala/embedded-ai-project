import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (640, 480), "format": "XRGB8888"}
))
picam2.start()

# capture video from the first camera (index 0)
# cap = cv2.VideoCapture(0)

# # check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# read a frame from the camera
frame = picam2.capture_array()
# if not ret:
#     print("Error: Could not read frame from camera.")
#     exit()

# save frame to a file
cv2.imwrite("test_frame.jpg", frame)
print("Frame captured and saved as test_frame.jpg")
picam2.stop()
