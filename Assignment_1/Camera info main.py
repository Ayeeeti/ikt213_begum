import cv2
import os
import sys

# open webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    sys.exit(1)

_= cam.read()

# gets the default frame
fps = cam.get(cv2.CAP_PROP_FPS)
FRAME_WIDTH = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Target folder and file
save_dir = os.path.expanduser("/IKT213_begum/assignment_1/solutions")
os.makedirs(save_dir, exist_ok=True)

file_path = os.path.join(save_dir, "Lab1 camera info.txt")

with open("Lab1 Camera info.txt", "w") as f:
    f.write(f"fps: {fps}\n")
    f.write(f"FRAME_WIDTH: {FRAME_WIDTH}\n")
    f.write(f"FRAME_HEIGHT: {FRAME_HEIGHT}\n")

    print("camera info saved")


cam.release()
cv2.destroyAllWindows()