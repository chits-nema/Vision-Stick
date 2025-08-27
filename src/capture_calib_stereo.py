import numpy as np
import cv2
from picamera2 import Picamera2

print('Starting the Calibration. Press and maintain the space bar to exit the script\n')
print('Push (s) to save the image you want andw push (c) to see next frame without saving the image')

id_image=0
 
# termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Call the two cameras
#CamR= cv2.VideoCapture(2)   # 1 -> Right Camera
#CamL= cv2.VideoCapture(1)   # 2 -> Left Camera

#Force both cameras to have the same resolution
#CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

picamR = Picamera2(1)  # Right Camera
picamL = Picamera2(0)  # Left Camera