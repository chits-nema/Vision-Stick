# Function: To Check Camera Index with Picamera2
from picamera2 import Picamera2
import cv2

cameras = Picamera2.global_camera_info()

if not cameras:
    print("No cameras found!")
else:
    for i, cam_info in enumerate(cameras):
        print(f"\nCamera {i} info: {cam_info}")

        try:
            picam2 = Picamera2(i)

            # Configure and start the camera
            config = picam2.create_preview_configuration()
            picam2.configure(config)
            picam2.start()

            # Capture one frame
            frame = picam2.capture_array()

            # Show the frame
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)

            # Get resolution
            height, width = frame.shape[:2]
            print(f"Resolution: {width} x {height}")

            picam2.close()
        except Exception as e:
            print(f"Camera {i} could not be opened: {e}")

cv2.destroyAllWindows()
