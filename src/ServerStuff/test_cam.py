#Function: To Check Camera Index
import cv2


for i in range(5):
    cap=cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame=cap.read()
        print(f"Camera {i} is available")
        cv2.imshow(f"Camera {i}", frame)
        cv2.waitKey(1000)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Resolution: {width} x {height}")

        cap.release()
    else:
        print(f"Camera {i} is not available")