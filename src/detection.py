import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import glob

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
names = model.names

# reference vals for calibration a4 width 21 cm
KNOWN_DISTANCE = 50.0
KNOWN_WIDTH = 21.0
 
# method using the similarity rate
def focal_length_calculation(known_distance, known_width, width_in_rf_image):
    return (width_in_rf_image * known_distance) / known_width

# calculate the distance bw cam and object
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
    if per_width == 0:
        return None
    return (knownWidth * focalLength) / perWidth

# for the first time for callibration
focalLength = None
calibrated = False

#Camera Calibration
# Defining the dimensions of checkerboard aka the width and height of the checkerboard

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpointsR = [] 
imgpointsL = [] 
 
# Defining the world coordinates for 3D points
# objp is the 3d points relating to the world coordinates 
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
 
# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0,67):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t= str(i)
    #second argument 0 grays the image
    ChessImaR= cv2.imread('chessboard-R'+t+'.png',0)    # Right side
    ChessImaL= cv2.imread('chessboard-L'+t+'.png',0)    # Left side

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    #corners is the output array of the detetcted corners
    #CHECKERBOARD == patternSize - Number of inner corners per a chessboard row and column 
    #if code doesn't work remove flags and set to none, just keeping flags rn to maintain robustness

    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)  # Left side
    

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        #takes in the OG image and the location of the corners(of the checkerboards) 
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

        # Draw and display the corners
        #just to give a detection of the cornerx 
        #can be commented out later or removed we are just checking that the algorithm actually words

        ChessImaR = cv2.drawChessboardCorners(ChessImaR, CHECKERBOARD, cornersR, ret)
        ChessImaL = cv2.drawChessboardCorners(ChessImaL, CHECKERBOARD, cornersL, ret)



    cv2.imshow('img',img)
    cv2.waitKey(0)
 
cv2.destroyAllWindows()
 
cap = cv2.VideoCapture(0)
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""

#ret = true if calibrated
#mtx -> intrinsic camera matrix (focal length fx, fy; optical center cx, cy)
#dist -> distortion coefficients of the lens
#rvecs -> rotation extrinsics for each image
#tvecs -> translation extrinsics for each image

"""Internal parameters of the camera/lens system. E.g. focal length, optical center, and radial distortion coefficients of the lens.
External parameters : This refers to the orientation (rotation and translation) of the camera with respect to some world coordinate system."""
# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))


#we use OmtxR, OmtxL, roiL and roiR for distrotion purposes of the camera(intrinsic parameters) Om are our new camera matrices from the old code :)
#new camera matrix for distortion old code next line
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


print('Cameras Ready to use')
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
#flags = 0
#flags |= cv2.CALIB_FIX_INTRINSIC
#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria = criteria_stereo,
                                                          flags = cv2.CALIB_FIX_INTRINSIC)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)

while True:
    read, frame = cap.read()
    if not read: 
        break

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
 
    # crop the image....this is cropping out the black parts of the image when things are undistorted
    x, y, w, h = roi
    #dist in now the undistoted frame
    dst = dst[y:y+h, x:x+w]
    

    results = model(frame)[0]
    #results = model.predict(stream=True, imgsz=512)

    detections = sv.Detections.from_ultralytics(results)

   # labels = [
   #     f"{class_name} {confidence:.2f}"
   #     for class_name, confidence
   #     in zip(detections['class_name'], detections.confidence)
   # ]

   # Camera calibration -> so we do a pre stream to fill the vector arrays or we count the number of frames 
   # we then use these to calibrate the corners to get the 2d points 

    labels = []
    for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        class_name = names[int(class_id)]

        # bounding box width in pixels 
        x1, y1, x2, y2 = xyxy
        per_width = x2 - x1

        # if callibration is not done yet we are calculating it with a4 paper
        if not calibrated and class_name == "bottle":  # paper as reference
            focalLength = focal_length_calculation(KNOWN_DISTANCE, KNOWN_WIDTH, per_width)
            calibrated = True
            print(f"[INFO] Camera is callibrated. Focal Length = {focalLength:.2f}")

        distance = None
        if calibrated:
            distance = distance_to_camera(KNOWN_WIDTH, focalLength, per_width)

        if distance:
            label_text = f"{class_name} {confidence:.2f}, {distance:.1f}cm"
        else:
            label_text = f"{class_name} {confidence:.2f}"

        labels.append(label_text)

        # warning
        if distance and distance < 100:
            print(f"[WARNING] {class_name} so close: {distance:.1f}cm")

    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_dst = box_annotator.annotate(scene= dst.copy(), detections=detections)
    annotated_dst_img = label_annotator.annotate(scene=annotated_dst, detections=detections, labels=labels)

    cv2.imshow("COCO Detection", annotated_frame)
    cv2.imshow("Undistorted", annotated_dst)

    if cv2.waitKey(1) == ord('q'):
        break

    for r in results: 
        for c in r.boxes.cls:
            print(names[int(c)])


cap.release()
cv2.destroyAllWindows()

