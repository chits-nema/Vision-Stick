import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

#Filtering
kernel= np.ones((3,3),np.uint8)

#Camera Calibration
# Defining the dimensions of checkerboard aka the width and height of the checkerboard

CHECKERBOARD = (8,5)
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

#Scale object points up to the sizes of the squares (2 cm -> 20 mm)
objp = objp *20
 
# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0,64):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
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
    print(t, ChessImaL.shape)
    print(t, ChessImaR.shape)

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

        #ChessImaR = cv2.drawChessboardCorners(ChessImaR, CHECKERBOARD, cornersR, retR)
        #ChessImaL = cv2.drawChessboardCorners(ChessImaL, CHECKERBOARD, cornersL, retL)


    cv2.waitKey(0)
 
cv2.destroyAllWindows()
 
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
hR,wR= ChessImaR.shape[:2]
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        (wR, hR),None,None)
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
hL,wL= ChessImaL.shape[:2]
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        (wL, hL), None, None)
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))


#we use OmtxR, OmtxL, roiL and roiR for distrotion purposes of the camera(intrinsic parameters) Om are our new camera matrices from the old code :)
#new camera matrix for distortion old code next line
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


print('Cameras Ready to use')
print("Camera matrix : \n")
print(mtxR)
print(mtxL)
print("dist : \n")
print(distR)
print(distL)
print("rvecs : \n")
print(rvecsR)
print(rvecsL)
print("tvecs : \n")
print(tvecsR)
print(tvecsL)

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
                                                          (wR, hR),
                                                          criteria = criteria_stereo,
                                                          flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_SAME_FOCAL_LENGTH)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 (wR, hR), R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped

# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             (wL, hL), cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              (wR, hR), cv2.CV_16SC2)

#STEP 3: CREATION OF DISPARITY MAP

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
#stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

#STEP 4: APPLY WLS FILTER
#WLS Parameters
#lmbda = 80000
#sigma = 1.8
#visual_multiplier = 1.0
 
#wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
#wls_filter.setLambda(lmbda)
#wls_filter.setSigmaColor(sigma)

#STEP 5: START STEREOVISION + CALCULATION OF DEPTH MAP

model = YOLO("yolo11n.pt")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
names = model.names

camR = cv2.VideoCapture(0)
camL = cv2.VideoCapture(1)

while True:
    retR, frameR = camR.read()
    retL, frameL = camL.read()
    if not retR or not retL: 
        break

    left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)
    right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

    #Convert from color (BGR) to gray
    grayR = cv2.cvtColor(right_nice, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(left_nice, cv2.COLOR_BGR2GRAY)

    #Compute the 2 images for the depth image
    #disp = stereo.compute(grayL, grayR).astype(np.float32)/16
    #dispL = disp                                                                                                                                                                                                                    
    #dispR = stereoR.compute(grayR, grayL)
    #dispL= np.int16(dispL)
    #dispR= np.int16(dispR)


    #Using WLS Filter
    #filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    #filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta = 0, alpha=255, norm_type = cv2.NORM_MINMAX)
    #filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
    #disp= ((disp.astype(np.float32)/16)-min_disp)/num_disp  #Calculation allowing us to have 0 for the most distant object able to detect

    disp_raw = stereo.compute(grayL, grayR).astype(np.float32)
    
    #Map Disparity to 3D World
    points_3D = cv2.reprojectImageTo3D(disp_raw, Q)

    # Resize the image for faster executions
    #dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

    #Filtering the results with a closing filter
    #closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)

    #Colors map
    #dispc = (closing-closing.min())*255
    #dispC = dispc.astype(np.uint8)
    #dispColor = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)
    #filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

    #Result for Depth_image
    #cv2.imshow("Filtered Color Depth", filt_Color)

    results = model(frameL)[0]
    #results = model.predict(stream=True, imgsz=512)

    detections = sv.Detections.from_ultralytics(results)

   # Camera calibration -> so we do a pre stream to fill the vector arrays or we count the number of frames 
   # we then use these to calibrate the corners to get the 2d points 

    labels = []
    for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        class_name = names[int(class_id)]

        # bounding box width in pixels 
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        bbox_points = points_3D[y1:y2, x1:x2, :]

        #Get average distance using the Z values
        mask = np.isfinite(bbox_points[:, :, 2])
        valid_Z = bbox_points[:, :, 2][mask]
        avg_distance = np.mean(valid_Z)/1000

        confidence_text = f"{confidence:.2f}"
        label_text = f"{class_name} {confidence_text}, {avg_distance:.2f}m"
        labels.append(label_text)

        #Warning for close objects: ALTER LATER FOR VIBRATION SENSOR
        if avg_distance and avg_distance < 1000:
            print(f"[WARNING] {class_name} so close: {avg_distance:.1f}m")

    annotated_frame = box_annotator.annotate(scene=frameL.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    #annotated_dst = box_annotator.annotate(scene= dst.copy(), detections=detections)
    #annotated_dst_img = label_annotator.annotate(scene=annotated_dst, detections=detections, labels=labels)

    cv2.imshow("Object Depth and Detection", annotated_frame)
    #cv2.imshow("Undistorted", annotated_dst)

    if cv2.waitKey(1) == ord('q'):
        break

    for r in results: 
        for c in r.boxes.cls:
            print(names[int(c)])


camL.release()
camR.release()
cv2.destroyAllWindows()

