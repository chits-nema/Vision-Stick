import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import socket
import json
import requests

SERVER_URL = "https://x"  # TODO: replace with your hosts IP!!!!!!
# Filtering
kernel = np.ones((3, 3), np.uint8)


def run_calibration_and_save(output_path: str = "stereo_params.npz"):
    # Camera Calibration
    # Defining the dimensions of checkerboard aka the width and height of the checkerboard

    CHECKERBOARD = (7, 5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpointsR = []
    imgpointsL = []

    # Defining the world coordinates for 3D points
    # objp is the 3d points relating to the world coordinates
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Scale object points up to the sizes of the squares (2 cm -> 20 mm)
    objp *= 34.0

    # Start calibration from the camera
    print('Starting calibration for the 2 cameras... ')
    # Call all saved images
    for i in range(0,
                   64):  # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
        t = str(i)

        # second argument 0 grays the image
        ChessImaR = cv2.imread('src/ServerStuff/chessboard-L' + t + '.png', 0)  # Right side
        ChessImaL = cv2.imread('src/ServerStuff/chessboard-R' + t + '.png', 0)  # Left side

        print("ChessImaR:", type(ChessImaR), ChessImaR.shape if ChessImaR is not None else "None")

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        # corners is the output array of the detetcted corners
        # CHECKERBOARD == patternSize - Number of inner corners per a chessboard row and column
        # if code doesn't work remove flags and set to none, just keeping flags rn to maintain robustness

        retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                                   CHECKERBOARD,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)  # Define the number of chees corners we are looking for
        retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                                   CHECKERBOARD,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)  # Left side
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
            # takes in the OG image and the location of the corners(of the checkerboards)
            cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)
            imgpointsL.append(cornersL)

        cv2.waitKey(0)

    cv2.destroyAllWindows()

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """

    # ret = true if calibrated
    # mtx -> intrinsic camera matrix (focal length fx, fy; optical center cx, cy)
    # dist -> distortion coefficients of the lens
    # rvecs -> rotation extrinsics for each image
    # tvecs -> translation extrinsics for each image


    # Determine the new values for different parameters
    #   Right Side
    hR, wR = ChessImaR.shape[:2]
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                            imgpointsR,
                                                            (wR, hR), None, None)
    OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
                                                (wR, hR), 1, (wR, hR))

    #   Left Side
    hL, wL = ChessImaL.shape[:2]
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                            imgpointsL,
                                                            (wL, hL), None, None)
    OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

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

    # ********************************************
    # ***** Calibrate the Cameras for Stereo *****
    # ********************************************

    # StereoCalibrate function
    # flags = 0
    # flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5
    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                               imgpointsL,
                                                               imgpointsR,
                                                               mtxL,
                                                               distL,
                                                               mtxR,
                                                               distR,
                                                               (wR, hR),
                                                               criteria=criteria_stereo,
                                                               flags=cv2.CALIB_FIX_INTRINSIC)

    # StereoRectify function
    rectify_scale = 1  # if 0 image croped, if 1 image nor croped
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                      (wR, hR), R, T,
                                                      rectify_scale, (0, 0))

    B_estimated = -1 / Q[3, 2]
    print("Baseline according to Q matrix:", B_estimated)
    fx = mtxR[0, 0]
    fy = mtxR[1, 1]
    f_q = Q[2, 3]
    print("Focal length according to mtx:", fx, fy)
    print("Focal length according to Q matrix:", f_q)

    # initUndistortRectifyMap function
    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                  (wL, hL),
                                                  cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                   (wR, hR), cv2.CV_16SC2)

    # ---------- SAVE calibration results to .npz ----------
    np.savez(
        'stereo_params.npz',
        mtxR=mtxR, mtxL=mtxL, distR=distR, distL=distL,
        RL=RL, RR=RR, PL=PL, PR=PR, Q=Q,
        left_map1=Left_Stereo_Map[0], left_map2=Left_Stereo_Map[1],
        right_map1=Right_Stereo_Map[0], right_map2=Right_Stereo_Map[1],
        wR=wR, hR=hR, wL=wL, hL=hL,
        baseline=B_estimated, fx=fx, fy=fy, f_q=f_q
    )
    print("[OK] Saved calibration to stereo_params.npz")


if __name__ == "__main__":
    run_calibration_and_save("stereo_params.npz")