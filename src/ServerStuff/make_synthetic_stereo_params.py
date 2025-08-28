# make_synthetic_stereo_params.py
import numpy as np
import cv2

# ---- Choose a test image size you'll use for frames ----
w, h = 640, 480

# ---- Synthetic intrinsics (reasonable defaults) ----
fx = fy = 700.0          # focal length [px]
cx, cy = w/2.0, h/2.0    # principal point at image center
mtxL = np.array([[fx, 0, cx],
                 [0, fy, cy],
                 [0,  0,  1]], dtype=np.float64)
mtxR = mtxL.copy()
distL = np.zeros((5, 1), dtype=np.float64)
distR = np.zeros((5, 1), dtype=np.float64)

# ---- Synthetic extrinsics: rectified pair with a baseline along X ----
baseline_m = 0.12  # 12 cm, arbitrary but plausible
R = np.eye(3, dtype=np.float64)
T = np.array([[-baseline_m, 0.0, 0.0]], dtype=np.float64).T  # left->right translation

# ---- Stereo rectification (this gives R1,R2,P1,P2,Q) ----
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1.0
)

# ---- Undistort/rectify maps (what your runtime loads) ----
Left_Stereo_Map  = cv2.initUndistortRectifyMap(mtxL, distL, RL, PL, (w, h), cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(mtxR, distR, RR, PR, (w, h), cv2.CV_16SC2)

# ---- Optional extras for completeness / debugging ----
B_estimated = -1.0 / Q[3, 2]  # should be close to baseline_m * fx normalization
print("Saved synthetic stereo params:",
      f"\n  size = {w}x{h}",
      f"\n  fx = {fx}, fy = {fy}, cx = {cx}, cy = {cy}",
      f"\n  baseline = {baseline_m} m",
      f"\n  Q[3,2] = {Q[3,2]:.6f}, baseline_from_Q ~= {B_estimated:.6f}", sep="")

# ---- Save exactly the keys your code expects ----
np.savez(
    "stereo_params.npz",
    # intrinsics & distortion (nice to have)
    mtxR=mtxR, mtxL=mtxL, distR=distR, distL=distL,
    # rectification components
    RL=RL, RR=RR, PL=PL, PR=PR, Q=Q,
    # undistort/rectify maps (these are required by your runtime)
    left_map1=Left_Stereo_Map[0], left_map2=Left_Stereo_Map[1],
    right_map1=Right_Stereo_Map[0], right_map2=Right_Stereo_Map[1],
    # sizes/extras
    wR=w, hR=h, wL=w, hL=h, baseline=B_estimated, fx=fx, fy=fy, f_q=Q[2,3]
)
