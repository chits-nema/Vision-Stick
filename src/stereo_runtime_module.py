import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import requests

# ----------------- module-level globals (initialized in init_runtime) -----------------
Left_Stereo_Map = None      # (map1, map2) for left
Right_Stereo_Map = None     # (map1, map2) for right
Q = None                    # 4x4 reprojection matrix
stereo = None               # cv2.StereoSGBM
stereoR = None              # right matcher
wls_filter = None           # WLS filter
model = None                # YOLO model
box_annotator = None
label_annotator = None
names = None

# Default Step-3/4 params
_DEFAULT_SGBM = dict(
    window_size=3,
    min_disp=39,
    num_disp=128,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=5
)
_DEFAULT_WLS = dict(
    lmbda=80000,
    sigma=1.8
)

def _build_matchers(sgbm_params=None, wls_params=None):
    """Build SGBM, right matcher and WLS once (called from init_runtime)."""
    global stereo, stereoR, wls_filter

    p = _DEFAULT_SGBM.copy()
    if sgbm_params:
        p.update(sgbm_params)

    # Create SGBM
    stereo_local = cv2.StereoSGBM_create(
        minDisparity=p["min_disp"],
        numDisparities=p["num_disp"],
        blockSize=p["window_size"],
        uniquenessRatio=p["uniquenessRatio"],
        speckleWindowSize=p["speckleWindowSize"],
        speckleRange=p["speckleRange"],
        disp12MaxDiff=p["disp12MaxDiff"],
        P1=8*3*p["window_size"]**2,
        P2=32*3*p["window_size"]**2
    )
    stereoR_local = cv2.ximgproc.createRightMatcher(stereo_local)

    w = _DEFAULT_WLS.copy()
    if wls_params:
        w.update(wls_params)

    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_local)
    wls.setLambda(w["lmbda"])
    wls.setSigmaColor(w["sigma"])

    stereo = stereo_local
    stereoR = stereoR_local
    wls_filter = wls

def init_runtime(params_path="stereo_params.npz", model_path="yolo11n.pt",
                 sgbm_params=None, wls_params=None):
    """
    Load maps/Q from the calibration .npz and build SGBM/WLS and YOLO once.
    Optionally override SGBM/WLS params by passing dicts; otherwise defaults are used.
    """
    global Left_Stereo_Map, Right_Stereo_Map, Q
    global model, box_annotator, label_annotator, names

    print(f"[INFO] Loading saved calibration: {params_path}")
    data = np.load(params_path)

    Q = data["Q"]
    Left_Stereo_Map  = (data["left_map1"],  data["left_map2"])
    Right_Stereo_Map = (data["right_map1"], data["right_map2"])

    # Build SGBM/WLS once
    _build_matchers(sgbm_params=sgbm_params, wls_params=wls_params)

    # YOLO & annotators once
    model = YOLO(model_path)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    names = model.names

    print("[INFO] Runtime initialized (maps/Q loaded, SGBM/WLS/YOLO ready).")

def detect_stereo_vision(frameL, frameR):
    """
    IMPORTANT: This function no longer re-creates SGBM/WLS/YOLO on every call.
    Returns: annotated_image, filtered_disparity, distance_m
    """
    assert Left_Stereo_Map is not None and Right_Stereo_Map is not None and Q is not None, \
        "Call init_runtime(...) before detect_stereo_vision(...)"

    # Rectify frames using precomputed maps
    left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1],
                          interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1],
                           interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    # To gray
    grayR = cv2.cvtColor(right_nice, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(left_nice,  cv2.COLOR_BGR2GRAY)

    # Disparities
    disp  = stereo.compute(grayL, grayR).astype(np.float32) / 16
    dispL = np.int16(disp)
    dispR = np.int16(stereoR.compute(grayR, grayL))

    # WLS filter
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)

    # 3D reprojection
    points_3D = cv2.reprojectImageTo3D(filteredImg, Q)

    # YOLO
    results = model(frameL)[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = [] 

    win_size = 5
    for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        class_name = names[int(class_id)]

        # bounding box width in pixels 
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        #Take Centroid of detected bounding box 
        cx = (x1 + x2) // 2 
        cy = (y1 + y2) // 2 
        #point_3D = points_3D[cy, cx, :]

        x_min = max(0, cx - win_size)
        x_max = min(points_3D.shape[1] - 1, cx + win_size)
        y_min = max(0, cy - win_size)
        y_max = min(points_3D.shape[0] - 1, cy + win_size)

        region = points_3D[y_min:y_max, x_min:x_max, :]
        X = region[:, :, 0].flatten()
        Y = region[:, :, 1].flatten()
        Z = region[:, :, 2].flatten()

        valid_mask = np.isfinite(Z)

        Xv, Yv, Zv = X[valid_mask], Y[valid_mask], Z[valid_mask]

        if len(Xv) == 0:
            distance_m = None
        else:
            dists_vals = np.sqrt(Xv**2 + Yv**2 + Zv**2)
            distance_m = np.median(dists_vals) / 1000

        confidence_text = f"{confidence:.2f}"
        label_text = f"{class_name} {confidence_text}, {distance_m:.2f}m"
        labels.append(label_text)

    # Annotate 
    annotated_frame = box_annotator.annotate(scene=frameL.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    payload = {
            "annotated_image": annotated_image.tolist(),
            "filtered_image": filteredImg.tolist(),
            "distance_m": distance_m
        }
    return payload