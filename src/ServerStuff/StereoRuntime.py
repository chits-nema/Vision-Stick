import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

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
    min_disp=16,
    num_disp=240,
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
    global stereo, stereoR, wls_filter

    p = _DEFAULT_SGBM.copy()
    if sgbm_params:
        p.update(sgbm_params)

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
    This function no longer re-creates SGBM/WLS/YOLO on every call.
    Returns a payload dict with serializable arrays + primary bbox.
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

    # YOLO on LEFT frame (matches annotated_image)
    results = model(frameL)[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = []
    det_summaries = [] 
    primary_bbox = None
    primary_distance_m = None
    shortest_distance = 10000000000

    relevant_objects = ["person", "bicycle", "car", "motorcycle", "fire hydrant", "stop sign", "chair", "bus", "traffic light"]

    win_size = 5
    for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        class_name = names[int(class_id)]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # local window around center
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
            d_this = None
        else:
            dists_vals = np.sqrt(Xv**2 + Yv**2 + Zv**2)
            d_this = float(np.median(dists_vals) / 1000.0)  # meters

        # labels for the overlay
        confidence_text = f"{float(confidence):.2f}"
        label_text = f"{class_name} {confidence_text}"
        if d_this is not None:
            label_text += f", {d_this:.2f}m"
        labels.append(label_text)

        if class_name in relevant_objects:
            det_summaries.append({
                "class": class_name,
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "distance_m": d_this
            })

        #  pick primary bbox by largest area
        if d_this < shortest_distance:
            shortest_distance = d_this
            primary_bbox = [float(x1), float(y1), float(x2), float(y2)]
            primary_distance_m = d_this

    # Annotate (boxes + labels)
    annotated_frame = box_annotator.annotate(scene=frameL.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    payload = {
        "annotated_image": annotated_image.tolist(),
        "filtered_image": filteredImg.tolist(),
        "distance_m": primary_distance_m,
        "bbox": primary_bbox,                 
        "detections": det_summaries           
    }
    return payload