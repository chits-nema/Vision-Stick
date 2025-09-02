# servermanager.py
import os
import ssl
import sys

import cv2
import time
import base64
import signal
import threading
import numpy as np
from queue import Queue
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify

import StereoRuntime as stereo
import CalibrationModule as calib


# =======================
# Display Manager (small always-on-top preview)
# =======================
class DisplayManager:
    """
    Small, always-on-top window that shows the most recent (annotated) frame.
    Runs in its own thread and never blocks Flask request handling.
    """

    def __init__(self, window_name="Pi Stereo Stream", width=640, always_on_top=True):
        self.window_name = window_name
        self.width = int(width)
        self.always_on_top = bool(always_on_top)
        self._q = Queue(maxsize=1)  # keep only the latest frame
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="DisplayManager", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def show(self, frame_bgr: np.ndarray):
        """Offer a new frame to display (drop previous if queue is full)."""
        if not self._running:
            return
        try:
            if not self._q.empty():
                self._q.get_nowait()
            self._q.put_nowait(frame_bgr)
        except Exception:
            pass

    def _loop(self):
        # try to open a window; if headless, silently stop
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, int(self.width * 9 / 16))
            try:
                cv2.setWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_TOPMOST,
                    1.0 if self.always_on_top else 0.0
                )
            except Exception:
                pass
        except Exception:
            self._running = False
            return

        last = None
        while self._running:
            try:
                if not self._q.empty():
                    last = self._q.get(timeout=0.02)
                if last is not None:
                    h, w = last.shape[:2]
                    scale = self.width / float(w)
                    preview = cv2.resize(last, (self.width, int(h * scale)))
                    cv2.imshow(self.window_name, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # close preview thread
                    self.stop()
                elif key == ord('1'):  
                    self.always_on_top = not self.always_on_top
                    try:
                        cv2.setWindowProperty(
                            self.window_name,
                            cv2.WND_PROP_TOPMOST,
                            1.0 if self.always_on_top else 0.0
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


# =======================
# Vision result cache (served to Pi via /vision_receiver)
# =======================
class VisionCache:
    """
    Thread-safe cache of the latest 'vision result' the Pi will GET from /vision_receiver.
    Fields:
      - ts: server timestamp (float)
      - distance_m: Optional[float] (None if unknown)
      - bbox: Optional[List[float]] = [x1,y1,x2,y2] in pixels
      - frame_w: Optional[int] (width of the frame used for bbox)
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "ts": 0.0,
            "distance_m": None,
            "bbox": None,
            "frame_w": None,
        }

    def update(self, *, distance_m: Optional[float], bbox: Optional[List[float]], frame_w: Optional[int]) -> None:
        with self._lock:
            self._data = {
                "ts": time.time(),
                "distance_m": float(distance_m) if isinstance(distance_m, (int, float)) else None,
                "bbox": [float(x) for x in bbox] if (isinstance(bbox, (list, tuple)) and len(bbox) == 4) else None,
                "frame_w": int(frame_w) if isinstance(frame_w, (int, float)) else None,
            }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)


# =======================
# Stereo Vision Processor
# =======================
class StereoVisionProcessor:
    """
    Wraps calibration + stereo + runtime. The runtime function
    stereo.detect_stereo_vision(frameL, frameR) MUST return a payload dict containing:
      - 'annotated_image' : list (HxWx3, uint8)  -> we display it
      - 'filtered_image'  : list (optional)
      - 'distance_m'      : float|None          -> distance for PRIMARY bbox
      - 'bbox'            : [x1,y1,x2,y2]|None  -> PRIMARY bbox on LEFT frame
      - 'detections'      : list (optional summaries per detection)
    """
    def __init__(self, params_path="stereo_params.npz", model_path="yolo11n.pt"):
        # Prepare calibration + stereo + runtime
        stereo._build_matchers()
        stereo.init_runtime(params_path=params_path, model_path=model_path)

        # Performance Tracking
        self.frame_count_total = 0
        self.sum_processing_total = 0.0
        self.start_time = time.time()

        # Serialize GPU
        self._lock = threading.Lock()

        print(f"[INFO] StereoVisionProcessor initialized with params: {params_path} and model: {model_path}")

    @staticmethod
    def decode_b64_jpeg(b64_str: str) -> Optional[np.ndarray]:
        try:
            buf = base64.b64decode(b64_str)
            arr = np.frombuffer(buf, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def process_stereo(self, frameL: np.ndarray, frameR: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()
        try:
            with self._lock:
                # returns dict with distance_m and bbox
                payload = stereo.detect_stereo_vision(frameL, frameR)
            dt = time.time() - t0
            self.sum_processing_total += dt
            self.frame_count_total += 1
            payload["processing_time"] = dt
            return payload
        except Exception as e:
            return {"error": str(e)}

    def stats(self) -> Dict[str, Any]:
        if self.frame_count_total == 0:
            return {}
        elapsed = time.time() - self.start_time
        return {
            "frames_processed_total": self.frame_count_total,
            "avg_processing_time": self.sum_processing_total / self.frame_count_total,
            "avg_fps_since_start": self.frame_count_total / elapsed if elapsed > 0 else 0.0,
            "elapsed_time_sec": elapsed
        }


# =======================
# Utilities
# =======================
def _tolist_to_np_uint8(img_like) -> Optional[np.ndarray]:
    """Convert a list-based image (H x W x 3) back to np.uint8 BGR if possible."""
    if img_like is None:
        return None
    try:
        arr = np.array(img_like, dtype=np.uint8)
        
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            if arr.shape[2] == 4:
                
                arr = arr[:, :, :3]
            return arr
    except Exception:
        pass
    return None


# =======================
# Flask App
# =======================
app = Flask(__name__)

processor = StereoVisionProcessor()
display = DisplayManager(window_name="YOLO Stereo Preview", width=640, always_on_top=True)
display.start()
vision_cache = VisionCache()


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    Accepts stereo JSON: {"frameL": "<b64 JPEG>", "frameR": "<b64 JPEG>"}.
    Processes frames, updates the preview window, and:
      1) Returns distance + bbox in the HTTP response, and
      2) Updates a server-side cache that the Pi polls via GET /vision_receiver
         (contains distance_m, bbox, and frame_w for bias logic).
    """
    try:
        data = request.get_json(force=True, silent=False)
        frameL_b64 = data.get("frameL")
        frameR_b64 = data.get("frameR")

        if not (frameL_b64 and frameR_b64):
            return jsonify({"error": "Both frameL and frameR required for stereo"}), 400

        L = processor.decode_b64_jpeg(frameL_b64)
        R = processor.decode_b64_jpeg(frameR_b64)
        if L is None or R is None:
            return jsonify({"error": "Failed to decode stereo frames"}), 400

        payload = processor.process_stereo(L, R)
        if "error" in payload:
            return jsonify(payload), 500

        # ----- Preview window -----
        ann_img = _tolist_to_np_uint8(payload.get("annotated_image"))
        if ann_img is not None:
            display.show(ann_img)

        # ----- Distance + primary bbox -----
        distance_m = payload.get("distance_m", None)   # float|None
        primary_bbox = payload.get("bbox", None)       # [x1,y1,x2,y2]|None
        frame_w = int(L.shape[1])

        # Update cache so the Pi can GET /vision_receiver
        vision_cache.update(distance_m=distance_m, bbox=primary_bbox, frame_w=frame_w)

        # Return compact info
        return jsonify({
            "mode": "stereo",
            "left": {
                "processing_time": payload.get("processing_time"),
                "distance_m": distance_m,
                "bbox": primary_bbox
            },
            "right": None,
            "vision_cache": vision_cache.snapshot()
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/vision_receiver", methods=["GET"])
def vision_receiver():
    """
    Pi polls this endpoint. We return a compact JSON containing:
      - distance_m: Optional[float]
      - bbox: Optional[List[float]] = [x1,y1,x2,y2]
      - frame_w: Optional[int]
      - ts: float (server time of this reading)
    """
    return jsonify(vision_cache.snapshot())


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(processor.stats())


@app.route("/health", methods=["GET"])
def health():
    gpu_info = "CPU only"
    try:
        cnt = cv2.cuda.getCudaEnabledDeviceCount()
        gpu_info = [f"cuda:{i}" for i in range(cnt)] if cnt > 0 else "CPU only"
    except Exception:
        pass
    return jsonify({"status": "healthy", "model_loaded": True, "opencv_gpu": gpu_info})


@app.route("/about", methods=["GET"])
def about():
    return jsonify({
        "user": os.getenv("STICK_USER", "unknown"),
        "note": os.getenv("STICK_NOTE", ""),
        "server_time": time.time(),
        "version": "1.0.0"
    })

# =======================
# TLS (self-signed)
# =======================
def create_ssl_context() -> ssl.SSLContext:
    """
    Create HTTPS context. If server.crt/key not present, try to auto-generate a self-signed one.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    crt = "server.crt"
    key = "server.key"

    if not (os.path.exists(crt) and os.path.exists(key)):
        print("[INFO] SSL certificates not found. Generating self-signed cert (requires openssl)…")
        try:
            _generate_self_signed_cert(crt, key)
        except Exception as e:
            print(f"[WARN] Could not auto-generate certs: {e}")
            print("      Provide server.crt and server.key manually or install OpenSSL.")
    context.load_cert_chain(crt, key)
    return context


def _generate_self_signed_cert(crt_path: str, key_path: str):
    import subprocess
    subj = "/C=DE/ST=BW/L=Heilbronn/O=BIE/OU=VisonStick/CN=visonstick.com"
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes",
        "-out", crt_path, "-keyout", key_path, "-days", "365",
        "-subj", subj
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print("[INFO] Self-signed certificate generated.")


# =======================
# Main
# =======================
_shutdown = threading.Event()
def _handle_sig(*_):
    _shutdown.set()
    try:
        display.stop()
    except Exception:
        pass
    sys.exit(0)
    # Flask dev server exits on next loop

if __name__ == "__main__":
    print("[INFO] Starting Stereo YOLO Server…")
    try:
        cnt = cv2.cuda.getCudaEnabledDeviceCount()
        print("[INFO] OpenCV CUDA devices:",
              [f"cuda:{i}" for i in range(cnt)] if cnt > 0 else "CPU only")
    except Exception:
        print("[INFO] OpenCV CUDA devices: unavailable (OpenCV not built with CUDA)")

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    ssl_context = create_ssl_context()

    # GPU work is serialized via a lock
    app.run(
        host="0.0.0.0",
        port=8443,
        ssl_context=ssl_context,
        threaded=True,
        debug=False
    )
