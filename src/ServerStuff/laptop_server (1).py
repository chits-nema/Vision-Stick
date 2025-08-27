import os
import ssl
import cv2
import time
import base64
import signal
import threading
import numpy as np
from queue import Queue
from typing import Dict, Any, List, Tuple, Optional
from flask import Flask, request, jsonify

import supervision as sv
from ultralytics import YOLO

import stereo_runtime_module as stereo
import calibration_module as calib

# =======================
# Display Manager (small always-on-top preview)
# =======================
class DisplayManager:
    """
    Small, always-on-top window that shows the most recent (composited) frame.
    Runs in its own thread and never blocks Flask request handling.
    """
    def __init__(self, window_name="Pi Stereo Stream", width=640, always_on_top=True):
        self.window_name = window_name
        self.width = int(width)
        self.always_on_top = bool(always_on_top)
        self._q = Queue(maxsize=1)   # keep only the latest frame
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
                elif key == ord('1'):  # toggle always-on-top
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
# YOLO Processor
# =======================
"""class YOLOProcessor:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # performance
        self.frame_count_total = 0
        self.sum_processing_total = 0.0
        self.start_time = time.time()

        # serialize GPU work across concurrent Flask threads
        self._lock = threading.Lock()

        print(f"[INFO] YOLO model loaded: {model_path} ({len(self.names)} classes)")

    @staticmethod
    def decode_b64_jpeg(b64_str: str) -> Optional[np.ndarray]:
        try:
            buf = base64.b64decode(b64_str)
            arr = np.frombuffer(buf, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _annotate(self, frame_bgr: np.ndarray, detections: sv.Detections) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        labels = [
            f"{self.names[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]
        img = self.box_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
        img = self.label_annotator.annotate(scene=img, detections=detections, labels=labels)

        det_info = []
        for cid, conf, bbox in zip(detections.class_id, detections.confidence, detections.xyxy):
            det_info.append({
                "class": self.names[cid],
                "confidence": float(conf),
                "bbox": [float(x) for x in bbox.tolist()]  # xyxy
            })
        return img, det_info

    def process_one(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        try:
            with self._lock:
                result = self.model(frame_bgr)[0]
            detections = sv.Detections.from_ultralytics(result)
            annotated, det_info = self._annotate(frame_bgr, detections)

            dt = time.time() - t0
            self.sum_processing_total += dt
            self.frame_count_total += 1

            return annotated, {
                "detections": det_info,
                "processing_time": dt
            }
        except Exception as e:
            return frame_bgr, {"error": str(e)}

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
"""

class StereoVisionProcessor:
    def __init__(self, params_path="stereo_params.npz", model_path="yolo11n.pt"):
        calib.run_calibration_and_save(output_path=params_path)
        stereo._build_matchers()
        stereo.init_runtime(params_path=params_path, model_path=model_path)

        #Performance Tracking
        self.frame_count_total = 0
        self.sum_processing_total = 0.0
        self.start_time = time.time()

        #Seralize GPU work across concurrent Flask threads
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
def hstack_pad(l: np.ndarray, r: np.ndarray, pad: int = 6, pad_color=(30, 30, 30)) -> np.ndarray:
    """Side-by-side composition with a thin separator."""
    h = max(l.shape[0], r.shape[0])
    def fit_h(x):
        if x.shape[0] == h:
            return x
        scale = h / float(x.shape[0])
        w = int(x.shape[1] * scale)
        return cv2.resize(x, (w, h))
    l2 = fit_h(l)
    r2 = fit_h(r)
    sep = np.full((h, pad, 3), pad_color, dtype=np.uint8)
    return np.hstack([l2, sep, r2])

def mono_or_stereo_preview(imgL: Optional[np.ndarray], imgR: Optional[np.ndarray]) -> np.ndarray:
    if imgL is not None and imgR is not None:
        return hstack_pad(imgL, imgR, pad=8)
    return imgL if imgL is not None else imgR

# =======================
# Flask App
# =======================
app = Flask(__name__)

processor = StereoVisionProcessor()
display = DisplayManager(window_name="YOLO Stereo Preview", width=640, always_on_top=True)
display.start()

@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    Accepts:
      - Stereo JSON: {
          "frameL": "<b64 JPEG>", "frameR": "<b64 JPEG>",
          "shapeL": [h, w], "shapeR": [h, w],
          "timestamp": <float>
        }
      - OR Mono JSON: { "frame": "<b64 JPEG>", "timestamp": <float> }
    Returns detections/timings for left/right (if present).
    Also updates a small preview window with the latest annotated frame(s).
    """

    global L, R
    try:
        data = request.get_json(force=True, silent=False)

        frameL_b64 = data.get("frameL")
        frameR_b64 = data.get("frameR")
        mono_b64   = data.get("frame")

        annL = annR = None
        resL: Dict[str, Any] = {}
        resR: Dict[str, Any] = {}

        if frameL_b64 or frameR_b64:  # stereo path
            if frameL_b64:
                L = processor.decode_b64_jpeg(frameL_b64)
                if L is None:
                    return jsonify({"error": "Failed to decode frameL"}), 400
                annL, resL = processor.process_one(L)
            if frameR_b64:
                R = processor.decode_b64_jpeg(frameR_b64)
                if R is None:
                    return jsonify({"error": "Failed to decode frameR"}), 400
                annR, resR = processor.process_one(R)

            preview = mono_or_stereo_preview(annL, annR)
            if preview is not None:
                display.show(preview)

            return jsonify({
                "mode": "stereo",
                "left": resL if frameL_b64 else None,
                "right": resR if frameR_b64 else None,
            }), 200

        elif mono_b64:  # mono path (backward compatible)
            M = processor.decode_b64_jpeg(mono_b64)
            if M is None:
                return jsonify({"error": "Failed to decode frame"}), 400
            annM, resM = processor.process_one(M)
            display.show(annM)
            return jsonify({
                "mode": "mono",
                "result": resM
            }), 200

        else:
            return jsonify({"error": "No frame payload provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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


# =======================
# TLS (self-signed)
# =======================
def create_ssl_context() -> ssl.SSLContext:
    """
    Create HTTPS context. If server.crt/key not present, generate a self-signed one.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    crt = "server.crt"
    key = "server.key"

    if not (os.path.exists(crt) and os.path.exists(key)):
        print("[INFO] SSL certificates not found. Generating self-signed cert (requires openssl)…")
        _generate_self_signed_cert(crt, key)

    context.load_cert_chain(crt, key)
    return context

def _generate_self_signed_cert(crt_path: str, key_path: str):
    import subprocess
    subj = "/C=DE/ST=BW/L=Karlsruhe/O=PiStereo/OU=Video/CN=localhost"
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes",
        "-out", crt_path, "-keyout", key_path, "-days", "365",
        "-subj", subj
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("[INFO] Self-signed certificate generated.")
    except Exception as e:
        print(f"[WARN] Could not auto-generate certs: {e}")
        print("      Provide server.crt and server.key manually or install OpenSSL.")


# =======================
# Main
# =======================
def _handle_sig(*_):
    try:
        display.stop()
    except Exception:
        pass
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

    # threaded=True allows overlapping requests; GPU work is serialized via a lock
    app.run(
        host="0.0.0.0",
        port=8443,
        ssl_context=ssl_context,
        threaded=True,
        debug=False
    )