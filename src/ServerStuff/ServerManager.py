import os
import ssl
import cv2
import time
import base64
import signal
import threading
import numpy as np
from queue import Queue
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify

import StereoRuntime as stereo
import CalibrationModule as calib


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


class StereoVisionProcessor:
    def __init__(self, params_path="stereo_params.npz", model_path="yolo11n.pt"):
        calib.run_calibration_and_save(output_path=params_path)
        stereo._build_matchers()
        stereo.init_runtime(params_path=params_path, model_path=model_path)

        # Performance Tracking
        self.frame_count_total = 0
        self.sum_processing_total = 0.0
        self.start_time = time.time()

        # Serialize GPU work across concurrent Flask threads
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
# StereoRuntime returns a single annotated left image; keep a single-window preview
display = DisplayManager(window_name="YOLO Stereo Preview", width=640, always_on_top=True)
display.start()


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    Accepts stereo JSON: {"frameL": "<b64 JPEG>", "frameR": "<b64 JPEG>"}

    Minimal fix: use StereoVisionProcessor.process_stereo and forward its
    detections + distance back to the client (no YOLO-only paths).
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

        # Show the annotated image returned by StereoRuntime
        try:
            ann = np.array(payload.get("annotated_image"), dtype=np.uint8)
            display.show(ann)
        except Exception:
            pass

        # Forward detections + distance_m for biasing logic (client keeps the contract)
        return jsonify({
            "mode": "stereo",
            "left": {
                "processing_time": payload.get("processing_time"),
                "distance_m": payload.get("distance_m"),
                "detections": payload.get("detections", [])
            },
            "right": None
        }), 200

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
    Create HTTPS context. If server.crt/key not present, print a hint.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    crt = "server.crt"
    key = "server.key"

    if not (os.path.exists(crt) and os.path.exists(key)):
        print("[INFO] SSL certificates not found. Provide server.crt and server.key or install OpenSSL.")

    context.load_cert_chain(crt, key)
    return context


# =======================
# Main
# =======================

def _handle_sig(*_):
    try:
        display.stop()
    except Exception:
        pass


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