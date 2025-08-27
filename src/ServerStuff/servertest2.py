import os
import ssl
import cv2
import time
import json
import base64
import signal
import struct
import threading
import numpy as np
from queue import Queue
from flask import Flask, request, jsonify

import supervision as sv
from ultralytics import YOLO

# =======================
# Display Manager (GUI)
# =======================
class DisplayManager:
    """
    Small, always-on-top window that shows the most recent frame.
    Runs in its own thread and never blocks Flask request handling.
    """
    def __init__(self, window_name="YOLO Detection Server", width=480, always_on_top=True):
        self.window_name = window_name
        self.width = width
        self.always_on_top = always_on_top
        self._q = Queue(maxsize=1)  # keep only most recent frame
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def show(self, frame_bgr):
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
            # No GUI available (headless)
            self._running = False
            return

        last_frame = None
        while self._running:
            try:
                if not self._q.empty():
                    last_frame = self._q.get(timeout=0.02)
                if last_frame is not None:
                    h, w = last_frame.shape[:2]
                    scale = self.width / float(w)
                    preview = cv2.resize(last_frame, (self.width, int(h * scale)))
                    cv2.imshow(self.window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
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
# YOLO Processor
# =======================
class YOLOProcessor:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.sum_processing = 0.0
        self._lock = threading.Lock()  # serialize GPU inference if needed

        print(f"[INFO] YOLO model loaded: {model_path}")
        print(f"[INFO] Available classes: {len(self.names)}")

    @staticmethod
    def decode_frame(frame_b64):
        """Decode base64 JPEG into a BGR numpy array."""
        try:
            frame_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"[ERROR] Frame decode error: {e}")
            return None

    def process_frame(self, frame_bgr):
        """Run YOLO, annotate, push to display, and return JSON-able results."""
        t0 = time.time()
        try:
            # serialize inference for safety (shared GPU)
            with self._lock:
                results = self.model(frame_bgr)[0]

            detections = sv.Detections.from_ultralytics(results)
            labels = [
                f"{self.names[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]

            # Draw boxes + labels
            annotated = self.box_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            # Build detection info
            detection_info = []
            for class_id, confidence, bbox in zip(detections.class_id, detections.confidence, detections.xyxy):
                detection_info.append({
                    "class": self.names[class_id],
                    "confidence": float(confidence),
                    "bbox": [float(x) for x in bbox.tolist()]  # xyxy
                })

            # Update perf counters
            processing_time = time.time() - t0
            self.sum_processing += processing_time
            self.frame_count += 1

            return annotated, {
                "detections": detection_info,
                "processing_time": processing_time,
                "total_frames": self.frame_count
            }

        except Exception as e:
            print(f"[ERROR] Processing error: {e}")
            return None, {"error": str(e)}

    def get_stats(self):
        if self.frame_count == 0:
            return {}
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        avg_proc = self.sum_processing / self.frame_count
        return {
            "frames_processed": self.frame_count,
            "average_fps": avg_fps,
            "average_processing_time": avg_proc,
            "elapsed_time": elapsed
        }

# =======================
# Flask App
# =======================
app = Flask(__name__)

# Global components
processor = YOLOProcessor()
display = DisplayManager(window_name="YOLO Detection Server", width=480, always_on_top=True)
display.start()

@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    Accepts JSON: {"frame": "<base64 JPEG>", "shape": [h,w,3], "timestamp": <float>}
    Returns detections & timings. Also shows the annotated frame in a small window.
    """
    try:
        data = request.get_json(force=True, silent=False)
        frame_b64 = data.get("frame")

        if not frame_b64:
            return jsonify({"error": "No frame provided"}), 400

        # Decode
        frame = processor.decode_frame(frame_b64)
        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        # Process
        annotated, result = processor.process_frame(frame)
        if annotated is not None:
            # push to preview window (non-blocking)
            display.show(annotated)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(processor.get_stats())

@app.route("/health", methods=["GET"])
def health():
    gpu_info = "CPU only"
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        gpu_info = [f"cuda:{i}" for i in range(gpu_count)] if gpu_count > 0 else "CPU only"
    except Exception:
        pass
    return jsonify({"status": "healthy", "model_loaded": True, "opencv_gpu": gpu_info})

# =======================
# TLS (self-signed)
# =======================
def create_ssl_context():
    """
    Create HTTPS context. If server.crt/key not present, generate a self-signed one.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    crt = "server.crt"
    key = "server.key"

    if not (os.path.exists(crt) and os.path.exists(key)):
        print("[INFO] SSL certificates not found. Generating self-signed cert (requires openssl)...")
        generate_self_signed_cert(crt, key)

    context.load_cert_chain(crt, key)
    return context

def generate_self_signed_cert(crt_path, key_path):
    import subprocess
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes",
        "-out", crt_path, "-keyout", key_path, "-days", "365",
        "-subj", "/C=DE/ST=BW/L=Karlsruhe/O=RaspberryPi/OU=Video/CN=localhost"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("[INFO] Self-signed certificate generated.")
    except Exception as e:
        print(f"[WARN] Failed to generate certificate automatically: {e}")
        print("       Please install OpenSSL or provide server.crt and server.key manually.")

# =======================
# Graceful shutdown
# =======================
_shutdown = threading.Event()

def _handle_sig(*_):
    _shutdown.set()
    try:
        display.stop()
    except Exception:
        pass
    # Flask dev server will exit on next loop

# =======================
# Main
# =======================
if __name__ == "__main__":
    print("[INFO] Starting YOLO Processing Server…")
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        print("[INFO] OpenCV CUDA devices:", [f"cuda:{i}" for i in range(gpu_count)] if gpu_count > 0 else "CPU only")
    except Exception:
        print("[INFO] OpenCV CUDA devices: unavailable (OpenCV not built with CUDA)")

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    ssl_context = create_ssl_context()

    # Important: threaded=True allows concurrent requests,
    # but GPU inference is serialized by processor._lock.
    app.run(
        host="0.0.0.0",
        port=8443,
        ssl_context=ssl_context,
        threaded=True,
        debug=False
    )