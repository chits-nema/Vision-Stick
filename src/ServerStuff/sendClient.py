from __future__ import annotations

import base64
import time
from typing import Optional
import threading

import cv2
import requests
from picamera2 import Picamera2

# Defaults 
_BASE_URL = "https://192.168.0.123:8443" # HOST'S IP!!!!
_POST_PATH = "/process_frame"
_VERIFY_TLS = False           # set True if trusted cert available cert
_TARGET_WIDTH = 640
_FPS = 8.0                    # tick() will send at most this often
_JPEG_QUALITY = 80
_HTTP_TIMEOUT_S = 0.25        


class StereoFrameSender:
    """Stereo camera sender module :D
    very nice because we use threads. This takes advatange of the multiple cores
    of the ARM Cortex A76 CPU in the Raspberry Pi 5.
    """

    _instance: Optional["StereoFrameSender"] = None

    @classmethod
    # Class definition!!!!!!!!!
    def get(
        cls,
        *,
        base_url: str | None = None,
        post_path: str | None = None,
        verify_tls: Optional[bool] = None,
        target_width: Optional[int] = None,
        fps: Optional[float] = None,
        jpeg_quality: Optional[int] = None,
        http_timeout_s: Optional[float] = None,
    ) -> "StereoFrameSender":
        if cls._instance is None:
            cls._instance = cls(
                base_url or _BASE_URL,
                post_path or _POST_PATH,
                _VERIFY_TLS if verify_tls is None else bool(verify_tls),
                _TARGET_WIDTH if target_width is None else int(target_width),
                _FPS if fps is None else float(fps),
                _JPEG_QUALITY if jpeg_quality is None else int(jpeg_quality),
                _HTTP_TIMEOUT_S if http_timeout_s is None else float(http_timeout_s),
            )
        return cls._instance

    def __init__(
        self,
        base_url: str,
        post_path: str,
        verify_tls: bool,
        target_width: int,
        fps: float,
        jpeg_quality: int,
        http_timeout_s: float,
    ) -> None:
        # config
        self.post_url = base_url.rstrip("/") + post_path
        self.verify_tls = verify_tls
        self.period = 1.0 / max(1e-6, fps)
        self.jpeg_quality = int(jpeg_quality)
        self.target_width = int(target_width)
        self.http_timeout_s = float(http_timeout_s)

        # state
        self._picamL: Optional[Picamera2] = None
        self._picamR: Optional[Picamera2] = None
        self._opened = False
        self._next_t = 0.0

        # thread state
        self._thread: Optional[threading.Thread] = None
        self._stop_evt: Optional[threading.Event] = None

    # ---- lifecycle (non-threaded) ----
    def ensure_open(self) -> bool:
        """Open cameras once; safe to call multiple times."""
        if self._opened:
            return True
        try:
            self._picamL = Picamera2(0)
            self._picamR = Picamera2(1)
            cfg = {"size": (1280, 720), "format": "RGB888"}
            self._picamL.configure(self._picamL.create_preview_configuration(main=cfg))
            self._picamR.configure(self._picamR.create_preview_configuration(main=cfg))
            self._picamL.start(); self._picamR.start()
            self._opened = True
            self._next_t = time.time()  # send immediately
            return True
        except Exception as e:
            print(f"[StereoFrameSender] Camera init failed: {e}")
            self._opened = False
            return False

    def close(self) -> None:
        # To close the cameras
        if not self._opened:
            return
        try:
            if self._picamL: self._picamL.stop()
            if self._picamR: self._picamR.stop()
        except Exception:
            pass
        self._opened = False

    # ---- non-threaded cadence ----
    def tick(self) -> bool:
        """Capture+POST when due; otherwise return quickly.
        Returns True iff a frame pair was sent this call.
        """
        if not self._opened and not self.ensure_open():
            return False

        now = time.time()
        if now < self._next_t:
            return False  # not time yet
        self._next_t = now + self.period

        try:
            l = self._picamL.capture_array()
            r = self._picamR.capture_array()
            if l is None or r is None:
                return False
            b64_l, hw_l = self._encode_jpeg(l, self.target_width, self.jpeg_quality)
            b64_r, hw_r = self._encode_jpeg(r, self.target_width, self.jpeg_quality)
            payload = {
                "frameL": b64_l, "frameR": b64_r,
                "shapeL": [int(hw_l[0]), int(hw_l[1])],
                "shapeR": [int(hw_r[0]), int(hw_r[1])],
                "timestamp": now,
            }
            resp = requests.post(self.post_url, json=payload,
                                 timeout=self.http_timeout_s, verify=self.verify_tls)
            return resp.status_code == 200
        except requests.RequestException:
            return False
        except Exception:
            return False

    # ---- utilities ----
    @staticmethod
    def _encode_jpeg(frame, tw: int, quality: int) -> tuple[str, tuple[int, int]]:
        h, w = frame.shape[:2]
        if w != tw:
            nh = int(h * (tw / float(w)))
            frame = cv2.resize(frame, (tw, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(buf).decode("ascii"), (frame.shape[0], frame.shape[1])

    # ---- optional threaded API ----
    def start(self) -> None:
        """Start background sending loop (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run_thread,
                                        name="StereoFrameSender", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background loop and close cameras."""
        if self._stop_evt:
            self._stop_evt.set()
        t = self._thread
        if t:
            t.join(timeout=1.5)
        self._thread = None
        self._stop_evt = None
        self.close()

    def _run_thread(self) -> None:
        """Loop that reuses tick() and sleeps precisely until the next due send."""
        # Open once (safe if already open)
        self.ensure_open()

        # honor cadence
        while self._stop_evt and not self._stop_evt.is_set():
            now = time.time()
            # If it's time, tick() will send and also push _next_t
            sent = self.tick()
            # Sleep until next frame time (or a tiny nap if not open yet)
            if self._opened:
                # compute remaining time to next target instant
                sleep_for = max(0.0, self._next_t - time.time())
                # wake up a touch early to amortize capture latency
                if sleep_for > 0:
                    self._stop_evt.wait(sleep_for)
            else:
                # camera not opened yet; back off slightly
                self._stop_evt.wait(0.05)
