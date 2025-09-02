"""
Pi 5 — HTTPS motor control client + dual‑camera frame sender (unified, non‑blocking)
-----------------------------------------------------------------------------------
- Polls **HTTPS GET** on Windows server for vision results (distance_m + bbox) at /vision_receiver.
- Drives two vibration motors with OR logic against local ultrasonics.
- Captures **two Picamera2** streams and **POSTs** them as JPEGs to /process_frame on the same server.
- Frame sending runs in a background thread (throttled), so the motor loop stays responsive.
- All network operations use short timeouts to avoid deadlocks.

Edit the BASE_URL below (e.g. "https://192.168.0.123:8443").
If you use a self‑signed cert for testing, set VERIFY_TLS=False.
"""

from __future__ import annotations

import json
import math
import ssl
import time
from time import sleep
from typing import Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from gpiozero import PWMOutputDevice
from picamera2 import Picamera2
import all_new_distancesensor
import sendClient

# ---------------------------- Server / HTTPS config ----------------------------
BASE_URL = "https://x.x.x.x/8443"   # TODO: CHANGE TO YOUR HOST'S IP!!!!
VISION_GET_PATH = "/vision_receiver"     # GET distance/bbox from this route :)
FRAMES_POST_PATH = "/process_frame"      # POST stereo frames here!
VERIFY_TLS = False                         # True if server cert is trusted; False for self‑signed (dev only)...

VISION_URL = BASE_URL.rstrip("/") + VISION_GET_PATH
FRAMES_URL = BASE_URL.rstrip("/") + FRAMES_POST_PATH

VISION_POLL_PERIOD_S = 0.10   # 10 Hz polling for metadata. Kept low to avoid server overload
VISION_FRESHNESS_S = 0.75     # the max age to accept a reading
HTTP_TIMEOUT_S = 0.40         # keep tight to avoid blocking and "stalling"
DEFAULT_FRAME_W = 640         # fallback if server doesn't send frame_w! (null) D:
VISION_TRIGGER_CM = 200.0      # trigger if camera says <= this (OR with ultrasonics)

# ---------------------------- Pins / control ----------------------------------
MOTOR_L_PIN, MOTOR_R_PIN = 13, 12
TRIG_L, ECHO_L = 18, 14
TRIG_R, ECHO_R = 24, 15
PWM_HZ = 100
MIN_CM, MAX_CM = 5.0, 400.0
MIN_DUTY = 0.01
SYNCED = True
MAX_DUTY_NEAR = 0.25
FAR_DUTY_FLOOR = 0.06 # Noticeable baseline even when far

# ---------------------------- Bias behavior -----------------------------------
BIAS_MIN = 0.5
BIAS_MAX = 1.5
BIAS_L = 1.0
BIAS_R = 1.0
CENTER_DEAD_ZONE_FRAC = 0.05
MAX_REASONABLE_FRAME_W = 10000

# ---------------------------- Globals (updated live) ---------------------------

_last_poll_ts = 0.0
_last_seen_ts = 0.0
_last_vision_cm: Optional[float] = None
_last_bias: Tuple[float, float] = (1.0, 1.0)

_last_bbox: Optional[Tuple[float, float, float, float]] = None
_last_frame_w: float = float(DEFAULT_FRAME_W)

# ---------------------------- Helpers -----------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _safe_float(x) -> Optional[float]:
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def is_valid_cm(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return 0.0 <= v <= 200.0


def map_dist_to_duty(d_cm: float) -> float:
    if d_cm <= MIN_CM:
        return MAX_DUTY_NEAR
    if d_cm >= MAX_CM:
        return 0.0
    # Linear mapping in [MIN_CM, MAX_CM], because we had complaints that the decay was too fast
    t = (d_cm -  MIN_CM) / (MAX_CM - MIN_CM) # Number in [0, 1] within detection range
    # This is a linear mapping that decays much slower so our vision detection is more useful :)
    duty = FAR_DUTY_FLOOR + (MAX_DUTY_NEAR - FAR_DUTY_FLOOR) * (1.0 - t)
    return max(0.0, duty)


def post_modulate(duty_l: float, duty_r: float) -> tuple[float, float]:
    global BIAS_L, BIAS_R
    if SYNCED:
        common = max(duty_l, duty_r)
        duty_l, duty_r = common, common
    duty_l *= BIAS_L
    duty_r *= BIAS_R
    return clamp(duty_l, 0.0, 1.0), clamp(duty_r, 0.0, 1.0)


def apply_deadzone(duty: float) -> float:
    if duty == 0.0:
        return 0.0
    return max(duty, MIN_DUTY)

# ---------------------------- Ultrasonic --------------------------------------

def safe_get_ultra_cm(trig_pin: int, echo_pin: int) -> Optional[float]:
    try:
        d = all_new_distancesensor.get_distance(trig_pin, echo_pin)
    except Exception:
        return None
    return float(d) if is_valid_cm(d) else None

# ---------------------------- HTTPS helpers -----------------------------------

def _http_get_json(url: str, timeout: float, verify_tls: bool) -> Optional[bytes]:
    # urllib: need custom SSL context to toggle verification
    if verify_tls:
        ctx = ssl.create_default_context()
    else:
        ctx = ssl._create_unverified_context()
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except (HTTPError, URLError, TimeoutError):
        return None
    except Exception:
        return None

# ---------------------------- Vision JSON parser ------------------------------

def _parse_vision_json(raw: bytes) -> Tuple[Optional[float], Tuple[float, float]]:
    """Return (distance_cm|None, (bias_l, bias_r)).
    Also updates module globals _last_bbox and _last_frame_w for printing elsewhere.
    Ignores y; bias from bbox x-center only.
    """
    # --- new globals to remember the last bbox + frame width (no API changes) ---
    global _last_bbox, _last_frame_w
    try:
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        if not isinstance(data, dict):
            return None, (1.0, 1.0)
    except Exception:
        return None, (1.0, 1.0)

    # distance m -> cm
    d_cm = None
    dm_f = _safe_float(data.get("distance_m"))
    if dm_f is not None:
        candidate = dm_f * 100.0
        d_cm = candidate if is_valid_cm(candidate) else None

    # frame width (fallback to default)
    frame_w = _safe_float(data.get("frame_w"))
    if frame_w is None or frame_w <= 0 or frame_w > MAX_REASONABLE_FRAME_W:
        frame_w = float(DEFAULT_FRAME_W)
    _last_frame_w = frame_w  # remember for external printing

    # bbox parsing (x1,y1,x2,y2) or (x1,x2)
    _last_bbox = None
    x1 = y1 = x2 = y2 = None
    bbox = data.get("bbox")
    if isinstance(bbox, (list, tuple)):
        if len(bbox) == 4:
            x1 = _safe_float(bbox[0]); y1 = _safe_float(bbox[1])
            x2 = _safe_float(bbox[2]); y2 = _safe_float(bbox[3])
        elif len(bbox) == 2:  # (x1, x2) only
            x1 = _safe_float(bbox[0]); x2 = _safe_float(bbox[1])
    if x1 is not None and x2 is not None:
        if x2 < x1:
            x1, x2 = x2, x1
        _last_bbox = (x1, y1, x2, y2)

    # ---- Dynamic bias from horizontal error (normalized, then clamped) ----
    bias = (1.0, 1.0)
    if x1 is not None and x2 is not None:
        w = x2 - x1
        if 0 < w <= 2 * frame_w:
            cx = (x1 + x2) / 2.0
            cx = clamp(cx, 0.0, frame_w)

            # dead-zone around image center
            dead = CENTER_DEAD_ZONE_FRAC * frame_w
            center = frame_w / 2.0
            if abs(cx - center) <= dead:
                bias = (1.0, 1.0)
            else:
                # normalized signed error in [-1, 1]: + means target is left of center!
                err = (center - cx) / center
                # gain limits the swing you can choose 0.25 for +/-25% before clamping
                gain = 0.25
                bias_l = 1.0 + gain * err
                bias_r = 1.0 - gain * err
                bias = (
                    clamp(bias_l, BIAS_MIN, BIAS_MAX),
                    clamp(bias_r, BIAS_MIN, BIAS_MAX)
                )

    return d_cm, bias

# ---------------------------- Vision polling cache ----------------------------

def poll_vision_if_due():
    # Poll vision JSON and if due update globals.
    global _last_poll_ts, _last_seen_ts, _last_vision_cm, _last_bias
    now = time.time()
    if (now - _last_poll_ts) < VISION_POLL_PERIOD_S:
        return
    _last_poll_ts = now

    raw = _http_get_json(VISION_URL, timeout=HTTP_TIMEOUT_S, verify_tls=VERIFY_TLS)
    if raw is None:
        return
    d_cm, bias = _parse_vision_json(raw)
    _last_vision_cm = d_cm
    _last_bias = bias
    _last_seen_ts = now


def snapshot_vision() -> Tuple[Optional[float], Tuple[float, float]]:
    # Return the last-seen vision distance_cm (or None if stale) and the last bias (always).
    age = time.time() - _last_seen_ts
    if age > VISION_FRESHNESS_S:
        return None, (1.0, 1.0)
    return _last_vision_cm, _last_bias

# ---------------------------- Main control loop --------------------------------

def main():
    global BIAS_L, BIAS_R

    # Start stereo frame sender (thread in the background).
    sender = sendClient.StereoFrameSender.get(base_url=BASE_URL, verify_tls=False, fps=6.0, http_timeout_s=0.2)
    sender.start()

    # Setup ultrasonics + motors
    all_new_distancesensor.setup()
    motor_l = PWMOutputDevice(MOTOR_L_PIN, frequency=PWM_HZ, initial_value=0.0)
    motor_r = PWMOutputDevice(MOTOR_R_PIN, frequency=PWM_HZ, initial_value=0.0)

    try:
        while True:
            # Poll vision metadata (short, non‑blocking)
            poll_vision_if_due()
            vision_cm, vision_bias = snapshot_vision()

            # Apply bias
            BIAS_L, BIAS_R = vision_bias

            # Read ultrasonics
            d_l = safe_get_ultra_cm(TRIG_L, ECHO_L)
            d_r = safe_get_ultra_cm(TRIG_R, ECHO_R)

            # OR logic: either source can trigger vibration! We use the higher duty cycle, i.e. most urgent obstacles!!!
            def decide_duty(ultra_cm: Optional[float]) -> float:
                duty_ultra = map_dist_to_duty(ultra_cm) if ultra_cm is not None else 0.0
                duty_vis = 0.0
                if vision_cm is not None and vision_cm <= VISION_TRIGGER_CM:
                    duty_vis = map_dist_to_duty(VISION_TRIGGER_CM)
                return max(duty_ultra, duty_vis)

            duty_l = decide_duty(d_l)
            duty_r = decide_duty(d_r)
            duty_l, duty_r = post_modulate(duty_l, duty_r)

            motor_l.value = apply_deadzone(duty_l)
            motor_r.value = apply_deadzone(duty_r)
            # ---- Print status ----
            # Debug
            def fmt(v):
                return f"{v:5.1f}" if is_valid_cm(v) else " --.-"

            def fmt_bbox(bb):
                if not bb:
                    return "(None)"
                x1, y1, x2, y2 = bb

                def f(a):
                    return "--" if a is None else f"{a:.1f}"

                return f"(x1={f(x1)}, y1={f(y1)}, x2={f(x2)}, y2={f(y2)})"

            print(
                f"UL(L={fmt(d_l)} cm, R={fmt(d_r)} cm) | "
                f"VISION={fmt(vision_cm)} cm | "
                f"BIAS(L={BIAS_L:.2f}, R={BIAS_R:.2f}) | "
                f"PWM(L={motor_l.value:.3f}, R={motor_r.value:.3f}) | "
                f"BBOX={fmt_bbox(_last_bbox)} FW={_last_frame_w:.0f}"
            )
            sleep(0.05)
    finally:
        sender.stop()
        motor_l.off(); motor_r.off()
        all_new_distancesensor.cleanup()


if __name__ == "__main__":
    main()
