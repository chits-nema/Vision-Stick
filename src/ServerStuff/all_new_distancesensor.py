# all_new_distancesensor.py  (Pi 5 friendly)
from gpiozero import DistanceSensor

_left = None
_right = None

# BCM pins — keep these matching your main script
TRIG_L, ECHO_L = 18, 14
TRIG_R, ECHO_R = 24, 15

def setup():
    global _left, _right
    # max_distance in meters; queue_len smooths readings
    _left  = DistanceSensor(echo=ECHO_L, trigger=TRIG_L, max_distance=2.0, queue_len=3)
    _right = DistanceSensor(echo=ECHO_R, trigger=TRIG_R, max_distance=2.0, queue_len=3)

def get_distance(trig_pin, echo_pin):
    # return centimeters; None if not ready
    s = None
    if (trig_pin, echo_pin) == (TRIG_L, ECHO_L):
        s = _left
    elif (trig_pin, echo_pin) == (TRIG_R, ECHO_R):
        s = _right
    if s is None:
        return None
    return s.distance * 100.0  # meters → cm

def cleanup():
    if _left: _left.close()
    if _right: _right.close()
