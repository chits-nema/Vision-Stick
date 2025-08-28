import RPi.GPIO as GPIO
import time

# ---- Pin setup (define constants only) ----
TRIG1 = 23
ECHO1 = 24
TRIG2 = 25
ECHO2 = 8

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(TRIG1, GPIO.OUT)
    GPIO.setup(ECHO1, GPIO.IN)
    GPIO.setup(TRIG2, GPIO.OUT)
    GPIO.setup(ECHO2, GPIO.IN)
    GPIO.output(TRIG1, False)
    GPIO.output(TRIG2, False)
    time.sleep(0.1)  # short settle delay

def get_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    timeout = 0.04
    start_time = time.time()

    # Wait for echo start
    while GPIO.input(echo) == 0:
        pulse_start = time.time()
        if (pulse_start - start_time) > timeout:
            return None

    # Wait for echo end
    while GPIO.input(echo) == 1:
        pulse_end = time.time()
        if (pulse_end - pulse_start) > timeout:
            return None

    pulse_duration = pulse_end - pulse_start
    return round(pulse_duration * 17150, 2)

def cleanup():
    GPIO.cleanup()

# ---- Run test loop only if executed directly ----
if __name__ == "__main__":
    try:
        setup()
        while True:
            d1 = get_distance(TRIG1, ECHO1)
            d2 = get_distance(TRIG2, ECHO2)
            print(f"Sensor 1: {d1} cm | Sensor 2: {d2} cm")
            time.sleep(1)
    finally:
        cleanup()
