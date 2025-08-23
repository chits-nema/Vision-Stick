from time import sleep
import RPi.GPIO as GPIO
from gpiozero import PWMOutputDevice
import all_new_distancesensor

# ---- Pin map (BCM numbering) ----
MOTOR_L_PIN = 13 # hardware PWM
MOTOR_R_PIN = 12 # hardware PWM

# Left sensor
TRIG_L = 23
ECHO_L = 24

# Right sensor
TRIG_R = 25
ECHO_R = 8

# ---- Tuning ----
PWM_HZ = 100 # 100–300 Hz feels good for coin motors
MIN_CM, MAX_CM = 5.0, 80.0 # map distances to intensity
MIN_DUTY = 0.01 # overcome static friction; tweak per motor

# If True, make both motors share a common duty (synchronized).
# You can change how they sync in post_modulate().
SYNCED = True

# Optional bias to favor one motor (e.g., from CV), 0..1 scale factor
BIAS_L = 1.0
BIAS_R = 1.0

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def map_dist_to_duty(d_cm: float) -> float:
    """Closer -> stronger (1.0). Farther -> weaker (0.0). Linear mapping."""
    if d_cm <= MIN_CM:
        return 1.0
    if d_cm >= MAX_CM:
        return 0.0
    return 1.0 - (d_cm - MIN_CM) / (MAX_CM - MIN_CM)

def post_modulate(duty_l: float, duty_r: float) -> tuple[float, float]:
    """
    Hook for 'wild logic nuances':
    - keep them synced,
    - add biases,
    - inject computer vision weights, etc.
    """
    # Example 1: synchronized — both take the stronger of the two
    if SYNCED:
        common = max(duty_l, duty_r)
        duty_l, duty_r = common, common

    # Example 2: apply biases (e.g., from CV later)
    duty_l *= BIAS_L
    duty_r *= BIAS_R

    # Clamp after modulation
    return clamp(duty_l, 0.0, 1.0), clamp(duty_r, 0.0, 1.0)

def apply_deadzone(duty: float) -> float:
    """Prevent weak buzzing. Zero-out small values; keep full off at 0."""
    if duty == 0.0:
        return 0.0
    return max(duty, MIN_DUTY)

def main():
    all_new_distancesensor.setup()
    motor_l = PWMOutputDevice(MOTOR_L_PIN, frequency=PWM_HZ, initial_value=0.0)
    motor_r = PWMOutputDevice(MOTOR_R_PIN, frequency=PWM_HZ, initial_value=0.0)

    try:
        while True:
            try:
                d_l = all_new_distancesensor.get_distance(TRIG_L, ECHO_L)
                d_r = all_new_distancesensor.get_distance(TRIG_R, ECHO_R)

                duty_l = map_dist_to_duty(d_l)
                duty_r = map_dist_to_duty(d_r)

                duty_l, duty_r = post_modulate(duty_l, duty_r)

                motor_l.value = apply_deadzone(duty_l)
                motor_r.value = apply_deadzone(duty_r)

                # Debug (optional)
                print(f"L: {d_l:5.1f} cm -> {duty_l:0.2f} | R: {d_r:5.1f} cm -> {duty_r:0.2f}")

            except Exception as e:
                # On sensor failure, stop both motors briefly
                motor_l.value = 0.0
                motor_r.value = 0.0
                print("Sensor error:", e)

            sleep(0.05) # ~20 Hz update
    finally:
        motor_l.off()
        motor_r.off()
        all_new_distancesensor.cleanup()

if __name__ == "__main__":
    main()