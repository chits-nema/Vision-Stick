from gpiozero import PWMOutputDevice
from time import sleep

motor1 = PWMOutputDevice(13, frequency=100)
motor1.value = 0.03
motor2 = PWMOutputDevice(12, frequency=100)
motor2.value = 0.03
sleep(5)
motor1.off()
motor2.off()