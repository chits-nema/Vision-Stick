#!/usr/bin/python
import RPi.GPIO as GPIO
import time

try:
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)	
	TRIG = 23
	ECHO = 24

	GPIO.setup(TRIG, GPIO.OUT)
	GPIO.setup(ECHO, GPIO.IN)
	GPIO.output(TRIG, False)
	print ("Waiting for sensor to settle")
	time.sleep(2)
	print ("Calculating distance")
	GPIO.output(TRIG, True)
	time.sleep(0.00001)
	GPIO.output(TRIG, False)
#	timeout = 0.02 

#	start_time = time.time()
	while GPIO.input(ECHO) == 0:
		pulse_start = time.time()
#		if pulse_start_time - start_time > timeout:
#			print("something")
#			break
#	start_time = time.time()

	while GPIO.input(ECHO) == 1:
		pulse_end = time.time()
	#	if pulse_end_time - start_time > timeout:
	#		print("something")
	#		break
#		print ("almost done")

	pulse_duration = pulse_end - pulse_start
	distance = round(pulse_duration*17150, 2)
	
	print ("Distance: ", distance, " cm")

finally:
	GPIO.cleanup()