22# Vision-Stick
The VisionStick is a smart system integrated with a mobility cane, aimed at helping the visually impaired navigate their surroundings. The device is designed to detect obstacles around the user. When an obstacle is detected, users are alerted through an alert system. 

## Common Detected Objects
person, bicycle, car, motorcycle, fire hydrant, stop sign, chair, bus, traffic light

---

## Why Vision-Stick?
Cities like Heilbronn have constant construction and complex traffic zones that make navigation especially difficult for people with visual impairments.  
Vision-Stick enhances a traditional white cane with real-time obstacle awareness and discreet haptic feedback, aiming to increase safety and independence.

---

## Features
- **Obstacle detection** with stereo cameras and YOLO object detection.  
- **Distance measurement** using stereo depth + ultrasonic sensors.  
- **Haptic feedback** via two vibration motors (direction and urgency encoded).  
- **Lightweight Raspberry Pi client** + external server for real-time processing.  
- **Stereo calibration** with checkerboard for accurate depth estimation.

---

## Hardware
- Raspberry Pi 5 (8 GB)
- 2 × Raspberry Pi Cameras (IMX219)  
- 2 × HC-SR04 ultrasonic sensors  
- 2 × coin vibration motors  
- 3D-printed mounting bracket for the cane

---

## Software Stack
- Python 3.10+  
- OpenCV (contrib) — stereo calibration & disparity maps  
- Ultralytics YOLO — object detection  
- Flask — HTTPS API for communication between Pi and server  
- gpiozero / RPi.GPIO — ultrasonic + motor control

---

## Dependencies & Setup
1) We created a HTTPS server using the Flask framework with the purpose of storing incoming frames from the Pi 5 cameras. This server runs in an external host (i.e. Windows Laptop); with the main goal of executing the Stereo-Vision module in this device. Thus, this relieves the Raspberry Pi 5 of executing CPU-intensive operations, reducing the overhead and thus allowing the system to work within the real-time constraints of such an assisting device.

2) Camera Calibration
  
3) In the source codes, we used many ubiquitous python libraries such as: flask, numpy, opencv, supervision, requests and openssl. This enables the logical integration of the stereovision model with the server manager code and various mathematical operations.

5) In order to program the logic of the implemented circuits we have used various phyton libraries as well; such as gpiozero, picamera2, numpy and cv. As one can infer by their name these allow to define the explicit behavior of the gpio pins and the connected camera modules.

6) Of course, we also implemented our own phyton classes and modules to manage important functionalities and objects, these are SendingClient, and generic classes such as DisplayManager and StereoVissionProcessor.
  
---

## Acknowledgements
- Developed as part of the Embedded Systems, Cyber-Physical Systems and Robotics (INHN0018) course at TUM. 
- Thanks to BSVW Heilbronn for field insights
- Built on OpenCV and Ultralytics YOLO
