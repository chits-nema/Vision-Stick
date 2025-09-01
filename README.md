# Vision-Stick
The VisionStick is a smart system integrated with a mobility cane, aimed at helping the visually impaired navigate their surroundings. The device is designed to detect obstacles around the user. When an obstacle is detected, users are alerted through an alert system. 

##Common Detected Objects
person, bicycle, car, motorcycle, fire hydrant, stop sign, chair", "bus", "traffic light"

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
1) Server

2) Camera Calibration

3) Raspberry Pi

---

## Acknowledgements
- Developed as part of the Embedded Systems, Cyber-Physical Systems and Robotics (INHN0018) course at TUM. 
- Thanks to BSVW Heilbronn for field insights
- Built on OpenCV and Ultralytics YOLO

---
## Team Members
- [Advitya Chopra] (https://github.com/adv1ch)
- [Chitsidzo V. Nemazuwa] (https://github.com/chits-nema)
- [Kanaya N. Ozora] (https://github.com/nayachewsudon)
- [Oscar E. Navarro Banderas] (https://github.com/Oscar27NX)
- [Sofia Libman] (http://github.com/morisdann)
- [Syed R. Qamar] ()
- [Süeda Özkaya] (https://github.com/suedaozkaya)
