# raspberry_pi_client.py
import cv2
import requests
import base64
import json
import time
import numpy as np
from io import BytesIO
from picamera2 import Picamera2

class FrameCaptureClient:
    def __init__(self, server_url, quality=80, resize_width=640):
        self.server_url = server_url
        self.quality = quality
        self.resize_width = resize_width
        try:
            self.picamL = Picamera2(0)
            self.picamR = Picamera2(1)
        except Exception as e:
            print(f"Camera start failed: {e}")
            exit(1)
        
        # Set camera properties for better performance
        configL = self.picamL.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
        configR = self.picamR.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"}) #config should have done YUV to RGB conversion

        self.picamL.configure(configL)
        self.picamR.configure(configR)
        self.picamL.start()
        self.picamR.start()

    def encode_frame(self, frame):
        """Encode frame to base64 JPEG with compression"""
        # Resize frame to reduce bandwidth
        height, width = frame.shape[:2]
        new_height = int(height * (self.resize_width / width))
        frame = cv2.resize(frame, (self.resize_width, new_height))
        
        # Encode as JPEG with quality setting
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64, frame.shape
    
    def send_stereo_frames(self, frame_dataL, frame_dataR, frame_shapeL, frame_shapeR):
        """Send frame to processing server"""
        try:
            payload = {
                'frameL': frame_dataL,
                'frameR': frame_dataR,
                'shapeL': frame_shapeL,
                "shapeR": frame_shapeR,
                'timestamp': time.time()
            }
            
            response = requests.post(
                f"{self.server_url}/process_frame",
                json=payload,
                timeout=5,
                verify=False  # For self-signed certificates
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Server error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return None
    
    def capture_frames(self):
        frameL = self.picamL.capture_array()
        frameR = self.picamR.capture_array()

        return frameL, frameR
    
    
    def run(self):
        """Main capture and send loop"""
        print("Starting frame capture client...")
        print(f"Server: {self.server_url}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frameL, frameR = self.capture_frames()
                if frameL is None or frameR is None:
                    print("Failed to read frame")
                    continue
                
                # Encode frame
                frame_dataL, frame_shapeL = self.encode_frame(frameL)
                frame_dataR, frame_shapeR = self.encode_frame(frameR)
                
                # Send to server
                result = self.send_stereo_frames(frame_dataL, frame_dataR, frame_shapeL, frame_shapeR)
                
                if result: 
                    # Display results if available
                    if "detectionsL" in result and "detectionsR" in result:
                        print(f"Detections Left: {result['detectionsL']}")
                        print(f"Detections Right: {result['detectionsR']}")

                    # Calculate and display FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"Average FPS: {fps:.2f}")
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.033)  # ~30 FPS max
                
        except KeyboardInterrupt:
            print("\nShutting down client...")
        finally:
            self.picamL.stop()
            self.picamR.stop()

if __name__ == "__main__":
    # Configuration
    SERVER_URL = "https://YOUR_LAPTOP_IP:8443"  # Replace with your laptop's IP
    CAMERA_ID = 0
    QUALITY = 80  # JPEG quality (1-100)
    RESIZE_WIDTH = 640  # Resize width for bandwidth optimization
    
    client = FrameCaptureClient(SERVER_URL, QUALITY, RESIZE_WIDTH)
    client.run()
