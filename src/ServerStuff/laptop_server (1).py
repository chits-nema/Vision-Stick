# gaming_laptop_server.py
import cv2
import supervision as sv
from ultralytics import YOLO
import base64
import numpy as np
from flask import Flask, request, jsonify
import ssl
import threading
import time
from io import BytesIO

app = Flask(__name__)

class YOLOProcessor:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = []
        
        print(f"YOLO model loaded: {model_path}")
        print(f"Available classes: {len(self.names)}")
    
    def decode_frame(self, frame_data, frame_shape):
        """Decode base64 frame back to numpy array"""
        try:
            # Decode base64
            frame_bytes = base64.b64decode(frame_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(frame_bytes, np.uint8)
            
            # Decode JPEG
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame
        except Exception as e:
            print(f"Frame decode error: {e}")
            return None
    
    def process_frame(self, frame):
        """Process frame with YOLO and return results"""
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(frame)[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Create labels
            labels = [
                f"{self.names[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]
            
            # Annotate frame
            annotated_frame = self.box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )
            annotated_image = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            
            # Extract detection info
            detection_info = []
            for i, (class_id, confidence, bbox) in enumerate(zip(
                detections.class_id, detections.confidence, detections.xyxy
            )):
                detection_info.append({
                    'class': self.names[class_id],
                    'confidence': float(confidence),
                    'bbox': bbox.tolist()
                })
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            # Display annotated frame
            cv2.imshow("YOLO Detection Server", annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                return None
            
            # Print detections to console (matching your original code)
            for detection in detection_info:
                print(f"{detection['class']} ({detection['confidence']:.2f})")
            
            return {
                'detections': detection_info,
                'processing_time': processing_time,
                'total_frames': self.frame_count
            }
            
        except Exception as e:
            print(f"Processing error: {e}")
            return {'error': str(e)}
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_processing_time = np.mean(self.processing_times)
        
        return {
            'frames_processed': self.frame_count,
            'average_fps': avg_fps,
            'average_processing_time': avg_processing_time,
            'elapsed_time': elapsed_time
        }

# Global processor instance
processor = YOLOProcessor()

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """API endpoint to process frames"""
    try:
        data = request.json
        frame_data = data.get('frame')
        frame_shape = data.get('shape')
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Decode frame
        frame = processor.decode_frame(frame_data, frame_shape)
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Process frame
        result = processor.process_frame(frame)
        if result is None:
            return jsonify({'shutdown': True}), 200
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """API endpoint to get processing statistics"""
    return jsonify(processor.get_stats())

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

def create_ssl_context():
    """Create SSL context for HTTPS"""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    # Generate self-signed certificate if needed
    try:
        context.load_cert_chain('server.crt', 'server.key')
    except FileNotFoundError:
        print("SSL certificates not found. Generating self-signed certificate...")
        generate_self_signed_cert()
        context.load_cert_chain('server.crt', 'server.key')
    
    return context

def generate_self_signed_cert():
    """Generate self-signed SSL certificate"""
    import subprocess
    import os
    
    if not os.path.exists('server.crt'):
        cmd = [
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096', '-nodes',
            '-out', 'server.crt', '-keyout', 'server.key', '-days', '365',
            '-subj', '/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=localhost'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print("Self-signed certificate generated successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate certificate: {e}")
            print("Please install OpenSSL or provide your own SSL certificates")

if __name__ == "__main__":
    print("Starting YOLO Processing Server...")
    print("Available GPU devices:", [f"cuda:{i}" for i in range(cv2.cuda.getCudaEnabledDeviceCount())] if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "CPU only")
    
    # Create SSL context
    ssl_context = create_ssl_context()
    
    # Start server
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=8443,
        ssl_context=ssl_context,
        threaded=True,
        debug=False
    )