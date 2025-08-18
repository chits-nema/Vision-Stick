import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np


model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
names = model.names

# reference vals for calibration a4 width 21 cm
KNOWN_DISTANCE = 50.0
KNOWN_WIDTH = 21.0
 
# method using the similarity rate
def focal_length_calculation(known_distance, known_width, width_in_rf_image):
    return (width_in_rf_image * known_distance) / known_width

# calculate the distance bw cam and object
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
    if per_width == 0:
        return None
    return (knownWidth * focalLength) / perWidth

# for the first time for callibration
focalLength = None
calibrated = False

while True:
    read, frame = cap.read()
    if not read: 
        break

    results = model(frame)[0]
    #results = model.predict(stream=True, imgsz=512)

    detections = sv.Detections.from_ultralytics(results)

   # labels = [
   #     f"{class_name} {confidence:.2f}"
   #     for class_name, confidence
   #     in zip(detections['class_name'], detections.confidence)
   # ]

    labels = []
    for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        class_name = names[int(class_id)]

        # bounding box width in pixels 
        x1, y1, x2, y2 = xyxy
        per_width = x2 - x1

        # if callibration is not done yet we are calculating it with a4 paper
        if not calibrated and class_name == "bottle":  # paper as reference
            focalLength = focal_length_calculation(KNOWN_DISTANCE, KNOWN_WIDTH, per_width)
            calibrated = True
            print(f"[INFO] Camera is callibrated. Focal Length = {focalLength:.2f}")

        distance = None
        if calibrated:
            distance = distance_to_camera(KNOWN_WIDTH, focalLength, per_width)

        if distance:
            label_text = f"{class_name} {confidence:.2f}, {distance:.1f}cm"
        else:
            label_text = f"{class_name} {confidence:.2f}"

        labels.append(label_text)

        # warning
        if distance and distance < 100:
            print(f"[WARNING] {class_name} so close: {distance:.1f}cm")

    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    cv2.imshow("COCO Detection", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

    for r in results: 
        for c in r.boxes.cls:
            print(names[int(c)])


cap.release()
cv2.destroyAllWindows()

