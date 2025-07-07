import cv2
import supervision as sv
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
annotator = sv.BoxAnnotator()

while True:
    read, frame = cap.read()
    if not read: 
        break

    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    annotated_frame = annotator.annotate(
        scene=frame.copy(), detections=detections
    )

    cv2.imshow("COCO Detection", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()