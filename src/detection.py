import cv2
import supervision as sv
from ultralytics import YOLO



model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
names = model.names

while True:
    read, frame = cap.read()
    if not read: 
        break

    results = model(frame)[0]
    #results = model.predict(stream=True, imgsz=512)
    

    detections = sv.Detections.from_ultralytics(results)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )

    annotated_image = label_annotator.annotate(
    scene=annotated_frame, detections=detections, labels=labels)

    cv2.imshow("COCO Detection", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

    for r in results: 
        for c in r.boxes.cls:
            print(names[int(c)])


cap.release()
cv2.destroyAllWindows()

