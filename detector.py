# detector.py
import cv2
from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")  # person class only

def detect_people(frame):
    results = yolo(frame, conf=0.4, classes=[0])  # class 0 = person
    persons = []

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        persons.append({
            "person_id": i + 1,
            "bbox": [x1, y1, x2, y2]
        })

    return persons
