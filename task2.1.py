from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture("https://storage.yandexcloud.net/o-code/tests/computer-vision/video-tracking.mp4")


model = YOLO('yoloWeights/labeling.pt')


classNames = ["person with hardhat", "person without hardhat"]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.1)

totalCount = []

while cap.isOpened():
    success, img = cap.read()
    results = model(img, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1) , int(x2), int(y2)
            w, h = x2-x1, y2-y1


            #Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            #class name
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            if currentClass == "person with hardhat":
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                cvzone.putTextRect(img, f'  {"person with hardhat"}', (max(0, x1), max(35, y1)))
            if currentClass == "person without hardhat":
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                cvzone.putTextRect(img, f'  {"person without hardhat"}', (max(0, x1), max(35, y1)))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(Id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        if totalCount.count(Id) == 0:
            totalCount.append(Id)
        cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.imshow("Image", img)
    cv2.waitKey(1)


