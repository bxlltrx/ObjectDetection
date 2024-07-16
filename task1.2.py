import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

image = cv2.imread("720x.jpg")
model = YOLO('yoloWeights/count person.pt')
results = model(image)

detections = results[0].boxes.data.cpu().numpy()
labels = results[0].names

def distanceRect(rect1, rect2, threshold=100):
    x1, y1, x2, y2 = rect1[:4]
    x1_, y1_, x2_, y2_ = rect2[:4]
    center1 = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
    center2 = (x1_ + (x2_ - x1_) // 2, y1_ + (y2_ - y1_) // 2)
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    return distance < threshold

groups = []
for rect in detections:
    added = False
    for group in groups:
        if any(distanceRect(rect, other_rect) for other_rect in group):
            group.append(rect)
            added = True
            break
    if not added:
        groups.append([rect])


for group in groups:
    if len(group) > 1:
        x_min = min(rect[0] for rect in group)
        y_min = min(rect[1] for rect in group)
        x_max = max(rect[2] for rect in group)
        y_max = max(rect[3] for rect in group)
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
        cv2.putText(image, f'{len(group)}', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

cv2.imwrite("task1.2_out.jpg", image)