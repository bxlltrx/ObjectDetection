from ultralytics import YOLO
import cv2


model = YOLO("yoloWeights/count person.pt")  #

results = model.predict("720x.jpg", save=False, imgsz=640)

image = cv2.imread("720x.jpg")

people_boxes = []

for result in results:
    for box in result.boxes:
        people_boxes.append(box.xyxy[0])


for box in people_boxes:
    x1, y1, x2, y2 = [int(coord) for coord in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

num_people = len(people_boxes)

cv2.putText(image, f"Count: {num_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imwrite("task1.1_out.jpg", image)
