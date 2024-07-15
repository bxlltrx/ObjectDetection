from ultralytics import YOLO
import cv2


model = YOLO("yoloWeights/hardhat labeling.pt")

results = model.predict("720x.jpg", save=False, imgsz=640)

image = cv2.imread("720x.jpg")

without_hardhat = []
with_hardhat = []

for result in results:
    for box in result.boxes:
        if box.cls == 0:
            with_hardhat.append(box.xyxy[0])
        if box.cls == 1:
            without_hardhat.append(box.xyxy[0])

for box in without_hardhat:
    x1, y1, x2, y2 = [int(coord) for coord in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

for box in with_hardhat:
    x1, y1, x2, y2 = [int(coord) for coord in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

num_no_helmet = len(without_hardhat)
num_helmet = len(with_hardhat)

cv2.putText(image, f"without hardhat: {num_no_helmet}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(image, f"with hardhat: {num_helmet}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imwrite("task1.3_out.jpg", image)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
