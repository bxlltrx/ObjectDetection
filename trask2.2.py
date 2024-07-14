from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

cap = cv2.VideoCapture("video-tracking.mp4")


model = YOLO('yoloWeights/labeling.pt')


classNames = ["person with hardhat", "person without hardhat"]
track_history = defaultdict(lambda: [])

totalCount = []

while cap.isOpened():
    success, img = cap.read()
    if success:
        results = model.track(img, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:

            boxes = results[0].boxes.xywh.cpu()  # xywh координаты боксов
            track_ids = results[0].boxes.id.int().cpu().tolist()  # идентификаторы треков
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # class IDs

            count_with_hardhat = class_ids.count(0)
            count_without_hardhat = class_ids.count(1)

            annotated_frame = results[0].plot()

            # Отрисовка треков
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box  # координаты центра и размеры бокса
                track = track_history[track_id]
                track.append((float(x), float(y)))  # добавление координат центра объекта в историю
                if len(track) > 30:  # ограничение длины истории до 30 кадров
                    track.pop(0)

                # Рисование линий трека
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                cv2.putText(annotated_frame, f'with hardhat: {count_with_hardhat}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 0, 0), 2)

                cv2.putText(annotated_frame, f'without hardhat: {count_without_hardhat}', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (200, 213, 48), 2)


            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            # Если объекты не обнаружены, просто отображаем кадр
            cv2.imshow("YOLOv8 Tracking", img)

            # Прерывание цикла при нажатии клавиши 'Esc'
        if cv2.waitKey(1) == 27:
            break

    else:
        print("Video End")
        break
    # Освобождение видеозахвата и закрытие всех окон OpenCV
cap.release()




