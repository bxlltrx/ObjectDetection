from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

cap = cv2.VideoCapture("https://storage.yandexcloud.net/o-code/tests/computer-vision/video-tracking.mp4")

#дообученная yolov8l
model = YOLO('yoloWeights/hardhat labeling.pt')


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('task2.2_out.mp4', fourcc, fps, (width, height))

track_history = defaultdict(lambda: [])
detected_track_ids = set()

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:

            boxes = results[0].boxes.xywh.cpu()  # xywh координаты боксов
            track_ids = results[0].boxes.id.int().cpu().tolist()  # идентификаторы треков
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # class IDs

            count_with_hardhat = class_ids.count(0)
            count_without_hardhat = class_ids.count(1)
            detected_track_ids.update(track_ids)

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
                cv2.putText(annotated_frame, f'total tracks: {len(detected_track_ids)}', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)


            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            out.write(annotated_frame)
        else:

            cv2.imshow("YOLOv8 Tracking", frame)
            out.write(frame)


        if cv2.waitKey(1) == 27:
            break

    else:
        print("Video End")
        break

cap.release()
out.release()
cv2.destroyAllWindows()