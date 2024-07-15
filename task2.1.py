import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict


model = YOLO("yoloWeights/count person.pt")

cap = cv2.VideoCapture("https://storage.yandexcloud.net/o-code/tests/computer-vision/video-tracking.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('task2.1_out.mp4', fourcc, fps, (width, height))


previous_boxes = defaultdict(lambda: None)
smooth_factor = 0.5


detected_track_ids = set()

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            #сглаживания рамок на выходном видео
            for i, box in enumerate(boxes):
                track_id = track_ids[i]
                if previous_boxes[track_id] is None:
                    previous_boxes[track_id] = box
                else:
                    previous_boxes[track_id] = smooth_factor * previous_boxes[track_id] + (1 - smooth_factor) * box


            detected_track_ids.update(track_ids)

            annotated_frame = results[0].plot()

            cv2.putText(annotated_frame, f'total tracks: {len(detected_track_ids)}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

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
