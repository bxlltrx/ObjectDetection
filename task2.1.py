import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLOv8 model
model = YOLO("yoloWeights/labeling.pt")

cap = cv2.VideoCapture("video-tracking.mp4")

# Define class names


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get bounding box coordinates, track IDs, and class IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs

            # Count instances of each class
            count_with_hardhat = class_ids.count(0)
            count_without_hardhat = class_ids.count(1)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display class counts on the frame
            cv2.putText(annotated_frame, f'with hardhat: {count_with_hardhat}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 0, 0), 2)

            cv2.putText(annotated_frame, f'without hardhat: {count_without_hardhat}', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (200, 213, 48), 2)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Прерывание цикла при нажатии клавиши 'Esc'
        if cv2.waitKey(1) == 27:
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
