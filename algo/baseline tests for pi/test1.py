#first test of yolov8n on pi, using cam module feed

from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

# Use Pi camera (0 usually works)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=320,        # VERY IMPORTANT (reduce load)
        conf=0.4,
        device="cpu",
        verbose=False
    )

    annotated = results[0].plot()

    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Pi YOLO Test", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()