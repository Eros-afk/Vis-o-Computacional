from ultralytics import YOLO
import cv2
import time
import torch

DEVICE = "cpu"

model = YOLO("yolov8n.pt")
model.to(DEVICE)

cap = cv2.VideoCapture(0)

fps_list = []

while True:
    start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    for r in results:
        frame = r.plot()

    end = time.perf_counter()
    fps = 1 / (end - start)
    fps_list.append(fps)

    cv2.putText(
        frame,
        f"CPU | FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO CAMERA CPU", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("FPS médio:", sum(fps_list) / len(fps_list))
