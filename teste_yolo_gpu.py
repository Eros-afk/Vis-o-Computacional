from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("Su57.jpg", device=0)
