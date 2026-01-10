from ultralytics import YOLO
import cv2
import time
import os
import torch

# ===== CONFIG =====
IMG_PATH = "data/imagens/Su57.jpg"
OUT_PATH = "YOLO/resultados/yolo_cpu.jpg"
DEVICE = "cpu"

os.makedirs("resultados", exist_ok=True)

# ===== MODELO =====
model = YOLO("yolov8n.pt")
model.to(DEVICE)

# ===== IMAGEM =====
img = cv2.imread(IMG_PATH)

start = time.perf_counter()
results = model(img)
end = time.perf_counter()

annotated = results[0].plot()

# ===== SAÍDA =====
cv2.namedWindow("YOLO CPU", cv2.WINDOW_NORMAL)
cv2.imshow("YOLO CPU", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(OUT_PATH, annotated)

print("=== YOLO IMAGEM ===")
print(f"Device: {DEVICE}")
print(f"Tempo: {end - start:.4f} segundos")
print(f"Imagem salva em: {OUT_PATH}")
