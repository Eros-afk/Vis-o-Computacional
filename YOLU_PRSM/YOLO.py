from ultralytics import YOLO
import cv2

# Carrega modelo pré-treinado (rápido e leve)
model = YOLO("yolov8n.pt")

# Lê imagem
img = cv2.imread("imagem.jpg")

print(img is None)

# Detecta objetos
results = model(img)

# Desenha as detecções
annotated = results[0].plot()

cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
cv2.imshow("YOLO", annotated)
cv2.waitKey(0)

# Mostra na tela
cv2.imshow("YOLO - Detecção de Objetos", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("resultado.jpg", annotated)
print("Imagem salva como resultado.jpg")
