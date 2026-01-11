import cv2
import pytesseract
import time
import os

# ===== CONFIG =====
IMG_PATH = "data/imagens/Diario1.jpg"
OUT_PATH = "OCR_tesseract/resultados/ocr_tesseract_gpu.txt"
DEVICE = "gpu (não suportado pelo Tesseract)"

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.makedirs("OCR_tesseract/resultados", exist_ok=True)

# ===== EXECUÇÃO =====
start = time.perf_counter()

img = cv2.imread(IMG_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

texto = pytesseract.image_to_string(
    thresh,
    lang="por",
    config="--psm 6"
)

end = time.perf_counter()

# ===== SAÍDA =====
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(texto)

print("=== TESSERACT OCR ===")
print(f"Device: {DEVICE}")
print("Obs: Tesseract roda apenas em CPU")
print(f"Tempo: {end - start:.4f} segundos")
print(f"Resultado salvo em: {OUT_PATH}")
print("\nTexto reconhecido:\n")
print(texto)
