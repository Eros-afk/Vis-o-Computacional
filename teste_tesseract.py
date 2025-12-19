import cv2
import pytesseract

# Se for Windows, ajuste se necessário:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Ler imagem
img = cv2.imread("image.png")

# Converter para cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarização
_, thresh = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# OCR
texto = pytesseract.image_to_string(
    thresh,
    lang="por",
    config="--psm 6"
)

# Salvar em arquivo TXT
with open("resultado_ocr.txt", "w", encoding="utf-8") as f:
    f.write(texto)

print("OCR concluído. Texto salvo em resultado_ocr.txt")
