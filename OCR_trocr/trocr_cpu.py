from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import time
import os

# ===== CONFIG =====
IMG_PATH = "data/imagens/image.png"
OUT_PATH = "resultados/ocr_trocr_cpu.txt"
DEVICE = "cpu"

os.makedirs("resultados", exist_ok=True)

# ===== MODELO =====
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
).to(DEVICE)

# ===== IMAGEM =====
image = Image.open(IMG_PATH).convert("RGB")
pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values.to(DEVICE)

# ===== EXECUÇÃO =====
start = time.perf_counter()

with torch.no_grad():
    generated_ids = model.generate(pixel_values)

text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

end = time.perf_counter()

# ===== SAÍDA =====
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("=== TrOCR ===")
print(f"Device: {DEVICE}")
print(f"Tempo: {end - start:.4f} segundos")
print(f"Resultado salvo em: {OUT_PATH}")
print("\nTexto reconhecido:\n")
print(text)
