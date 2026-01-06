from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Carregar processor e modelo
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

# Abrir imagem
image = Image.open("image.png").convert("RGB")

# Preparar imagem para o modelo
pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

# Gerar texto
with torch.no_grad():
    generated_ids = model.generate(pixel_values)

text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

# Mostrar no terminal
print("TEXTO RECONHECIDO:")
print(text)

# Salvar em arquivo TXT
with open("resultado_trocr.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Arquivo 'resultado_trocr.txt' gerado com sucesso.")
