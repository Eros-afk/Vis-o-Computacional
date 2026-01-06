import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

print("Rodando em:", device)
