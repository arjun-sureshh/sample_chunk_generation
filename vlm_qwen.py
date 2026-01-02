import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
from dotenv import load_dotenv

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"[VLM] Loading {MODEL_ID}")

processor = AutoProcessor.from_pretrained(MODEL_ID , use_fast=False , token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    token=HF_TOKEN,
    device_map="auto"
)

def analyze_frames(frames):
    """
    frames = list of OpenCV images (BGR)
    """
    pil_images = []
    for f in frames:
        # convert OpenCV → PIL
        pil_images.append(Image.fromarray(f[:, :, ::-1]))

    prompt = "Describe what is happening in these images."

    inputs = processor(
        images=pil_images,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=128)

    result = processor.decode(output[0], skip_special_tokens=True)
    return result
