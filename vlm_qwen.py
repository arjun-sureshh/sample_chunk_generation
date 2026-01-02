import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from dotenv import load_dotenv
import cv2
# ========================
# CONFIG
# ========================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"[VLM] Loading {MODEL_ID}")

# ========================
# Processor
# ========================
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN
)

# ========================
# Model (GPU optimized)
# ========================
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

model.eval()
print("[VLM] Model loaded successfully")

# ========================
# Inference
# ========================
def analyze_frames(frames):
    """
    frames = list of OpenCV images (BGR)
    """

    images = []
    for f in frames:
        f = cv2.resize(f, (448, 448))   # VERY important
        images.append(Image.fromarray(f[:, :, ::-1]))  # BGR → RGB → PIL

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what is happening in these frames."},
                *[{"type": "image", "image": img} for img in images]
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(  
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0)

    result = processor.decode(output[0], skip_special_tokens=True)
    return result
