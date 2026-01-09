import os
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from dotenv import load_dotenv
from config import MODEL, VLM , RESIZE 


load_dotenv()

MODEL_ID = MODEL["id"]
HF_TOKEN = os.getenv("HF_TOKEN")
PROMPT = VLM["prompt"]
h = RESIZE["height"]
w = RESIZE["width"]
print(f"[VLM] Loading {MODEL_ID}")

# ========================
# Load Processor
# ========================
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN
)

# ========================
# Load Model
# ========================
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

model.eval()
print("[VLM] Model loaded successfully")

# ========================
def analyze_frame(frame_path):
    """
    Takes one frame path and returns its AI description
    """

    img = cv2.imread(frame_path)
    img = cv2.resize(img, (w, h))
    img = Image.fromarray(img[:, :, ::-1])  # BGR → RGB

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image", "image": img}
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,
            top_p=0.9
        )

    result = processor.decode(output[0], skip_special_tokens=True)

    if "assistant" in result:
        result = result.split("assistant")[-1].strip()

    return result
