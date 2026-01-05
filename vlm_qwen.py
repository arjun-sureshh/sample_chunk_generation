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

    PROMPT = """
You are a professional AI retail video intelligence system analyzing a shop surveillance video.

The video shows people with bounding boxes and tracking IDs.
You must assign:
- SID (Staff ID) → for employees
- CID (Customer ID) → for customers

Each SID and CID must remain consistent for the same person across all frames.
Never merge, split, or rename IDs.

TASKS:
1. Detect all people in the video.
2. Classify each person as Staff, Customer, or Unknown.
3. Assign a unique SID to every staff member and CID to every customer.
4. For each person, describe:
   - Clothing (especially shirt color)
   - What they are holding
   - What actions they perform over time
5. Identify staff wearing black shirts.
6. Count total staff and customers.
7. Describe the shop layout.
8. Describe customer behavior.
9. Provide a complete story of what happens.

OUTPUT FORMAT:

### People Detected
List all SIDs and CIDs with dress and items.

### Staff Wearing Black Shirts
List all SIDs.

### Individual Activity Timeline
Show actions for each SID and CID.

### Shop Description
Describe store layout and type.

### Customer Behavior Summary

### Overall Video Summary

### Key Activities
Bullet points using SID and CID.

### Suspicious or Unusual Activity
If none: "No suspicious activity observed."
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
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
