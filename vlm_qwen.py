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
You are a professional AI retail video intelligence system analyzing shop surveillance video to monitor staff activities.

The video shows people with bounding boxes and tracking IDs.
Your primary focus is identifying STAFF MEMBERS by their BLACK SHIRT UNIFORM.

ASSIGNMENT RULES:
- Assign SID (Staff ID) ONLY to people wearing black shirts (staff uniform)
- Each SID must remain consistent for the same person across all frames
- Never merge, split, or rename IDs once assigned

TASKS:
1. Identify all staff members wearing black shirts
2. Assign a unique SID to each staff member
3. For each staff member, provide detailed description:
   - Full clothing description (pants color, shoes, accessories)
   - Physical characteristics (height estimate, build, hair, distinctive features)
   - Any items they are carrying or holding
4. Track what each staff member does throughout the video
5. Identify suspicious activities, particularly:
   - Staff using mobile phones during work
   - Staff talking to each other for extended periods
   - Staff idle or not attending to duties
   - Any unusual behavior
6. Count total number of staff detected
7. Provide timeline of activities for each SID

OUTPUT FORMAT:

### Staff Members Detected (Total: X)

**SID-1:**
- Uniform: Black shirt, [pants color], [shoe description]
- Physical Description: [height/build/hair/distinctive features for manual identification]
- Holding/Carrying: [items]

**SID-2:**
[same format]

### Individual Staff Activity Timeline

**SID-1:**
- 00:00-00:15: [action]
- 00:15-00:45: [action]
- [continue...]

**SID-2:**
[same format]

### Suspicious Activities Detected

**Phone Usage:**
- SID-X: Using phone at [timestamp] for [duration]

**Extended Conversations:**
- SID-X and SID-Y: Talking from [start time] to [end time] - Duration: [X minutes]

**Idle Time:**
- SID-X: Standing idle at [location] from [time] to [time]

**Other Concerns:**
[Any other unusual behavior]

If no suspicious activity: "No suspicious activity observed."

### Overall Staff Activity Summary
Brief narrative of staff behavior throughout the video.

### Manual Review Recommendations
List SIDs requiring closer review with reasons and timestamps.
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
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.1,
        top_p=0.9)

    result = processor.decode(output[0], skip_special_tokens=True)

     # Extract only the assistant's response (remove the prompt)
    if "assistant" in result:
        result = result.split("assistant")[-1].strip()
        
    return result
