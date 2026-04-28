import os
import torch
import cv2
from PIL import Image
try:
    from transformers import AutoProcessor
except ImportError as exc:
    raise ImportError(
        "transformers not installed. Run: pip install transformers"
    ) from exc
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError as exc:
        raise ImportError(
            "Installed transformers is incompatible with this project. "
            "Install a 4.x version: pip install \"transformers>=4.40.0,<5\""
        ) from exc
from dotenv import load_dotenv
from config import MODEL, VLM , RESIZE 


load_dotenv()

MODEL_ID = MODEL["id"]
HF_TOKEN = os.getenv("HF_TOKEN")
PROMPT = VLM["prompt"]
h = RESIZE["height"]
w = RESIZE["width"]
MAX_TOKENS = MODEL["max_tokens"]

# ========================
# Load Processor
# ========================
_processor = None
_model = None


def load_vlm():
    """
    Lazily load and cache the VLM processor + model.

    This prevents expensive model initialization at import time, and allows
    non-VLM parts of the pipeline (detection/tracking/zones) to run independently.
    """

    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    print(f"[VLM] Loading {MODEL_ID}")
    _processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    use_cuda = torch.cuda.is_available()
    model_device = "cuda:0" if use_cuda else "cpu"
    model_dtype = torch.float16 if use_cuda else torch.float32

    _model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=model_dtype,
        device_map={"": model_device},
        token=HF_TOKEN,
    )
    _model.eval()
    print("[VLM] Model loaded successfully")
    return _processor, _model

# ========================
def analyze_frame(frame_path):
    """
    Takes one frame path and returns its AI description
    """

    processor, model = load_vlm()
    img = cv2.imread(frame_path)

    if img is None:
        raise ValueError(f"Failed to read image {frame_path}")
    
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
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            temperature=0,
            top_p=0.9
        )

    result = processor.decode(output[0], skip_special_tokens=True)

    if "assistant" in result:
        result = result.split("assistant")[-1].strip()

    return result
