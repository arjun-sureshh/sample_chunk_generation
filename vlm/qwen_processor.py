"""Qwen VLM helpers for per-person classification."""

from __future__ import annotations

import json
import time
from typing import Any

import cv2
import torch
from PIL import Image

import vlm_processor

PROMPT = """
You are analyzing a cropped image of a person from
a retail store CCTV camera.

Determine if this person is a STORE STAFF member or CUSTOMER.

Look for these staff indicators:
- Uniform or matching outfit
- ID badge or name tag
- Staff lanyard
- Apron or vest
- Walkie-talkie or earpiece
- Professional attire matching store theme

Return ONLY valid JSON, no extra text:
{
  "role": "staff" or "customer",
  "confidence": 0.0 to 1.0,
  "reason": "one sentence"
}

If image is too dark or unclear:
{"role": "customer", "confidence": 0.3, "reason": "image unclear"}
"""


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse model response as JSON with fenced-block fallback."""
    cleaned = raw_text.strip()
    if "```" in cleaned:
        chunks = cleaned.split("```")
        for chunk in chunks:
            candidate = chunk.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                cleaned = candidate
                break

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    parsed = json.loads(cleaned)
    role = str(parsed.get("role", "customer")).lower()
    if role not in {"staff", "customer", "uncertain"}:
        role = "customer"
    confidence = float(parsed.get("confidence", 0.3))
    confidence = max(0.0, min(1.0, confidence))
    reason = str(parsed.get("reason", "vlm classification"))
    return {
        "role": role,
        "confidence": confidence,
        "source": "vlm",
        "reason": reason,
    }


def classify_person(crop_bgr) -> dict[str, Any]:
    """Classify one person crop as staff/customer/uncertain via Qwen."""
    start = time.perf_counter()
    try:
        processor, model = vlm_processor.load_vlm()
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return {
                "role": "customer",
                "confidence": 0.3,
                "source": "vlm",
                "reason": "image unclear",
            }

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image", "image": image},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=vlm_processor.MAX_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=0.9,
            )

        decoded = processor.decode(output[0], skip_special_tokens=True)
        if "assistant" in decoded:
            decoded = decoded.split("assistant")[-1].strip()

        result = _parse_json_response(decoded)
        result["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return result
    except Exception:
        return {
            "role": "customer",
            "confidence": 0.3,
            "source": "vlm",
            "reason": "parse error",
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }
    finally:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(f"[VLM] classify_person took {elapsed_ms}ms")
