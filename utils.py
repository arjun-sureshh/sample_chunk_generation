import shutil
import os

import cv2

def clean_directory(path: str):
    """
    Deletes a directory if it exists, then recreates it.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[CLEANUP] Removed existing directory: {path}")

    os.makedirs(path, exist_ok=True)
    print(f"[CLEANUP] Created directory: {path}")




# OpenCV uses BGR, not RGB
COLOR_MAP = {
    "RED":    (0, 0, 255),
    "GREEN":  (0, 255, 0),
    "BLUE":   (255, 0, 0),
    "YELLOW": (0, 255, 255)
}

def draw_zones(frame):
    """
    Draws fixed colored zones on the frame for VLM guidance
    """
    h, w, _ = frame.shape

    zones = [
        ("RED",    (0, 0),        (w//2, h//2)),   # Top-left
        ("GREEN",  (w//2, 0),     (w, h//2)),      # Top-right
        ("BLUE",   (0, h//2),     (w//2, h)),      # Bottom-left
        ("YELLOW", (w//2, h//2),  (w, h))          # Bottom-right
    ]

    overlay = frame.copy()
    alpha = 0.25  # transparency

    for color_name, pt1, pt2 in zones:
        cv2.rectangle(overlay, pt1, pt2, COLOR_MAP[color_name], -1)
        cv2.rectangle(frame, pt1, pt2, COLOR_MAP[color_name], 2)

    # Blend overlay with original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame
