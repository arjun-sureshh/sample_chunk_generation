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






# OpenCV uses BGR format
import cv2

COLOR_MAP = {
    "RED":    (0, 0, 255),
    "GREEN":  (0, 255, 0),
    "BLUE":   (255, 0, 0),
    "YELLOW": (0, 255, 255),
    "WHITE":  (255, 255, 255)
}

def draw_zones(frame):
    """
    Grid-based zone boundaries:
    - Full visual separation
    - Zero overlap
    - Model-safe
    """
    h, w, _ = frame.shape
    thickness = 2

    # Outer boundary (optional but recommended)
    cv2.rectangle(frame, (0, 0), (w, h), COLOR_MAP["WHITE"], 1)

    # Vertical center line
    cv2.line(frame, (w // 2, 0), (w // 2, h), COLOR_MAP["WHITE"], thickness)

    # Horizontal center line
    cv2.line(frame, (0, h // 2), (w, h // 2), COLOR_MAP["WHITE"], thickness)

    return frame
