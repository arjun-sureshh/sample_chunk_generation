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








COLOR_MAP = {
    "RED":    (0, 0, 255),
    "GREEN":  (0, 255, 0),
    "BLUE":   (255, 0, 0),
    "YELLOW": (0, 255, 255)
}

def draw_zones(frame):
    
    h, w, _ = frame.shape
    t = 2  # thickness 

    # RED zone (Top-Left)
    cv2.rectangle(frame, (0, 0), (w//2, h//2), COLOR_MAP["RED"], t)

    # GREEN zone (Top-Right)
    cv2.rectangle(frame, (w//2, 0), (w, h//2), COLOR_MAP["GREEN"], t)

    # BLUE zone (Bottom-Left)
    cv2.rectangle(frame, (0, h//2), (w//2, h), COLOR_MAP["BLUE"], t)

    # YELLOW zone (Bottom-Right)
    cv2.rectangle(frame, (w//2, h//2), (w, h), COLOR_MAP["YELLOW"], t)

    return frame
