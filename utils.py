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
COLOR_MAP = {
    "RED":    (0, 0, 255),
    "GREEN":  (0, 255, 0),
    "BLUE":   (255, 0, 0),
    "YELLOW": (0, 255, 255)
}

def draw_zones(frame):
    """
    Zone-safe overlay:
    - Border-first visualization
    - Ultra-low transparency fill
    - Preserves small object visibility (phones, hands, text)
    """
    h, w, _ = frame.shape

    zones = [
        ("RED",    (0, 0),        (w // 2, h // 2)),   # Top-left
        ("GREEN",  (w // 2, 0),   (w, h // 2)),       # Top-right
        ("BLUE",   (0, h // 2),   (w // 2, h)),       # Bottom-left
        ("YELLOW", (w // 2, h // 2), (w, h))          # Bottom-right
    ]

    overlay = frame.copy()

    # 🔒 SAFE SETTINGS (do not increase)
    alpha = 0.08          # ultra-light tint
    border_thickness = 2  # clear but non-intrusive

    for color_name, pt1, pt2 in zones:
        # Draw ultra-light fill (safe)
        cv2.rectangle(
            overlay,
            pt1,
            pt2,
            COLOR_MAP[color_name],
            thickness=-1
        )

        # Draw clear border (main visual cue)
        cv2.rectangle(
            frame,
            pt1,
            pt2,
            COLOR_MAP[color_name],
            thickness=border_thickness
        )

    # Blend overlay gently
    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )

    return frame

