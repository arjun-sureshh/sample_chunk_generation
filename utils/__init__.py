"""Shared utility helpers for the sample chunk generation project."""

from __future__ import annotations

import os
import shutil

import cv2


def clean_directory(path: str) -> None:
    """Delete a directory if it exists, then recreate it."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[CLEANUP] Removed existing directory: {path}")

    os.makedirs(path, exist_ok=True)
    print(f"[CLEANUP] Created directory: {path}")


COLOR_MAP = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "YELLOW": (0, 255, 255),
}


def draw_zones(frame):
    """Draw four colored quadrant guides on a frame."""
    h, w, _ = frame.shape
    thickness = 2
    cv2.rectangle(frame, (0, 0), (w // 2, h // 2), COLOR_MAP["RED"], thickness)
    cv2.rectangle(frame, (w // 2, 0), (w, h // 2), COLOR_MAP["GREEN"], thickness)
    cv2.rectangle(frame, (0, h // 2), (w // 2, h), COLOR_MAP["BLUE"], thickness)
    cv2.rectangle(frame, (w // 2, h // 2), (w, h), COLOR_MAP["YELLOW"], thickness)
    return frame
