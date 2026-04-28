"""Suggest LAB A/B ranges for staff uniform classification from sample images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def get_dominant_lab(crop: np.ndarray) -> tuple[float, float]:
    """Return median A and B channel values for a BGR image."""
    if crop.size == 0:
        return 0.0, 0.0

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    return float(np.median(a_channel)), float(np.median(b_channel))


def _collect_image_paths(folder: Path) -> list[Path]:
    """Return image paths in a folder sorted by filename."""
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(folder.glob(pattern))
    return sorted(image_paths)


def tune_lab_values(folder: Path) -> int:
    """Compute and print suggested LAB range from staff samples."""
    image_paths = _collect_image_paths(folder)
    if not image_paths:
        print(f"No images found in: {folder}")
        return 1

    a_values: list[float] = []
    b_values: list[float] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            print(f"Skipping unreadable image: {image_path}")
            continue
        a_median, b_median = get_dominant_lab(image)
        a_values.append(a_median)
        b_values.append(b_median)
        print(f"{image_path.name}: median_a={a_median:.2f}, median_b={b_median:.2f}")

    if not a_values or not b_values:
        print("No valid images were processed.")
        return 1

    a_min = int(np.floor(min(a_values)))
    a_max = int(np.ceil(max(a_values)))
    b_min = int(np.floor(min(b_values)))
    b_max = int(np.ceil(max(b_values)))

    print("\nSuggested classification.staff_uniform_lab range:")
    print(f"  a_min: {a_min}")
    print(f"  a_max: {a_max}")
    print(f"  b_min: {b_min}")
    print(f"  b_max: {b_max}")
    return 0


def main() -> int:
    """Parse CLI args and run LAB range tuning."""
    parser = argparse.ArgumentParser(description="Tune LAB A/B ranges from staff image samples.")
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Folder containing staff sample images.",
    )
    args = parser.parse_args()
    return tune_lab_values(args.folder)


if __name__ == "__main__":
    raise SystemExit(main())
