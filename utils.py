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


def crop_person(frame, bbox, out_dir, person_id):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    path = os.path.join(out_dir, f"person_{person_id}.jpg")
    cv2.imwrite(path, crop)
    return path