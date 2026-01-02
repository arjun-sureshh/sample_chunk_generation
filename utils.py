import shutil
import os

def clean_directory(path: str):
    """
    Deletes a directory if it exists, then recreates it.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[CLEANUP] Removed existing directory: {path}")

    os.makedirs(path, exist_ok=True)
    print(f"[CLEANUP] Created directory: {path}")
