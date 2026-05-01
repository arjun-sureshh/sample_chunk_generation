import json, os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "result", "summaries_index.json")

def load_frames():
    if not os.path.exists(INDEX_FILE):
        return []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
