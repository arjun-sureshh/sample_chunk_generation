import os, json

def build_index():
   

    JSON_DIR = "result/summaries_json"
    INDEX_FILE = "result/summaries_index.json"

    frames = []

    for file in os.listdir(JSON_DIR):
        if file.endswith(".json"):
            with open(os.path.join(JSON_DIR, file), "r") as f:
                frames.append(json.load(f))

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(frames, f, indent=4)

    print(f"[INDEX READY] {len(frames)} frames indexed")
