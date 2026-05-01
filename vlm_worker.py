import os
from config import PATHS
from vlm_processor import analyze_frame
import json 
import time
from datetime import datetime

JSON_DIR =PATHS["json_dir"]
os.makedirs(JSON_DIR, exist_ok=True)

def vlm_worker(vlm_queue):
    print("[VLM] Worker started")

    while True:
        item = vlm_queue.get()

        if item is None:
            print("[VLM] Worker stopped")
            break

        chunk_id = item["chunk_id"]
        frame_names = item["frame_names"]
        frames = item["frames"]
        fps = {"fps":item['fps']}

        print(f"[VLM] Processing {len(frame_names)} frames in {chunk_id}")

        for frame_path in frame_names:
            try:
                print(f"[VLM] Analyzing {frame_path}")

                start_time = time.perf_counter()  #time tracking start

                summary = analyze_frame(frame_path)

                end_time = time.perf_counter() #time tracking end
                processing_time = round(end_time - start_time)  # seconds

                print(f"[VLM] Frame processed in {processing_time}s")

                txt_path = frame_path.replace(".jpg", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(summary)
                        f.write("\n" + str(fps))
                        f.write(f"\nProcessing Time (seconds): {processing_time}")
                json_filename = (
                    f"{chunk_id}_"
                    f"{os.path.basename(frame_path).replace('.jpg','.json')}"
                )
                json_path = os.path.join(JSON_DIR, json_filename)
                data = {
                       "video_id": os.path.basename(item["video_path"]),
                        "chunk_id": chunk_id,
                        "frame_name": os.path.basename(frame_path),
                        "frame_path": frame_path,

                        "fps": item["fps"],
                        "processing_time_seconds": processing_time,

                        "violations": {
                            "phone": "phone_detected" in summary.lower(),
                            "staff": "staff id" in summary.lower(),
                            "crowd": "crowd" in summary.lower()
                        },

                        "raw_summary": summary,
                        "created_at": datetime.utcnow().isoformat()
                    }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                    print(summary)
            except Exception as e:
                print(f"[VLM ERROR] {frame_path}: {e}")
