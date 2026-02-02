import os
from vlm_processor import analyze_frame
import json 

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

                summary = analyze_frame(frame_path)
                txt_path = frame_path.replace(".jpg", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(summary)
                        f.write("\n" + str(fps))
                json_path = frame_path.replace(".jpg", ".json")
                data = {
                        "chunk_id": chunk_id,
                        "frame_name": os.path.basename(frame_path),
                        "fps": item["fps"],
                        "analysis": {
                            "phone_usage_detected": "phone_detected" in summary.lower(),
                            "staff_detected": "Staff ID 1" in summary.lower(),
                            "crowd_detected": "staff_crowd_detected" in summary.lower()
                        },
                        "raw_summary": summary
                    }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                    print(summary)
            except Exception as e:
                print(f"[VLM ERROR] {frame_path}: {e}")
