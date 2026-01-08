import os
from vlm_processor import analyze_frame

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

                print(f"[VLM] Saved → {txt_path}")

            except Exception as e:
                print(f"[VLM ERROR] {frame_path}: {e}")
