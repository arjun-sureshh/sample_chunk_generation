import os
from vlm_qwen import analyze_frames

SUMMARY_DIR = "result/summaries"


def save_summary(chunk_id, text):
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    path = os.path.join(SUMMARY_DIR, f"{chunk_id}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[VLM] Summary saved → {path}")


def vlm_worker(vlm_queue):
    print("[VLM] Worker started")

    while True:
        item = vlm_queue.get()

        if item is None:
            print("[VLM] Worker stopped")
            break

        chunk_id = item["chunk_id"]
        frames = item["frames"]

        print(f"[VLM] Processing chunk {chunk_id} with {len(frames)} frames")

        try:
            summary = analyze_frames(frames)

            # Print to terminal
            print("\n" + "=" * 80)
            print(f"[VLM RESULT] Chunk {chunk_id}")
            print(summary)
            print("=" * 80)

            # Save to disk
            save_summary(chunk_id, summary)

        except Exception as e:
            print(f"[VLM ERROR] Chunk {chunk_id}: {e}")
