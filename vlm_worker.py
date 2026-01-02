from vlm_qwen import analyze_frames

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
            summary = analyze_frames(
                frames
            )

            print("\n" + "=" * 80)
            print(f"[VLM RESULT] Chunk {chunk_id}")
            print(summary)
            print("=" * 80)

        except Exception as e:
            print(f"[VLM ERROR] Chunk {chunk_id}: {e}")
