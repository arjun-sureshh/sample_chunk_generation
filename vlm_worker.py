import os
import json
from datetime import datetime
from vlm_qwen import analyze_frames

SUMMARY_FILE = "result/video_analysis.json"


def load_existing_summaries():
    
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "video_path": None,
        "total_chunks": 0,
        "analysis_timestamp": None,
        "chunks": []
    }


def save_summary(chunk_id, chunk_path, video_path, start_frame, end_frame, fps, summary_text):
    
    
    # Load existing data
    data = load_existing_summaries()
    
    # Update metadata
    if data["video_path"] is None:
        data["video_path"] = video_path
    data["total_chunks"] = len(data["chunks"]) + 1
    data["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate time in seconds
    start_time = start_frame / fps if fps > 0 else 0
    end_time = end_frame / fps if fps > 0 else 0
    
    # Create chunk entry
    chunk_entry = {
        "chunk_id": chunk_id,
        "chunk_number": data["total_chunks"],
        "chunk_path": chunk_path,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_seconds": round(start_time, 2),
        "end_time_seconds": round(end_time, 2),
        "duration_seconds": round(end_time - start_time, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary_text
    }
    
    # Add to chunks list
    data["chunks"].append(chunk_entry)
    
    # Save to file
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[VLM] Summary saved to {SUMMARY_FILE}")


def vlm_worker(vlm_queue):
    print("[VLM] Worker started")

    while True:
        item = vlm_queue.get()

        if item is None:
            print("[VLM] Worker stopped")
            print(f"[VLM] All summaries saved in: {SUMMARY_FILE}")
            break

        chunk_id = item["chunk_id"]
        frames = item["frames"]
        video_path = item.get("video_path", "unknown")
        chunk_path = item.get("chunk_path", "unknown")
        start_frame = item.get("start_frame", 0)
        end_frame = item.get("end_frame", 0)
        fps = item.get("fps", 30)

        print(f"[VLM] Processing chunk {chunk_id} with {len(frames)} frames")

        try:
            summary = analyze_frames(frames)

            # Print to terminal
            print("\n" + "=" * 80)
            print(f"[VLM RESULT] Chunk {chunk_id}")
            print(summary)
            print("=" * 80)

            # Save to consolidated JSON
            save_summary(
                chunk_id=chunk_id,
                chunk_path=chunk_path,
                video_path=video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                summary_text=summary
            )

        except Exception as e:
            print(f"[VLM ERROR] Chunk {chunk_id}: {e}")