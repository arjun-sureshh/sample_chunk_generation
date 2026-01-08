# chunker.py
import cv2
import os
import queue

def generate_chunks(video_path: str, chunk_duration: int, chunk_overlap: int, chunk_queue: queue.Queue):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"chunker.py fps:{fps}, total_frames:{total_frames}")

    frames_per_chunk = int(chunk_duration * fps)
    overlap_frames = int(chunk_overlap * fps)

    # Calculate chunk start indices upfront
    start_indices = list(range(0, total_frames, frames_per_chunk - overlap_frames))
    
    print(f"start indices :{start_indices}")

    for chunk_id, start_idx in enumerate(start_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []

        for _ in range(frames_per_chunk):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if not frames:
            continue

        # Save chunk video
        h, w, _ = frames[0].shape
        chunk_filename = f"chunk_{chunk_id}.mp4"
        chunk_path = os.path.join("result", "chunks", chunk_filename)
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)

        writer = cv2.VideoWriter(chunk_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        # Add chunk info to queue
        # chunk_queue.put({
        #     "chunk_id": str(uuid.uuid4()),
        #     "chunk_path": chunk_path,
        #     "video_path": video_path,
        #     "start_frame": start_idx,
        #     "end_frame": start_idx + len(frames) - 1
        # })
           
        chunk_queue.put({ 
            "chunk_id": f"chunk_{chunk_id}",
            "start_frames": start_idx,
            "chunk_path": chunk_path,
            "video_path": video_path,
            })


        print(f"[CHUNKER] Created chunk {chunk_path} with {len(frames)} frames")

    cap.release()
    print("[CHUNKER] All chunks generated")



