import os
import cv2
import queue  


def get_sampled_frames(chunk_queue: queue.Queue, vlm_queue: queue.Queue, frames_per_second):
    
    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break

        video_path = chunk["chunk_path"]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        interval = max(int(fps / frames_per_second), 1)

        sampled_frames = []
        frame_id = 0

        frame_dir = os.path.join("result", "frames", chunk["chunk_id"])
        os.makedirs(frame_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % interval == 0:
                sampled_frames.append(frame)
                cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_id}.jpg"), frame)

            frame_id += 1

        cap.release()

        vlm_queue.put({
            "chunk_id": chunk["chunk_id"],
            "chunk_path": chunk["chunk_path"],
            "video_path": chunk["video_path"],
            "start_frame": chunk["start_frame"],
            "end_frame": chunk["end_frame"],
            "fps": fps,
            "frames": sampled_frames
        })

        print(f"[SAMPLER] Sampled {len(sampled_frames)} frames from chunk {chunk['chunk_id']}")
