import os
import cv2
import queue  
from config import FRAME_GENERATION, RESIZE

h = RESIZE["height"]
w = RESIZE["width"]
mode = FRAME_GENERATION["mode"]

def get_sampled_frames(chunk_queue: queue.Queue, vlm_queue: queue.Queue, frames_per_second):
    
    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break

        video_path = chunk["chunk_path"]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"sampler.py Video FPS::{fps}")

        if mode == "fps":
            interval = max(int(fps / frames_per_second), 1)
            print(f"[SAMPLER] Mode: FPS | Interval: {interval}")

        elif mode == "all":
            interval = 1
            print(f"[SAMPLER] Mode: ALL | Interval: {interval}")

        else:
            raise ValueError(f"Unknown frame generation mode: {mode}")

        sampled_frames = []
        frame_names = []
        frame_id = 0

        frame_dir = os.path.join("result", "frames", chunk["chunk_id"])
        os.makedirs(frame_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % interval == 0:
                # Resize BEFORE storing or saving
                if RESIZE["enabled"]:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)

                sampled_frames.append(frame)
                frame_name= os.path.join(frame_dir, f"frame_{frame_id}.jpg")
                cv2.imwrite(frame_name, frame)
                frame_names.append(frame_name)
            # else:
            #     print(f"skipping frame {frame_id},{interval}")

            frame_id += 1

        cap.release()

        vlm_queue.put({
            "chunk_id": chunk["chunk_id"],
            "chunk_path": chunk["chunk_path"],
            "video_path": chunk["video_path"],
            "start_frame": chunk["start_frames"],
            # "end_frame": chunk["end_frame"],
            "fps": fps,
            "frames": sampled_frames,
            "frame_names": frame_names
        })

        print(f"[SAMPLER] Sampled {len(sampled_frames)} frames from chunk {chunk['chunk_id']}")
