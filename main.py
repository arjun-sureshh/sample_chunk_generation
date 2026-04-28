"""Pipeline entrypoint for chunk generation and tracked video rendering."""

from __future__ import annotations

import json
import os
import queue
import threading

import cv2

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

from chunker import generate_chunks
from config import CONFIG, PATHS, VIDEO
from frame_processor import FrameProcessor
from utils import clean_directory
from utils.visualizer import FrameVisualizer


def _build_video_writer(output_path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    """Create a video writer for an annotated output chunk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def tracked_video_worker(chunk_queue: queue.Queue) -> None:
    """Consume chunk videos, run tracking, and save/display annotated video."""
    print("[TRACK] Video worker started")

    visualization_config = CONFIG.get("visualization", {})
    tracked_video_dir = str(visualization_config.get("tracked_video_dir", "result/tracked_videos"))
    annotated_frames_dir = str(visualization_config.get("annotated_frames_dir", "result/annotated"))
    display_video = bool(visualization_config.get("display_video", False))
    save_video = bool(visualization_config.get("save_video", True))

    processor = FrameProcessor()
    visualizer = FrameVisualizer(output_dir=annotated_frames_dir, zone_engine=processor.zone_engine)

    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            print("[TRACK] Video worker stopped")
            break

        chunk_id = chunk["chunk_id"]
        chunk_path = chunk["chunk_path"]
        chunk_start_frame = int(chunk["start_frames"])

        cap = cv2.VideoCapture(chunk_path)
        if not cap.isOpened():
            print(f"[TRACK ERROR] Cannot open chunk video {chunk_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path = os.path.join(tracked_video_dir, f"{chunk_id}_tracked.mp4")
        writer = _build_video_writer(output_path, fps, (width, height)) if save_video else None

        print(f"[TRACK] Processing {chunk_id} with {total_frames} frames at {fps:.2f} FPS")
        local_frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            absolute_frame_id = chunk_start_frame + local_frame_id
            timestamp = absolute_frame_id / fps if fps > 0 else 0.0

            frame_data = processor.process_frame(
                frame=frame,
                frame_id=absolute_frame_id,
                timestamp=timestamp,
            )
            annotated = visualizer.annotate(frame, frame_data)

            if writer is not None:
                visualizer.save_video_frame(writer, annotated)

            if display_video:
                try:
                    cv2.imshow("Retail Tracking", annotated)
                    delay = max(int(1000 / fps), 1)
                    key = cv2.waitKey(delay) & 0xFF
                    if key == ord("q"):
                        display_video = False
                        cv2.destroyAllWindows()
                except cv2.error as exc:
                    display_video = False
                    print(f"[TRACK] Live display disabled: {exc}")

            if local_frame_id % max(int(fps), 1) == 0:
                print(json.dumps(frame_data, indent=2))

            local_frame_id += 1

        cap.release()
        if writer is not None:
            writer.release()
            print(f"[TRACK] Saved tracked video to {output_path}")

    cv2.destroyAllWindows()


chunk_queue: queue.Queue = queue.Queue()

CHUNKS_DIR = PATHS["chunks_dir"]
FRAMES_DIR = PATHS["frames_dir"]
TRACKED_VIDEO_DIR = str(CONFIG.get("visualization", {}).get("tracked_video_dir", "result/tracked_videos"))
ANNOTATED_DIR = str(CONFIG.get("visualization", {}).get("annotated_frames_dir", "result/annotated"))

print("[SYSTEM] Cleaning old outputs...")
clean_directory(CHUNKS_DIR)
clean_directory(FRAMES_DIR)
clean_directory(TRACKED_VIDEO_DIR)
clean_directory(ANNOTATED_DIR)
print("[SYSTEM] Output directories are fresh")

video_path = VIDEO["path"]
chunk_duration = VIDEO["chunk_duration"]
chunk_overlap = VIDEO["chunk_overlap"]

t1 = threading.Thread(target=generate_chunks, args=(video_path, chunk_duration, chunk_overlap, chunk_queue))
t2 = threading.Thread(target=tracked_video_worker, args=(chunk_queue,))

t1.start()
t2.start()

t1.join()
chunk_queue.put(None)

t2.join()
