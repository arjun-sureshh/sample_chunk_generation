import threading
import queue
from chunker import generate_chunks
from sampler import get_sampled_frames
from yolo_worker import yolo_worker 
from utils import clean_directory
import os
from dotenv import load_dotenv
from config import VIDEO
from config import PATHS

load_dotenv()


chunk_queue = queue.Queue()
yolo_queue = queue.Queue()

CHUNKS_DIR = PATHS["chunks_dir"]
FRAMES_DIR = PATHS["frames_dir"]
DETECTIONS_DIR = PATHS["detections_dir"]

print("[SYSTEM] Cleaning old outputs...")

clean_directory(CHUNKS_DIR)
clean_directory(FRAMES_DIR)
clean_directory(DETECTIONS_DIR)

print("[SYSTEM] Output directories are fresh")


video_path = VIDEO["path"]
chunk_duration = VIDEO["chunk_duration"]
chunk_overlap = VIDEO["chunk_overlap"]
frames_per_second = VIDEO["frames_per_second"]

t1 = threading.Thread(target=generate_chunks, args=(video_path, chunk_duration, chunk_overlap, chunk_queue))
t2 = threading.Thread(target=get_sampled_frames, args=(chunk_queue, yolo_queue, frames_per_second))
t3 = threading.Thread(target=yolo_worker, args=(yolo_queue,))

t1.start()
t2.start()
t3.start()

t1.join()
chunk_queue.put(None)

t2.join()
yolo_queue.put(None)

t3.join()
