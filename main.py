import threading
import queue
from chunker import generate_chunks
from sampler import get_sampled_frames
from vlm_worker import vlm_worker
from utils import clean_directory
import os
from dotenv import load_dotenv


load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("❌ HF_TOKEN not found in .env file")

print("[SYSTEM] HuggingFace token loaded")


chunk_queue = queue.Queue()
vlm_queue = queue.Queue()

CHUNKS_DIR = "result/chunks"
FRAMES_DIR = "result/frames"

print("[SYSTEM] Cleaning old outputs...")

clean_directory(CHUNKS_DIR)
clean_directory(FRAMES_DIR)

print("[SYSTEM] Output directories are fresh")


video_path = os.getenv("video_path")
chunk_duration = 30
chunk_overlap = 2
frames_per_second = 1

t1 = threading.Thread(target=generate_chunks, args=(video_path, chunk_duration, chunk_overlap, chunk_queue))
t2 = threading.Thread(target=get_sampled_frames, args=(chunk_queue, vlm_queue, frames_per_second))
t3 = threading.Thread(target=vlm_worker, args=(vlm_queue,))

t1.start()
t2.start()
t3.start()

t1.join()
chunk_queue.put(None)

t2.join()
vlm_queue.put(None)

t3.join()
