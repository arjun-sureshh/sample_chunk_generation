import threading
import queue
from chunker import generate_chunks
from sampler import get_sampled_frames
from vlm_worker import vlm_worker
from utils import clean_directory
import os
from dotenv import load_dotenv
from config import VIDEO
from config import PATHS

load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("❌ HF_TOKEN not found in .env file")

print("[SYSTEM] HuggingFace token loaded")


chunk_queue = queue.Queue()
vlm_queue = queue.Queue()

CHUNKS_DIR = PATHS["chunks_dir"]
FRAMES_DIR = PATHS["frames_dir"]

print("[SYSTEM] Cleaning old outputs...")

clean_directory(CHUNKS_DIR)
clean_directory(FRAMES_DIR)

print("[SYSTEM] Output directories are fresh")


video_path = VIDEO["path"]
chunk_duration = VIDEO["chunk_duration"]
chunk_overlap = VIDEO["chunk_overlap"]
frames_per_second = VIDEO["frames_per_second"]

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
