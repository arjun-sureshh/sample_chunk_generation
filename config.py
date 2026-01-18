import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

VIDEO = CONFIG["video"]
MODEL = CONFIG["model"]
PATHS = CONFIG["paths"]
VLM = CONFIG["vlm"]
RESIZE = CONFIG["resize"]
FRAME_GENERATION = CONFIG.get("frame_generation", {"mode": "fps"})
