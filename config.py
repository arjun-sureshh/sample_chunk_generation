import yaml

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

VIDEO = CONFIG["video"]
PATHS = CONFIG["paths"]
MODEL = CONFIG["yolo"]
RESIZE = CONFIG["resize"]
FRAME_GENERATION = CONFIG["frame_generation"]
ZONE = CONFIG["zone"]