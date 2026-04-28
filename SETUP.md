# Setup Instructions

## Requirements
- Python 3.10 or 3.11
- Windows/Linux/Mac
- GPU recommended but CPU works

## Installation Steps

### Step 1: Create virtual environment
python -m venv venv

### Step 2: Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

### Step 3: Install all dependencies
pip install -r requirements.txt

### Step 4: Download YOLOv8 model (auto on first run)
# yolov8n.pt downloads automatically

### Step 5: Set up HuggingFace token for Qwen
# Create .env file in project root:
HF_TOKEN=your_token_here
# Get token from: https://huggingface.co/settings/tokens

### Step 6: Run the pipeline
python main.py

## Troubleshooting
- If torch install is slow: pip install torch --index-url https://download.pytorch.org/whl/cpu
- If transformers error: pip install transformers accelerate
- If CUDA error: set device: cpu in config.yaml

## Notes
- `sqlite3` is built into Python and does not need to be added to `requirements.txt`.
