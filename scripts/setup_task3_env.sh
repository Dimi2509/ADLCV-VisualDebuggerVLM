#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/zhome/32/b/227378/ADLCV-VisualDebuggerVLM}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.12.11}"

cd "$PROJECT_DIR"

module load "$PYTHON_MODULE"

python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install \
  accelerate \
  transformers \
  pillow \
  tqdm \
  safetensors \
  sentencepiece \
  protobuf \
  peft \
  qwen-vl-utils

python - <<'PY'
import torch
import transformers
import accelerate
from transformers import AutoModelForImageTextToText, AutoProcessor

print("Task 3 environment ready")
print(f"torch={torch.__version__}, cuda_build={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}, accelerate={accelerate.__version__}")
print(f"AutoModelForImageTextToText={AutoModelForImageTextToText.__name__}, AutoProcessor={AutoProcessor.__name__}")
PY
