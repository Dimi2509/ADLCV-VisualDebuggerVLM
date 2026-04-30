#!/bin/bash
#BSUB -J task3_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 01:00
#BSUB -cwd /zhome/32/b/227378/ADLCV-VisualDebuggerVLM
#BSUB -o /zhome/32/b/227378/ADLCV-VisualDebuggerVLM/logs/task3_smoke_%J.out
#BSUB -e /zhome/32/b/227378/ADLCV-VisualDebuggerVLM/logs/task3_smoke_%J.err

set -euo pipefail

PROJECT_DIR="/zhome/32/b/227378/ADLCV-VisualDebuggerVLM"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.12.11}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

# Ensure output directories exist.
mkdir -p logs outputs

# Keep caches inside the project instead of the default ~/.cache.
# This prevents Hugging Face / pip / torch from silently filling ~/.cache.
mkdir -p "$PROJECT_DIR/.cache/huggingface"
mkdir -p "$PROJECT_DIR/.cache/pip"
mkdir -p "$PROJECT_DIR/.cache/torch"
mkdir -p "$PROJECT_DIR/.cache/matplotlib"
mkdir -p "$PROJECT_DIR/tmp"

export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export TMPDIR="$PROJECT_DIR/tmp"

# Make Python output appear immediately in the log file.
export PYTHONUNBUFFERED=1

module load "$PYTHON_MODULE"

if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
else
  echo "Venv not found at $VENV_DIR; using PYTHON_BIN=$PYTHON_BIN"
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
POPE_FILE="${POPE_FILE:-data/benchmark/pope/coco_pope_popular.json}"
IMAGE_ROOT="${IMAGE_ROOT:-data/benchmark/coco_subset}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
VERIFIER_MAX_NEW_TOKENS="${VERIFIER_MAX_NEW_TOKENS:-256}"
NUM_CANDIDATES="${NUM_CANDIDATES:-1}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/task3_smoke.jsonl}"
VERIFIER_ARGS="${VERIFIER_ARGS:-}"
TASK3_EXTRA_ARGS="${TASK3_EXTRA_ARGS:-}"

echo "Job started on: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "MODEL_NAME=$MODEL_NAME"
echo "VENV_DIR=$VENV_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "HF_HOME=$HF_HOME"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "TMPDIR=$TMPDIR"
echo "CUDA check:"
nvidia-smi || true
"$PYTHON_BIN" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    print(f"torch.cuda.device_name={torch.cuda.get_device_name(0)}")
PY
echo

"$PYTHON_BIN" task3_pipeline.py \
  --model-name "$MODEL_NAME" \
  $VERIFIER_ARGS \
  --pope-file "$POPE_FILE" \
  --image-root "$IMAGE_ROOT" \
  --max-samples "$MAX_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --verifier-max-new-tokens "$VERIFIER_MAX_NEW_TOKENS" \
  --num-candidates "$NUM_CANDIDATES" \
  --skip-missing-images \
  $TASK3_EXTRA_ARGS \
  --output "$OUTPUT_FILE"

echo
echo "Job finished at: $(date)"
