#!/bin/bash
#BSUB -J task3_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 01:00
#BSUB -o logs/task3_smoke_%J.out
#BSUB -e logs/task3_smoke_%J.err

set -euo pipefail

cd /zhome/32/b/227378/ADLCV-VisualDebuggerVLM
mkdir -p logs outputs

VENV_DIR="${VENV_DIR:-/zhome/32/b/227378/ADLCV-VisualDebuggerVLM/.venv}"
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.12.11}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
