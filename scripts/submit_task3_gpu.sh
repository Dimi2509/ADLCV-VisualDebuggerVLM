#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/zhome/32/b/227378/ADLCV-VisualDebuggerVLM}"
cd "$PROJECT_DIR"

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
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.12.11}"

export MODEL_NAME
export POPE_FILE
export IMAGE_ROOT
export MAX_SAMPLES
export MAX_NEW_TOKENS
export VERIFIER_MAX_NEW_TOKENS
export NUM_CANDIDATES
export OUTPUT_FILE
export VERIFIER_ARGS
export TASK3_EXTRA_ARGS
export VENV_DIR
export PYTHON_MODULE

bsub -env all < submit_task3_smoke.sh
