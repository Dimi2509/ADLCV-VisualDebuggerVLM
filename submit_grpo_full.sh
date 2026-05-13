#!/bin/sh
### ============================================================
### LSF options for FULL GRPO training on DTU HPC
### ============================================================

### -- Job name --
#BSUB -J grpo_full

### -- Queue: gpua100 if available (faster), fallback to gpuv100 --
#BSUB -q gpuv100

### -- 1 GPU, exclusive --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- 4 CPU cores (DTU minimum for GPU jobs) --
#BSUB -n 4

### -- All cores on the same node --
#BSUB -R "span[hosts=1]"

### -- 8GB RAM per core => 32GB total --
#BSUB -R "rusage[mem=8GB]"

### -- Hard memory limit per core --
#BSUB -M 12GB

### -- Walltime: 18 hours (DTU GPU cap is 24h, leave headroom) --
#BSUB -W 18:00

### -- Email notifications --
#BSUB -u s232268@student.dtu.dk
#BSUB -B
#BSUB -N

### -- stdout / stderr files --
#BSUB -o logs/grpo_full_%J.out
#BSUB -e logs/grpo_full_%J.err

### ============================================================
### Job script
### ============================================================

echo "=== Job started at $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

# Load CUDA
module load cuda/12.1

cd $HOME/ADLCV-VisualDebuggerVLM

# HuggingFace cache to scratch (or to a roomy home subdir if no scratch)
export HF_HOME=/work3/$USER/hf_cache
mkdir -p $HF_HOME

# === FULL TRAINING RUN ===
# Key differences vs smoke test:
#   - No --max-train-samples (use the full ~8615 train samples)
#   - --save-strategy epoch (keep checkpoint per epoch in case of crash)
#   - --logging-steps 25 (don't spam logs every step)
#   - --num-train-epochs 1 (GRPO usually needs only 1 epoch)
uv run python GRPOTrain.py \
  --dataset-path generated_dataset.json \
  --sft-adapter-path models/sft_model_20260425_112716 \
  --num-generations 4 \
  --beta 0.04 \
  --lr 1e-6 \
  --temperature 0.7 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --logging-steps 25 \
  --save-strategy epoch \
  --model-output /work3/$USER/grpo_models \
  --metrics-dir /work3/$USER/grpo_metrics \
  --model-name grpo_full_${LSB_JOBID}

# === Run evaluation right after training, in the same job ===
# Saves you from queueing a separate job for eval
uv run python EvaluateModels.py \
  --dataset-path generated_dataset.json \
  --model-type sft-grpo \
  --adapter-path /work3/$USER/grpo_models/grpo_full_${LSB_JOBID} \
  --metrics-json /work3/$USER/grpo_metrics/grpo_full_${LSB_JOBID}/grpo_evaluate_metrics.json

echo "=== Job ended at $(date) ==="