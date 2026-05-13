#!/bin/sh
### ============================================================
### LSF options for GRPO training on DTU HPC
### ============================================================

### -- Job name --
#BSUB -J grpo_smoke

### -- Queue: gpuv100 is usually the easiest GPU queue to get --
### -- For full runs, consider gpua100 (faster but harder to grab) --
#BSUB -q gpuv100

### -- Request 1 GPU in exclusive mode --
### -- exclusive_process is mandatory: shared GPUs get all jobs killed on OOM --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- 4 CPU cores (DTU HPC requires at least 4 cores per GPU) --
#BSUB -n 4

### -- All cores must be on the same node --
#BSUB -R "span[hosts=1]"

### -- 8GB RAM per core => 32GB total (enough for 2B model + ref model) --
#BSUB -R "rusage[mem=8GB]"

### -- Hard memory limit per core: kill the job if it exceeds 12GB/core --
### -- (~50% headroom over the requested 8GB) --
#BSUB -M 12GB

### -- Walltime: 1h for smoke test; bump to 6:00 or 12:00 for full runs --
### -- DTU GPU queues have a 24h hard cap --
#BSUB -W 1:00

### -- Email notifications at job start (-B) and completion (-N) --
### -- Replace with your own DTU student email --
#BSUB -u your_student_id@student.dtu.dk
#BSUB -B
#BSUB -N

### -- stdout / stderr files (%J expands to the job ID) --
#BSUB -o logs/grpo_%J.out
#BSUB -e logs/grpo_%J.err

### ============================================================
### Job script
### ============================================================

# Print environment info up front to make debugging easier
echo "=== Job started at $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo ""

# Load CUDA (Qwen3-VL-2B works best with CUDA 12.x + bf16)
# Run `module avail cuda` on the login node to see exact versions available
module load cuda/12.1

# Move into the repo (assumes you git-cloned it into $HOME)
cd $HOME/ADLCV-VisualDebuggerVLM

# Redirect HuggingFace cache to scratch so we don't blow the $HOME quota
# /work3/$USER is DTU's high-capacity scratch filesystem
export HF_HOME=/work3/$USER/hf_cache
mkdir -p $HF_HOME

# Smoke test run: 50 training samples, K=4 rollouts per prompt
# Adapter and metrics also go to scratch (LoRA adapters are ~70MB each)
uv run python GRPOTrain.py \
  --dataset-path generated_dataset.json \
  --sft-adapter-path models/sft_model_20260425_112716 \
  --max-train-samples 50 \
  --num-generations 4 \
  --logging-steps 1 \
  --model-output /work3/$USER/grpo_models \
  --metrics-dir /work3/$USER/grpo_metrics

echo "=== Job ended at $(date) ==="