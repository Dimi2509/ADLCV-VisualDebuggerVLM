
# Task 3 pipeline

A drop-in script for the generate → verify → correct loop is available in
`task3_pipeline.py`, with usage notes in `docs/task3_pipeline.md`.

Run a quick single-image check:

```bash
python task3_pipeline.py \
  --image data/benchmark/coco_subset/COCO_val2014_000000310196.jpg \
  --model-name Qwen/Qwen3-VL-2B-Instruct
```

Run a no-GPU/no-model smoke test:

```bash
module load python3/3.12.11
source .venv/bin/activate

python task3_pipeline.py \
  --mock-backend \
  --pope-file data/benchmark/pope/coco_pope_popular.json \
  --image-root data/benchmark/coco_subset \
  --max-samples 1 \
  --num-candidates 2 \
  --skip-missing-images \
  --output outputs/task3_mock_smoke.jsonl
```

Set up the HPC environment and submit a one-image GPU smoke job:

```bash
bash scripts/setup_task3_env.sh
bash scripts/submit_task3_gpu.sh
```

For a faster first GPU test, use the smaller SmolVLM checkpoint:

```bash
MODEL_NAME=HuggingFaceTB/SmolVLM-500M-Instruct \
MAX_NEW_TOKENS=32 \
VERIFIER_MAX_NEW_TOKENS=128 \
OUTPUT_FILE=outputs/task3_gpu_smoke_smolvlm.jsonl \
bash scripts/submit_task3_gpu.sh
```

Run on the current POPE/COCO benchmark assets:

```bash
python task3_pipeline.py \
  --pope-file data/benchmark/pope/coco_pope_popular.json \
  --image-root data/benchmark/coco_subset \
  --max-samples 20 \
  --num-candidates 5 \
  --skip-missing-images \
  --output outputs/task3_popular.jsonl
```

When the Task 2 verifier is finished, pass it via `--verifier-model-name` or
`--verifier-adapter-path`.
