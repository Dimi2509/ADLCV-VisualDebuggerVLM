# Task 3 pipeline: generate → verify → correct

This is a scaffold for **Task 3: Closing the Loop**.

It is designed to work before the Task 2 verifier is finished:

1. Generate a response with the base VLM.
2. Verify the response claim-by-claim.
   - For now, the verifier can be the same model with a zero-shot prompt.
   - Later, pass the GRPO/SFT verifier checkpoint using `--verifier-model-name`
     and/or `--verifier-adapter-path`.
   - Use `--verifier-mode claim` for the Task 2 SFT/GRPO verifier, which was
     trained to classify one claim at a time as `CORRECT` or `HALLUCINATED`.
3. Correct only the claims marked `HALLUCINATED`.

## Single image

```bash
python task3_pipeline.py \
  --image data/benchmark/coco_subset/COCO_val2014_000000310196.jpg \
  --model-name Qwen/Qwen3-VL-2B-Instruct
```

## No-GPU smoke test

Use the mock backend to test the pipeline contract before the model or trained
verifier is available:

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

## HPC environment

The HPC login node does not expose `uv` on `PATH`, so the Task 3 environment is
created with `venv + pip` using the Python 3.12 module:

```bash
bash scripts/setup_task3_env.sh
```

This creates `.venv` in the repository and installs the CUDA 12.8 PyTorch
wheels, Transformers, Accelerate, PEFT, and Qwen VLM helpers.

Run the GPU smoke job on one A100 node:

```bash
bash scripts/submit_task3_gpu.sh
```

Check the job and logs:

```bash
bjobs <job-id>
tail -f logs/task3_smoke_<job-id>.out
tail -f logs/task3_smoke_<job-id>.err
```

Useful overrides:

```bash
MODEL_NAME=HuggingFaceTB/SmolVLM-500M-Instruct \
MAX_SAMPLES=1 \
MAX_NEW_TOKENS=32 \
VERIFIER_MAX_NEW_TOKENS=128 \
OUTPUT_FILE=outputs/task3_gpu_smoke_smolvlm.jsonl \
bash scripts/submit_task3_gpu.sh
```

For a no-download scheduler check, use the mock backend through the same LSF
script:

```bash
TASK3_EXTRA_ARGS="--mock-backend" bash scripts/submit_task3_gpu.sh
```

## Dataset run on the current POPE/COCO setup

```bash
python task3_pipeline.py \
  --pope-file data/benchmark/pope/coco_pope_popular.json \
  --image-root data/benchmark/coco_subset \
  --max-samples 20 \
  --skip-missing-images \
  --output outputs/task3_popular.jsonl
```

## Best-of-N baseline

Set `--num-candidates 5` to generate five sampled responses and choose the one
with the fewest verifier-flagged hallucinations.

```bash
python task3_pipeline.py \
  --pope-file data/benchmark/pope/coco_pope_popular.json \
  --image-root data/benchmark/coco_subset \
  --max-samples 20 \
  --num-candidates 5 \
  --skip-missing-images \
  --output outputs/task3_popular_best_of_5.jsonl
```

## When the verifier is ready

If Task 2 produces a full model checkpoint:

```bash
python task3_pipeline.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --verifier-model-name checkpoints/task2-verifier \
  --pope-file data/benchmark/pope/coco_pope_popular.json \
  --image-root data/benchmark/coco_subset \
  --skip-missing-images \
  --output outputs/task3_with_trained_verifier.jsonl
```

If Task 2 produces a LoRA/PEFT adapter:

```bash
python -m pip install peft

python task3_pipeline.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --verifier-model-name Qwen/Qwen3-VL-2B-Instruct \
  --verifier-adapter-path checkpoints/task2-verifier-lora \
  --verifier-mode claim \
  --pope-file data/benchmark/pope/coco_pope_popular.json \
  --image-root data/benchmark/coco_subset \
  --skip-missing-images \
  --output outputs/task3_with_lora_verifier.jsonl
```

The same claim mode matches the current SFT adapter layout on the `Task-2`
branch:

```bash
python task3_pipeline.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --verifier-model-name Qwen/Qwen3-VL-2B-Instruct \
  --verifier-adapter-path models/sft_model_20260425_112716 \
  --verifier-mode claim \
  --pope-file data/benchmark/pope/coco_pope_popular.json \
  --image-root data/benchmark/coco_subset \
  --skip-missing-images \
  --max-samples 20 \
  --output outputs/task3_with_sft_verifier.jsonl
```

## Output contract

Each JSONL record contains:

- `single_pass_response`: baseline generation.
- `verification`: claim-level verifier output.
- `corrected_response`: the generate-verify-correct result.
- `best_of_n_response`: optional best-of-N baseline if `--num-candidates > 1`.
- `meta`: model names, verifier checkpoint, latency, and POPE metadata.

Evaluation code can consume the JSONL file without changing the pipeline.

## Evaluate Task 3 outputs on POPE

After a Task 3 run, summarize verifier-flag metrics and POPE object-consistency
metrics:

```bash
python evaluate_task3_pope.py outputs/task3_popular_best_of_5.jsonl \
  --output-json outputs/task3_popular_best_of_5_metrics.json
```

The evaluator reports:

- verifier-flagged hallucinated claims per sample and per claim
- correction rate for `single_pass_response` → `corrected_response`
- average latency from the pipeline metadata
- POPE-style accuracy, precision, recall, F1, hallucination rate, and unknown
  rate for `single_pass_response`, `corrected_response`, and `best_of_n_response`

POPE object-consistency is a lightweight text check: it extracts the object from
the saved POPE question and checks whether each generated response mentions that
object. This is useful for fast iteration, but CHAIR or an LLM/VLM judge is still
better for final caption-level evaluation.

## Tests

The lightweight tests do not require Torch or a downloaded model:

```bash
python -m unittest discover -s tests -v
```

The LSF smoke script accepts the same mock backend through `TASK3_EXTRA_ARGS`:

```bash
TASK3_EXTRA_ARGS="--mock-backend" bash submit_task3_smoke.sh
```
