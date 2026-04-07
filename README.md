# ADLCV-VisualDebuggerVLM
Final project for the Advanced Deep Learning in Computer Vision course.


# Environment setup
We will be using uv as the python package manager. To set up the environment, run the following command in the terminal:

To install the dependencies, run:
```bash
uv sync --extra cpu
```
If you have a compatible NVIDIA GPU and want to install torch with CUDA support, run:
```bash
uv sync --extra cu128
```

## macOS and Apple GPU (MPS)

On MacBook, use the CPU extra:
```bash
uv sync --extra cpu
```

Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens) to create an access token. Then, set the token using the following command in the terminal:
```bash
uv run hf auth login
uv run hf auth whoami # to verify that the token is set correctly
```

# Benchmark 
In order to select a VLM, we will run a benchmark script to evaluate certain metrics for multiple vision-language models on a subset of the POPE benchmark dataset. The script will evaluate metrics such as accuracy, precision, recall, F1 score, hallucination rate, and latency for each model.

To download the benchmark dataset, run the following command in the terminal, specifying the numer of parallel download jobs:
```bash
BENCHMARK_DOWNLOAD_JOBS=8 uv run bash scripts/get_benchmark_dataset.sh 
```

# Model benchmark script
The `modelBenchmarks.py` script benchmarks multiple vision-language models on the POPE benchmark split and saves evaluation metrics to a JSON file.

What it does:
- Loads one of the POPE splits (`random`, `popular`, or `adversarial`).
- Evaluates each model in `MODEL_NAMES` on the selected samples.
- Prompts each model to answer each question with only `yes` or `no`.
- Computes summary metrics such as accuracy, precision, recall, F1, hallucination rate, and average latency.
- Writes all model results to an output JSON file (default: `benchmark_results.json`).

## Arguments
`modelBenchmarks.py` supports the following CLI arguments:

- `--pope-split {random,popular,adversarial}`
	- POPE split to evaluate.
	- Default: `random`.
- `--max-samples INT`
	- Maximum number of samples to evaluate.
	- Default: `500`.
- `--output PATH`
	- Path to output JSON file containing benchmark results.
	- Default: `benchmark_results.json`.
- `--image-root PATH`
	- Root directory containing benchmark images.
	- Default: `data/benchmark/coco_subset`.

## Example usage
Run with defaults:
```bash
uv run modelBenchmarks.py
```

Evaluate only 100 samples from the `popular` split:
```bash
uv run modelBenchmarks.py --pope-split popular --max-samples 100
```

Save output to a custom file:
```bash
uv run modelBenchmarks.py --output results_popular_100.json
```

Use a custom image root directory:
```bash
uv run modelBenchmarks.py --image-root /path/to/coco_subset
```

