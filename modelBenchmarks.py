import argparse
import json
import gc
import os
import re
import time
from statistics import mean
from typing import Dict, List

import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

MODEL_NAMES = [
    # "HuggingFaceTB/SmolVLM-500M-Instruct",
    # "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
]
DEVICE = Accelerator().device
DEVICE_TYPE = DEVICE.type


def pick_model_dtype(device_type: str) -> torch.dtype:
    # Use fp16 on CUDA for speed/memory and fp32 elsewhere for compatibility.
    if device_type == "cuda":
        return torch.float16
    return torch.float32


MODEL_DTYPE = pick_model_dtype(DEVICE_TYPE)

POPE_DIR = "data/benchmark/pope"
DEFAULT_SPLIT = "popular"
DEFAULT_POPE_FILE = os.path.join(POPE_DIR, f"coco_pope_{DEFAULT_SPLIT}.json")
DEFAULT_IMAGE_ROOT = "data/benchmark/coco_subset"
DEFAULT_OUTPUT_FILE = "benchmark_results.json"
DEFAULT_MAX_SAMPLES = 500
MAX_NEW_TOKENS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pope-split",
        type=str,
        default="random",
        choices=["random", "popular", "adversarial"],
        help="Split of POPE dataset to use (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum number of POPE samples to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Where to save benchmark results JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=DEFAULT_IMAGE_ROOT,
        help="Root directory for images (default: %(default)s).",
    )
    return parser.parse_args()


def load_pope(file_path: str, max_samples: int | None = None) -> List[Dict]:
    # POPE files are JSONL/NDJSON: one JSON object per line.
    data: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    if max_samples is not None:
        data = data[:max_samples]
    print(f"Loaded {len(data)} samples from {file_path}")
    return data


def build_prompt(prompt: str, image_path: str) -> str:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {
                    "type": "text",
                    "text": f"Answer with only 'yes' or 'no'.\nQuestion:{prompt}",
                },
            ],
        }
    ]


def parse_yes_no(text: str) -> str:
    lowered = text.lower()
    matches = list(re.finditer(r"\b(yes|no)\b", lowered))
    if not matches:
        return "unknown"
    return matches[0].group(1)


def run_inference(model, processor, model_input) -> List[str]:
    try:
        input_len = len(model_input.input_ids[0])
        with torch.no_grad():
            output_ids = model.generate(**model_input, max_new_tokens=MAX_NEW_TOKENS)
        text = processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )  # Model outputs the prompt, so we only decode the generated part which starts after the len of the input.
        del output_ids
        return text[0]
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return [f"error: {str(e)}"]


def compute_metrics(results: List[Dict]) -> Dict:
    tp = fp = tn = fn = unknown = 0

    for r in results:
        pred = r["pred"]
        gt = r["gt"]

        if pred not in {"yes", "no"}:
            unknown += 1
            if gt == "yes":
                fn += 1
            else:
                tn += 1
            continue

        if gt == "yes":
            if pred == "yes":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "yes":
                fp += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    hallucination_rate = fp / max(fp + tn, 1)
    unknown_rate = unknown / max(total, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": hallucination_rate,
        "unknown_rate": unknown_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_model(model_name: str, data: List[Dict], image_root: str) -> Dict:
    print(f"\nLoading model: {model_name}")
    load_start = time.time()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=MODEL_DTYPE,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    load_time = time.time() - load_start

    if DEVICE_TYPE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    results: List[Dict] = []
    latencies: List[float] = []
    skipped_missing_images = 0

    for sample in tqdm(data, desc=model_name):
        image_name = sample["image"]
        image_path = os.path.join(image_root, image_name)
        question = sample["text"]
        gt = sample["label"].strip().lower()

        if not os.path.exists(image_path):
            skipped_missing_images += 1
            continue

        prompt = build_prompt(question, image_path)
        model_input = processor.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE)

        start = time.time()
        output = run_inference(model, processor, model_input)
        # print(
        #     f"Sample: {image_name}, GT: {gt}, Prompt: {prompt}, Pred: {output}, Latency: {time.time() - start:.4f}s"
        # )
        latency = time.time() - start
        latencies.append(latency)

        pred = parse_yes_no(output)
        results.append({"pred": pred, "gt": gt, "latency": latency})

        del model_input

    metrics = compute_metrics(results)
    metrics.update(
        {
            "model": model_name,
            "samples_evaluated": len(results),
            "samples_missing_images": skipped_missing_images,
            "load_time_sec": load_time,
            "avg_latency_sec": mean(latencies) if latencies else 0.0,
        }
    )

    del model
    del processor
    if DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

    return metrics


def print_summary(all_metrics: List[Dict]) -> None:
    print("\n=== Benchmark Summary ===")
    print(f"{'Model':45} {'Acc':>7} {'F1':>7} {'Halluc':>8} {'AvgLat(s)':>10}")
    print("-" * 100)
    for m in all_metrics:
        model_short = m["model"][:45]
        print(
            f"{model_short:45} "
            f"{m['accuracy']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{m['hallucination_rate']:>8.4f} "
            f"{m['avg_latency_sec']:>10.4f} "
        )


def main() -> None:
    args = parse_args()
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{args.pope_split}.json")

    print(f"Device: {DEVICE}")
    print(f"Device type: {DEVICE_TYPE}")
    print(f"Model dtype: {MODEL_DTYPE}")
    print(f"POPE file: {pope_file}")
    print(f"Image root: {args.image_root}")
    print(f"Max samples: {args.max_samples}")

    if not os.path.exists(pope_file):
        raise FileNotFoundError(f"POPE file not found: {pope_file}")
    if not os.path.isdir(args.image_root):
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    data = load_pope(pope_file, args.max_samples)
    print(f"Loaded {len(data)} samples")

    all_metrics: List[Dict] = []
    for model_name in MODEL_NAMES:
        try:
            metrics = evaluate_model(model_name, data, args.image_root)
            all_metrics.append(metrics)
            print(f"Finished: {model_name}")
        except Exception as e:
            print(f"Failed model {model_name}: {e}")

    if not all_metrics:
        raise RuntimeError("No model completed successfully.")

    print_summary(all_metrics)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
