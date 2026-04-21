import argparse
import json
from typing import Any
import PIL.Image
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

from SFTTrain import MODEL_PROMPT, explode_claim_level_rows, pick_device, pick_dtype, split_full_dataset


VALID_LABELS = {"CORRECT", "HALLUCINATED"}


def load_pipeline(args: argparse.Namespace, device: torch.device):
    dtype = pick_dtype(device)

    if args.model_type == "baseline":
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    else:
        from peft import PeftModel

        base_model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        try:
            processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    if hasattr(processor, "tokenizer") and getattr(processor, "tokenizer", None):
        processor.tokenizer.padding_side = "left"

    model = model.to(device)
    model.eval()

    return pipeline(
        task="image-text-to-text",
        model=model,
        processor=processor,
        dtype=dtype,
    )


def build_chat_input(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = MODEL_PROMPT.format(claim=str(example["claim"]).strip())
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": PIL.Image.open(str(example["img_path"])).convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def extract_output_text(pred: Any) -> str:
    if isinstance(pred, list) and pred:
        pred = pred[0]

    if isinstance(pred, dict):
        value = pred.get("generated_text", pred)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list) and value:
            last = value[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                    return " ".join(text_parts).strip()
                return str(content).strip()
            return str(last).strip()
        return str(value).strip()

    return str(pred).strip()


def normalize_predicted_label(text: str) -> str:
    normalized = text.strip().upper()
    if not normalized:
        return "EMPTY"
    if "HALLUCINATED" in normalized or "HALLUCINATION" in normalized:
        return "HALLUCINATED"
    if "CORRECT" in normalized:
        return "CORRECT"
    return "INVALID"


def compute_binary_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for truth, pred in zip(y_true, y_pred, strict=True):
        truth_pos = truth == "HALLUCINATED"
        pred_pos = pred == "HALLUCINATED"
        if truth_pos and pred_pos:
            tp += 1
        elif (not truth_pos) and pred_pos:
            fp += 1
        elif truth_pos and (not pred_pos):
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total = len(y_true)
    # Compute balance between HALLUCINATED and CORRECT labels
    n_positive_labels = sum(1 for label in y_true if label == "HALLUCINATED")
    n_negative_labels = sum(1 for label in y_true if label == "CORRECT")

    return {
        "precision": precision,
        "recall": recall,
        "n_samples": float(total),
        "n_positive_labels": float(n_positive_labels),
        "n_negative_labels": float(n_negative_labels),
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def compute_metrics_by_hallucination_type(records: list[dict[str, str]]) -> dict[str, Any]:
    per_type: dict[str, dict[str, Any]] = {}

    for record in records:
        hal_type = record["hal_type"]
        if hal_type not in per_type:
            per_type[hal_type] = {
                "y_true": [],
                "y_pred": [],
                "empty_outputs": 0,
                "invalid_outputs": 0,
                "valid_outputs": 0,
                "n_samples": 0,
            }

        row = per_type[hal_type]
        row["y_true"].append(record["expected_label"])
        row["y_pred"].append(record["predicted_label"])
        row["n_samples"] += 1

        if record["predicted_label"] == "EMPTY":
            row["empty_outputs"] += 1
        elif record["predicted_label"] == "INVALID":
            row["invalid_outputs"] += 1
        else:
            row["valid_outputs"] += 1

    summary: dict[str, Any] = {}
    for hal_type, row in per_type.items():
        n_samples = row["n_samples"]
        metrics = compute_binary_metrics(row["y_true"], row["y_pred"])
        summary[hal_type] = {
            "n_samples": n_samples,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "empty_outputs": row["empty_outputs"],
            "invalid_outputs": row["invalid_outputs"],
            "valid_outputs": row["valid_outputs"],
            "empty_rate": row["empty_outputs"] / n_samples if n_samples > 0 else 0.0,
            "invalid_rate": row["invalid_outputs"] / n_samples if n_samples > 0 else 0.0,
            "valid_rate": row["valid_outputs"] / n_samples if n_samples > 0 else 0.0,
            "n_positive_labels": metrics["n_positive_labels"],
            "n_negative_labels": metrics["n_negative_labels"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple pipeline-based test evaluator")
    parser.add_argument("--dataset-path", required=True, help="Path to JSON dataset")
    parser.add_argument(
        "--images-dir",
        default="data/VG/VG_500",
        help="Directory containing VG images (default: %(default)s).",
    )
    parser.add_argument(
        "--model-type",
        choices=["baseline", "sft", "sft-grpo"],
        required=True,
        help="Model type to evaluate.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base model name from HuggingFace (default: %(default)s).",
    )
    parser.add_argument(
        "--adapter-path",
        default="models/",
        help="Path to LoRA adapter for sft/sft-grpo (default: %(default)s).",
    )
    parser.add_argument(
        "--train-val-test-split",
        nargs=3,
        type=float,
        default=[0.70, 0.15, 0.15],
        help="Train/Validation/Test split ratios (must match training) (default: %(default)s).",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Random seed (must match training) (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Pipeline batch size for inference (default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Max new tokens generated per sample (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of test samples (0 means all).",
    )
    parser.add_argument(
        "--metrics-json",
        default="evaluate_metrics.json",
        help="Output path for metrics JSON (default: %(default)s).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dict = load_dataset("json", data_files=args.dataset_path, num_proc=1)
    dataset = explode_claim_level_rows(dataset_dict["train"], args.images_dir)
    split_dataset = split_full_dataset(
        dataset=dataset,
        split_ratios=list(args.train_val_test_split),
        seed=args.train_seed,
    )

    print(f"Split dataset: {split_dataset}")

    test_dataset = split_dataset["test"]
    if args.limit > 0:
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))

    print(f"Test samples: {len(test_dataset)}")

    device = pick_device()
    print(f"Using device: {device}")

    pipe = load_pipeline(args, device)
    records: list[dict[str, str]] = []

    total_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size
    for start in tqdm(
        range(0, len(test_dataset), args.batch_size),
        total=total_batches,
        desc="Running pipeline inference",
        unit="batch",
    ):
        end = min(start + args.batch_size, len(test_dataset))
        batch_examples = [test_dataset[i] for i in range(start, end)]
        batch_inputs = [build_chat_input(ex) for ex in batch_examples]
        outputs = pipe(
            batch_inputs,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            return_full_text=False,
        )

        for idx, output in enumerate(outputs):
            ex = batch_examples[idx]
            predicted = extract_output_text(output)
            predicted_label = normalize_predicted_label(predicted)
            expected = str(ex.get("label_text", "UNKNOWN")).strip().upper()

            if expected not in VALID_LABELS:
                expected = "CORRECT"

            records.append(
                {
                    "hal_type": str(ex.get("hal_type", "unknown")),
                    "expected_label": expected,
                    "predicted_label": predicted_label,
                    "raw_output": predicted,
                }
            )

    metrics = compute_metrics_by_hallucination_type(records)
    print("=" * 80)
    print("METRICS BY HALLUCINATION TYPE")
    print(json.dumps(metrics, indent=2))

    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.metrics_json}")


if __name__ == "__main__":
    main()
