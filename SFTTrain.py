import argparse
import gc
import json
import locale
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict, Value, load_dataset
from tqdm import tqdm
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          TrainerCallback)

MODEL_PROMPT = (
    "You are an expert assistant for determining whether a claim is correct or incorrect based on the provided image and claim.\n"
    "Your answer must be exactly one token: CORRECT or HALLUCINATED.\n"
    "Claim: {claim}\n"
)

LABEL_MAP = {
    "CORRECT": "CORRECT",
    "Correct": "CORRECT",
    "correct": "CORRECT",
    "HALLUCINATED": "HALLUCINATED",
    "Hallucinated": "HALLUCINATED",
    "hallucinated": "HALLUCINATED",
}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_vlm(model_name: str, device: torch.device) -> tuple[Any, Any]:
    dtype = pick_dtype(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and getattr(processor, "tokenizer", None):
        processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.train()
    return model, processor


def unload_vlm(model: Any, processor: Any, device: torch.device) -> None:
    del model
    del processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_message(image_path: str, prompt: str) -> list[dict[str, Any]]:
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
                    "text": f"{prompt}",
                },
            ],
        }
    ]


def normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    return LABEL_MAP.get(str(label).strip())


def parse_predicted_label(text: str) -> str:
    norm = text.strip().upper()
    if "HALLUCINATED" in norm:
        return "HALLUCINATED"
    if "CORRECT" in norm:
        return "CORRECT"
    return "UNKNOWN"


def resolve_image_path(img_path: str, images_dir: str) -> str | None:
    path = Path(img_path)
    if path.exists():
        return path.resolve().as_posix()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path.as_posix()

    fallback = (Path(images_dir) / path.name).resolve()
    if fallback.exists():
        return fallback.as_posix()

    return None


def get_pad_token_id(processor: Any) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return 0
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    return 0


def build_collator(processor: Any, max_seq_length: int):
    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        user_messages: list[list[dict[str, Any]]] = []
        full_messages: list[list[dict[str, Any]]] = []

        for feature in features:
            claim = str(feature["claim"]).strip()
            image_path = str(feature["img_path"])
            label_text = str(feature["label_text"])

            prompt = MODEL_PROMPT.format(claim=claim)
            user_message = build_message(image_path, prompt)
            full_message = user_message + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label_text}],
                }
            ]

            user_messages.append(user_message)
            full_messages.append(full_message)

        prompt_inputs = processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        full_inputs = processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        labels = full_inputs["input_ids"].clone()

        for idx, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[idx, : int(prompt_len)] = -100

        labels[full_inputs["attention_mask"] == 0] = -100
        full_inputs["labels"] = labels

        return full_inputs

    return collate


def explode_claim_level_rows(dataset: Dataset, images_dir: str) -> Dataset:
    rows: list[dict[str, Any]] = []
    dropped = 0

    for item in dataset:
        image_id = (
            int(item.get("image_id")) if item.get("image_id") is not None else None
        )
        img_path = item.get("img_path")
        claims = item.get("claims")
        claim_labels = item.get("claim_labels")

        if image_id is None or img_path is None:
            dropped += 1
            continue

        resolved_path = resolve_image_path(str(img_path), images_dir)
        if resolved_path is None:
            dropped += 1
            continue

        if isinstance(claims, list) and isinstance(claim_labels, list):
            for claim, raw_label in zip(claims, claim_labels, strict=False):
                label_text = normalize_label(raw_label)
                if label_text is None or not str(claim).strip():
                    dropped += 1
                    continue
                rows.append(
                    {
                        "image_id": image_id,
                        "img_path": resolved_path,
                        "claim": str(claim).strip(),
                        "label_text": label_text,
                        "hal_type": str(item.get("hal_type", "unknown")),
                    }
                )
            continue

        claim = item.get("claim")
        raw_label = item.get("label") or item.get("label_text")
        label_text = normalize_label(raw_label)
        if label_text is None or claim is None or not str(claim).strip():
            dropped += 1
            continue

        rows.append(
            {
                "image_id": image_id,
                "img_path": resolved_path,
                "claim": str(claim).strip(),
                "label_text": label_text,
                "hal_type": str(item.get("hal_type", "unknown")),
            }
        )

    if not rows:
        raise ValueError("No valid claim-level samples found after preprocessing")

    if dropped > 0:
        print(f"Dropped {dropped} invalid samples during claim-level preprocessing")

    flattened = Dataset.from_list(rows)
    flattened = flattened.cast_column("image_id", Value("int32"))
    return flattened


def split_full_dataset(
    dataset: Dataset,
    split_ratios: list[float],
    seed: int,
) -> DatasetDict:
    """Split a dataset into train/val/test by image_id to avoid leakage.

    All rows that belong to the same image_id stay in the same split.
    """
    if len(split_ratios) != 3:
        raise ValueError("train/val/test split must contain exactly 3 ratios")

    total = sum(split_ratios)
    if total <= 0:
        raise ValueError("train/val/test split ratios must sum to a positive number")

    train_ratio, val_ratio, test_ratio = (r / total for r in split_ratios)

    if "image_id" not in dataset.column_names:
        raise ValueError("Dataset must contain an 'image_id' column before splitting")

    image_to_indices: dict[int, list[int]] = {}
    for idx, image_id in enumerate(dataset["image_id"]):
        image_to_indices.setdefault(int(image_id), []).append(idx)

    image_ids = list(image_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n_images = len(image_ids)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    n_test = n_images - n_train - n_val

    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train : n_train + n_val])
    test_ids = set(image_ids[n_train + n_val :])

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for image_id, indices in image_to_indices.items():
        if image_id in train_ids:
            train_indices.extend(indices)
        elif image_id in val_ids:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    split_dataset = DatasetDict(
        {
            "train": dataset.select(train_indices),
            "validation": dataset.select(val_indices),
            "test": dataset.select(test_indices),
        }
    )

    return split_dataset


def import_trl_sft() -> tuple[Any, Any]:
    original_getpreferredencoding = locale.getpreferredencoding
    original_read_text = Path.read_text

    def _utf8_preferred_encoding(do_setlocale: bool = True) -> str:
        return "utf-8"

    def _read_text_utf8(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        resolved_encoding = encoding if encoding is not None else "utf-8"
        return original_read_text(self, encoding=resolved_encoding, errors=errors)

    locale.getpreferredencoding = _utf8_preferred_encoding
    Path.read_text = _read_text_utf8
    try:
        from trl import SFTConfig, SFTTrainer

        return SFTConfig, SFTTrainer
    finally:
        locale.getpreferredencoding = original_getpreferredencoding
        Path.read_text = original_read_text


def import_lora_config() -> Any:
    try:
        from peft import LoraConfig

        return LoraConfig
    except ImportError as exc:
        raise ImportError(
            "LoRA requires the 'peft' package. Install it with: uv add peft"
        ) from exc


def compute_binary_metrics(
    y_true: list[str],
    y_pred: list[str],
    positive_label: str = "HALLUCINATED",
) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")

    tp = fp = fn = tn = 0
    unknown = 0

    for truth, pred in zip(y_true, y_pred, strict=True):
        if pred == "UNKNOWN":
            unknown += 1
        truth_pos = truth == positive_label
        pred_pos = pred == positive_label
        if truth_pos and pred_pos:
            tp += 1
        elif (not truth_pos) and pred_pos:
            fp += 1
        elif truth_pos and (not pred_pos):
            fn += 1
        else:
            tn += 1

    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "n_samples": float(total),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "unknown_predictions": float(unknown),
    }


def evaluate_split_metrics(
    model: Any,
    processor: Any,
    dataset: Dataset,
    collator: Any,
    device: torch.device,
    batch_size: int,
    split_name: str,
) -> dict[str, Any]:
    was_training = bool(getattr(model, "training", False))
    tokenizer = getattr(processor, "tokenizer", None)
    original_padding_side = (
        getattr(tokenizer, "padding_side", None) if tokenizer is not None else None
    )
    if tokenizer is not None:
        # Decoder-only generation should use left padding for batched inference.
        tokenizer.padding_side = "right"

    model.eval()
    y_true: list[str] = []
    y_pred: list[str] = []
    hal_types: list[str] = []
    pad_token_id = get_pad_token_id(processor)

    for start in tqdm(
        range(0, len(dataset), batch_size),
        desc=f"Evaluating {split_name}",
        unit="batch",
    ):
        end = min(start + batch_size, len(dataset))
        batch_features = [dataset[i] for i in range(start, end)]

        user_messages: list[list[dict[str, Any]]] = []
        for feature in batch_features:
            claim = str(feature["claim"]).strip()
            image_path = str(feature["img_path"])
            prompt = MODEL_PROMPT.format(claim=claim)
            user_messages.append(build_message(image_path, prompt))

        model_inputs = processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        model_inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in model_inputs.items()
        }

        input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=20,
                pad_token_id=pad_token_id,
            )

        for idx, feature in enumerate(batch_features):
            start_idx = int(input_lengths[idx])
            pred_ids = generated_ids[idx, start_idx:].detach().cpu().tolist()
            pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
            pred_label = parse_predicted_label(pred_text)

            true_label = normalize_label(feature.get("label_text")) or "UNKNOWN"
            y_true.append(true_label)
            y_pred.append(pred_label)
            hal_types.append(str(feature.get("hal_type", "unknown")))

    overall = compute_binary_metrics(y_true, y_pred)
    per_type: dict[str, dict[str, float]] = {}
    for hal_type in sorted(set(hal_types)):
        indices = [i for i, value in enumerate(hal_types) if value == hal_type]
        type_true = [y_true[i] for i in indices]
        type_pred = [y_pred[i] for i in indices]
        per_type[hal_type] = compute_binary_metrics(type_true, type_pred)

    results = {
        "overall": overall,
        "per_hallucination_type": per_type,
    }
    if tokenizer is not None and original_padding_side is not None:
        tokenizer.padding_side = original_padding_side
    if was_training:
        model.train()
    return results


def make_compute_metrics(processor, eval_dataset):
    tokenizer = processor.tokenizer
    hal_types = [str(sample["hal_type"]) for sample in eval_dataset]

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        y_pred = []
        y_true = []

        for pred_seq, label_seq in zip(predictions, labels):
            answer_positions = [i for i, l in enumerate(label_seq) if l != -100]
            if not answer_positions:
                y_pred.append("UNKNOWN")
                y_true.append("UNKNOWN")
                continue

            # Decode ALL answer tokens, not just the first one
            pred_token_ids = [int(pred_seq[i - 1]) for i in answer_positions if i > 0]
            true_token_ids = [int(label_seq[i]) for i in answer_positions]

            pred_text = tokenizer.decode(pred_token_ids, skip_special_tokens=True)
            true_text = tokenizer.decode(true_token_ids, skip_special_tokens=True)

            y_pred.append(parse_predicted_label(pred_text))  # your existing function
            y_true.append(
                parse_predicted_label(true_text)
            )  # already normalized in dataset

        if not y_true:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}

        overall = compute_binary_metrics(y_true, y_pred)

        per_type: dict[str, dict[str, float]] = {}
        for hal_type in sorted(set(hal_types)):
            indices = [i for i, h in enumerate(hal_types) if h == hal_type]
            indices = [i for i in indices if i < len(y_true)]
            if not indices:
                continue
            per_type[hal_type] = compute_binary_metrics(
                [y_true[i] for i in indices],
                [y_pred[i] for i in indices],
            )

        result = {f"overall_{k}": v for k, v in overall.items()}
        for hal_type, type_metrics in per_type.items():
            for k, v in type_metrics.items():
                result[f"{hal_type}_{k}"] = v

        return result

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    # Argmax on GPU before tensors are moved to CPU — avoids OOM from huge vocab logits
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


class EvalMetricsCallback(TrainerCallback):
    def __init__(
        self,
        metrics_dir: Path,
        file_prefix: str,
        history_filename: str,
    ) -> None:
        self.metrics_dir = metrics_dir
        self.file_prefix = file_prefix
        self.history_path = self.metrics_dir / history_filename
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        step = int(state.global_step)

        payload = {
            "step": step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "metrics": metrics or {},  # already contains eval_f1, eval_precision, etc.
        }

        step_path = self.metrics_dir / f"{self.file_prefix}_{step}.json"
        step_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self.history.append(payload)
        self.history_path.write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )

        print(f"Saved step metrics to {step_path.as_posix()}")
        return control


def plot_loss_curves(log_history: list[dict[str, Any]], output_path: str) -> None:
    train_steps: list[float] = []
    train_losses: list[float] = []
    eval_steps: list[float] = []
    eval_losses: list[float] = []

    for row in log_history:
        if "loss" in row:
            train_steps.append(float(row.get("step", len(train_steps) + 1)))
            train_losses.append(float(row["loss"]))
        if "eval_loss" in row:
            eval_steps.append(float(row.get("step", len(eval_steps) + 1)))
            eval_losses.append(float(row["eval_loss"]))

    if not train_steps and not eval_steps:
        print("No training/eval loss history found, skipping plot")
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_losses, label="train_loss")
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="val_loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output.as_posix(), dpi=160)
    plt.close()
    print(f"Saved loss plot to {output.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisualDebugger Task-2 SFT Training Script",
    )
    parser.add_argument(
        "--images-dir",
        default="data/VG/VG_500",
        help="Directory containing downloaded VG images. (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to JSON dataset",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model to use. (default: %(default)s).",
    )
    parser.add_argument(
        "--model-output",
        default="models",
        help="Base directory for model run outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Optional run/model name. If omitted, auto-generates sft_model_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device for training (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Batch size per device for evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: %(default)s).",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--train-val-test-split",
        nargs=3,
        type=float,
        default=[0.70, 0.15, 0.15],
        help="Train/Validation/Test split ratios. Input a space-separated list of three floats (default: %(default)s).",
    )
    parser.add_argument(
        "--base-max-new-tokens",
        type=int,
        default=220,
        help="Max new tokens for base model generation (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: %(default)s).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for training (default: %(default)s).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum tokenized sequence length for prompt+label (default: %(default)s).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log training metrics every N steps (default: %(default)s).",
    )
    parser.add_argument(
        "--save-strategy",
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="Checkpoint save strategy (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-strategy",
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="Evaluation strategy during training (default: %(default)s).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save interval in steps when --save-strategy=steps (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Eval interval in steps when --eval-strategy=steps (default: %(default)s).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=600,
        help="Warmup steps for scheduler (default: %(default)s).",
    )
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA adapter training for efficiency (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank r (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA injection (default: %(default)s).",
    )
    parser.add_argument(
        "--metrics-batch-size",
        type=int,
        default=4,
        help="Batch size for post-training metrics evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--metrics-dir",
        default="metrics",
        help="Base directory for metrics outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--metrics-output-json",
        default="sft_metrics_final.json",
        help="Final metrics summary filename under metrics/<model_name>/ (default: %(default)s).",
    )
    parser.add_argument(
        "--loss-plot-path",
        default="loss_curve.png",
        help="Loss plot filename under metrics/<model_name>/ (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-metrics-prefix",
        default="eval_metrics_step",
        help="Prefix for per-eval-step metrics files (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-metrics-history-file",
        default="eval_metrics_history.json",
        help="Filename for accumulated eval metrics history under metrics/<model_name>/ (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-post-metrics",
        action="store_true",
        help="Skip post-training precision/recall/F1 computation.",
    )
    parser.add_argument(
        "--skip-loss-plot",
        action="store_true",
        help="Skip plotting train/validation loss after training.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SFTConfig, SFTTrainer = import_trl_sft()

    model_name = (
        args.model_name.strip()
        or f"sft_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    model_output_dir = Path(args.model_output) / model_name
    metrics_dir = Path(args.metrics_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    loss_plot_path = metrics_dir / args.loss_plot_path
    final_metrics_path = metrics_dir / args.metrics_output_json

    print(f"Model name: {model_name}")
    print(f"Model output dir: {model_output_dir.as_posix()}")
    print(f"Metrics dir: {metrics_dir.as_posix()}")

    dataset_dict = load_dataset("json", data_files=args.dataset_path, num_proc=4)
    dataset = dataset_dict["train"]
    dataset = explode_claim_level_rows(dataset, args.images_dir)

    split_dataset = split_full_dataset(
        dataset=dataset,
        split_ratios=list(args.train_val_test_split),
        seed=args.train_seed,
    )

    print("\nDataset split summary:")
    for split_name, split_data in split_dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")

    device = pick_device()
    model, processor = load_vlm(args.model, device)
    collator = build_collator(processor, args.max_seq_length)

    training_args = SFTConfig(
        output_dir=model_output_dir.as_posix(),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="none",
        seed=args.train_seed,
        data_seed=args.train_seed,
        packing=False,
    )

    peft_config = None
    if args.use_lora:
        LoraConfig = import_lora_config()
        target_modules = [
            module.strip()
            for module in args.lora_target_modules.split(",")
            if module.strip()
        ]
        if not target_modules:
            raise ValueError(
                "--lora-target-modules cannot be empty when LoRA is enabled"
            )

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        print(
            "LoRA enabled: "
            f"r={args.lora_r}, alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}, targets={target_modules}"
        )
    else:
        print("LoRA disabled: full-model fine-tuning")

    trainer = SFTTrainer(
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["validation"],
        args=training_args,
        data_collator=collator,
        peft_config=peft_config,
        compute_metrics=make_compute_metrics(processor, split_dataset["validation"]),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.add_callback(
        EvalMetricsCallback(
            metrics_dir=metrics_dir,
            file_prefix=args.eval_metrics_prefix,
            history_filename=args.eval_metrics_history_file,
        )
    )

    if args.use_lora and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    trainer.train()
    trainer.save_model(model_output_dir.as_posix())
    processor.save_pretrained(model_output_dir.as_posix())

    if not args.skip_loss_plot:
        plot_loss_curves(trainer.state.log_history, loss_plot_path.as_posix())

    if not args.skip_post_metrics:
        final_validation_metrics = {
            "model_name": model_name,
            "final_step": int(trainer.state.global_step),
            "validation": evaluate_split_metrics(
                model=model,
                processor=processor,
                dataset=split_dataset["validation"],
                collator=collator,
                device=device,
                batch_size=args.metrics_batch_size,
                split_name="validation_final",
            ),
        }
        final_metrics_path.write_text(
            json.dumps(final_validation_metrics, indent=2), encoding="utf-8"
        )
        print(f"Saved final validation metrics to {final_metrics_path.as_posix()}")

    # test_metrics = trainer.evaluate(split_dataset["test"])
    # print("\nTest metrics:")
    # for key, value in test_metrics.items():
    #     print(f"  {key}: {value}")

    # if not args.skip_post_metrics:
    #     metrics_summary = {
    #         "validation": evaluate_split_metrics(
    #             model=model,
    #             processor=processor,
    #             dataset=split_dataset["validation"],
    #             collator=collator,
    #             device=device,
    #             batch_size=args.metrics_batch_size,
    #             split_name="validation",
    #         ),
    #         "test": evaluate_split_metrics(
    #             model=model,
    #             processor=processor,
    #             dataset=split_dataset["test"],
    #             collator=collator,
    #             device=device,
    #             batch_size=args.metrics_batch_size,
    #             split_name="test",
    #         ),
    #     }
    #     metrics_path = Path(args.metrics_output_json)
    #     metrics_path.parent.mkdir(parents=True, exist_ok=True)
    #     metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    #     print(f"Saved metrics summary to {metrics_path.as_posix()}")

    unload_vlm(model, processor, device)
