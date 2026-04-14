import argparse
import gc
import random
from typing import Any

import torch
import transformers
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, Value, load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_vlm(model_name: str, device: torch.device) -> tuple[Any, Any]:
    dtype = pick_dtype(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, processor


def unload_vlm(model: Any, processor: Any, device: torch.device) -> None:
    del model
    del processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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

    if n_test <= 0:
        raise ValueError("Split ratios produced an empty test split")

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
        default=None,
        help="Path to JSON dataset",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model to use. (default: %(default)s).",
    )
    parser.add_argument(
        "--model-output",
        default="models/",
        help="Final fine-tuned model output location (default: %(default)s).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="Batch size per device for training (default: %(default)s).",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_dict = load_dataset("json", data_files=args.dataset_path, num_proc=4)
    dataset = dataset_dict["train"]
    dataset = dataset.cast_column("image_id", Value("int32"))

    split_dataset = split_full_dataset(
        dataset=dataset,
        split_ratios=list(args.train_val_test_split),
        seed=args.train_seed,
    )

    print("\nDataset split summary:")
    for split_name, split_data in split_dataset.items():
        print(f"  {split_name}: {split_data.features} \n")
