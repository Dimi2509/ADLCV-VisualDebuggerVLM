"""
Dataset adaptation layer for GRPO training.

The pipeline is:
    raw JSON → explode_claim_level_rows (from SFTTrain)
             → to_grpo_format         (this module)
             → GRPOTrainer-compatible dataset

The GRPOTrainer expects each row to have a `prompt` column in conversational
format (a list of message dicts containing both image and text). Other columns
(`label_text`, `hal_type`) are kept around so the reward function can read
them via **kwargs at training time.

This module is intentionally model-free — no torch, no transformers, no
processor. That keeps it fast to test and easy to reason about.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from PIL import Image

from SFTTrain import MODEL_PROMPT


def _build_prompt_messages(image: Any, claim: str) -> list[dict[str, Any]]:
    """
    Construct the conversational prompt for one (image, claim) pair.

    Mirrors SFTTrain.build_message but with two differences:
      - We omit the assistant turn (GRPO will generate that itself).
      - We pass either a PIL.Image or an image path; the trainer's
        processing_class handles both.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": MODEL_PROMPT.format(claim=claim.strip())},
            ],
        }
    ]


def to_grpo_format(
    dataset: Dataset,
    *,
    load_images: bool = True,
) -> Dataset:
    """
    Adapt an exploded claim-level dataset to the format GRPOTrainer expects.

    Input columns (from explode_claim_level_rows):
        image_id (int), img_path (str), claim (str),
        label_text ("CORRECT"|"HALLUCINATED"), hal_type (str)

    Output columns:
        prompt (list[dict]) — conversational prompt with image+text
        label_text (str)    — kept verbatim, used by the reward function
        hal_type (str)      — kept verbatim, used for per-class metrics

    Args:
        dataset: The exploded claim-level dataset.
        load_images: If True, eagerly load each img_path into a PIL.Image and
            embed it in the prompt. If False, embed the path string and let
            the trainer load on demand. PIL is the safer default for trl
            1.x; switch to False only if you hit memory pressure.
    """
    required = {"img_path", "claim", "label_text", "hal_type"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"to_grpo_format expected columns {required}; missing {missing}"
        )

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        if load_images:
            image = Image.open(example["img_path"]).convert("RGB")
        else:
            image = example["img_path"]

        return {
            "prompt": _build_prompt_messages(image, example["claim"]),
            "label_text": example["label_text"],
            "hal_type": example["hal_type"],
        }

    # remove_columns drops everything else so the dataset is exactly what
    # GRPOTrainer's signature columns expect (prompt + reward kwargs).
    columns_to_drop = [c for c in dataset.column_names if c not in {}]
    return dataset.map(
        _convert,
        remove_columns=columns_to_drop,
        desc="Adapting dataset to GRPO format",
    )