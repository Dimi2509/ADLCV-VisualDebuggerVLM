"""
Dataset adaptation layer for GRPO training.

The pipeline is:
    raw JSON → explode_claim_level_rows (from SFTTrain)
             → to_grpo_format         (this module)
             → GRPOTrainer-compatible dataset

GRPOTrainer expects:
  - `prompt`: conversational messages (text only — image is referenced by
    structure but the actual PIL.Image lives in a separate top-level column).
  - `image`:  the PIL.Image as a top-level column. datasets stores this
    efficiently via its built-in Image feature type.
  - any other columns (label_text, hal_type) are forwarded to the reward
    function as kwargs.

Why split image out as a top-level column?
The arrow writer in `datasets` cannot serialise PIL.Image objects when they
are nested inside dict/list structures — it falls back to ujson, which
chokes on non-UTF-8 image bytes. Keeping the image at the top level lets
datasets use its Image() feature type, which encodes images correctly.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, Features, Image as ImageFeature, Sequence, Value
from PIL import Image

from SFTTrain import MODEL_PROMPT


def _build_prompt_messages(claim: str) -> list[dict[str, Any]]:
    """
    Build conversational messages for one claim.

    NOTE: the image is NOT embedded here. The {"type": "image"} entry has no
    "image" key — trl's chat-template path will fill it in from the dataset's
    top-level `image` column at training time.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # placeholder — image bound at runtime
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
        prompt (list[dict]) — conversational messages, image placeholder only
        image  (PIL.Image)  — top-level, datasets stores this efficiently
        label_text (str)    — used by reward function
        hal_type (str)      — used for per-class metrics

    Args:
        load_images: kept for API compatibility but currently has no effect —
            we always emit the path/Image into the top-level `image` column.
            datasets handles lazy decoding via its Image feature type.
    """
    required = {"img_path", "claim", "label_text", "hal_type"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"to_grpo_format expected columns {required}; missing {missing}"
        )

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": _build_prompt_messages(example["claim"]),
            "image": example["img_path"],  # datasets' Image feature accepts paths
            "label_text": example["label_text"],
            "hal_type": example["hal_type"],
        }

    columns_to_drop = list(dataset.column_names)

    # Declaring features explicitly tells datasets to use the Image() feature
    # type for the `image` column, which routes through the dedicated image
    # encoder instead of the broken ujson fallback.
    features = Features({
        "prompt": [
            {
                "role": Value("string"),
                "content": [
                    {
                        "type": Value("string"),
                        "text": Value("string"),
                    }
                ],
            }
        ],
        "image": ImageFeature(),
        "label_text": Value("string"),
        "hal_type": Value("string"),
    })

    return dataset.map(
        _convert,
        remove_columns=columns_to_drop,
        features=features,
        desc="Adapting dataset to GRPO format",
    )