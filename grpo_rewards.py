"""
Reward functions for GRPO training of the VisualDebugger verifier.

This module is intentionally model-free — it only manipulates strings and
floats. That makes it cheap to unit-test and easy to reason about. The
GRPOTrainer integration lives in GRPOTrain.py.
"""

from __future__ import annotations

from typing import Any


# Re-export the same parser used at SFT and eval time so train/eval are
# 100% consistent. If we changed the parser only here, baseline/SFT/GRPO
# evaluations would silently disagree.
from SFTTrain import parse_predicted_label

VALID_LABELS = {"CORRECT", "HALLUCINATED"}


# ---------------------------------------------------------------------------
# Core scalar reward
# ---------------------------------------------------------------------------
def compute_reward(
    predicted_label: str,
    true_label: str,
    *,
    weighted: bool = True,
) -> float:
    """
    Map a (predicted, true) pair to a scalar reward.

    Args:
        predicted_label: One of {"CORRECT", "HALLUCINATED", "UNKNOWN"} or any
            other string. Anything outside VALID_LABELS is treated as a
            format violation.
        true_label: One of {"CORRECT", "HALLUCINATED"} (the ground-truth label
            from generated_dataset.json).
        weighted: If True, use the asymmetric scheme that rewards catching
            hallucinations more than confirming truths and penalises misses
            more than false alarms. If False, use plain +1 / -1.

    Returns:
        A float reward.
    """
    if predicted_label not in VALID_LABELS:
        # Format violation. Strongly discouraged regardless of `weighted`.
        return -2.0

    correct = predicted_label == true_label

    if not weighted:
        return 1.0 if correct else -1.0

    if correct and true_label == "HALLUCINATED":
        return +1.5  # True positive — most valuable
    if correct and true_label == "CORRECT":
        return +1.0  # True negative
    if (not correct) and true_label == "CORRECT":
        return -1.0  # False positive — cried wolf
    if (not correct) and true_label == "HALLUCINATED":
        return -1.5  # False negative — missed a hallucination

    return -2.0  # Unreachable safety net


# ---------------------------------------------------------------------------
# TRL-compatible reward function
# ---------------------------------------------------------------------------
def _extract_text(completion: Any) -> str:
    """
    GRPOTrainer hands completions in two possible shapes:
      - Standard format: a plain string.
      - Conversational format: a list of message dicts, e.g.
        [{"role": "assistant", "content": "HALLUCINATED"}].
    We normalise both into a flat string before parsing.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            content = first.get("content", "")
            if isinstance(content, str):
                return content
            # Some processors give content as a list of {"type": ..., "text": ...}
            if isinstance(content, list):
                texts = [
                    seg.get("text", "")
                    for seg in content
                    if isinstance(seg, dict) and seg.get("type") == "text"
                ]
                return " ".join(texts)
    return str(completion)


def make_reward_func(weighted: bool = True):
    """
    Build a reward function with the signature GRPOTrainer expects.

    GRPOTrainer calls this as:
        reward_func(prompts=..., completions=..., **dataset_columns)
    where every non-`prompt` column in the train_dataset is forwarded as a
    kwarg whose value is a list aligned with `completions`. We only need
    `label_text` here.
    """

    def verifier_reward(completions, **kwargs) -> list[float]:
        labels = kwargs.get("label_text")
        if labels is None:
            raise KeyError(
                "verifier_reward expected a 'label_text' column in the dataset; "
                f"got kwargs keys = {list(kwargs.keys())}"
            )
        if len(labels) != len(completions):
            raise ValueError(
                f"length mismatch: {len(completions)} completions vs "
                f"{len(labels)} labels"
            )

        rewards: list[float] = []
        for completion, true_label in zip(completions, labels):
            text = _extract_text(completion)
            predicted = parse_predicted_label(text)
            rewards.append(
                compute_reward(predicted, str(true_label).upper(), weighted=weighted)
            )
        return rewards

    return verifier_reward