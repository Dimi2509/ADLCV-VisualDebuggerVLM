"""
VisualDebugger Task-2 GRPO Training Script.

Continues from an SFT-trained checkpoint and applies GRPO to optimise the
verifier directly against a reward signal (vs. SFT's token-level cross-
entropy). The dataset, prompt template, and parser are shared with SFT to
keep train/eval comparable.
"""

from __future__ import annotations

import argparse
import json
import locale
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

# We re-use SFT's data plumbing wholesale. If those functions move or change,
# both SFT and GRPO should still agree.
from SFTTrain import (
    pick_device,
    pick_dtype,
    explode_claim_level_rows,
    split_full_dataset,
)
from grpo_dataset import to_grpo_format
from grpo_rewards import make_reward_func


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisualDebugger Task-2 GRPO Training Script",
    )

    # --- Data ---
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to JSON dataset (same format as SFT).",
    )
    parser.add_argument(
        "--images-dir",
        default="data/VG/VG_500",
        help="Directory containing VG images (default: %(default)s).",
    )
    parser.add_argument(
        "--train-val-test-split",
        nargs=3,
        type=float,
        default=[0.70, 0.15, 0.15],
        help=(
            "MUST match the values used at SFT training time, otherwise the "
            "splits will leak and metrics will be invalid (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="MUST match SFT training seed (default: %(default)s).",
    )

    # --- Model ---
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base VLM identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--sft-adapter-path",
        required=True,
        help=(
            "Path to the SFT-trained LoRA adapter directory. GRPO uses this as "
            "its starting point — the project compares (a) baseline, (b) SFT, "
            "(c) SFT+GRPO."
        ),
    )

    # --- LoRA ---
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: %(default)s).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: %(default)s).",
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
        help="Comma-separated module names for LoRA (default: %(default)s).",
    )

    # --- GRPO algorithm ---
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="K — rollouts per prompt. Project spec recommends 4 (default: %(default)s).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL coefficient β. Smaller = more exploration (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Rollout sampling temperature. Must be > 0 for GRPO (default: %(default)s).",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=8,
        help="Max new tokens per rollout. SFT uses 8 — keep consistent (default: %(default)s).",
    )
    parser.add_argument(
        "--reward-weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use asymmetric class-balanced rewards vs. plain ±1 (default: %(default)s).",
    )

    # --- Optimiser / training ---
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="GRPO learning rate. Typically much smaller than SFT (default: %(default)s).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="GRPO usually needs 1 epoch (default: %(default)s).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Forward batch size per device (default: %(default)s).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: %(default)s).",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1024,
        help="Truncate prompt tokens beyond this (default: %(default)s).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Log every N steps (default: %(default)s).",
    )
    parser.add_argument(
        "--save-strategy",
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="Checkpoint save strategy (default: %(default)s).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save interval when --save-strategy=steps (default: %(default)s).",
    )

    # --- Outputs ---
    parser.add_argument(
        "--model-output",
        default="models",
        help="Base directory for run outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Run name. Defaults to grpo_model_<timestamp>.",
    )
    parser.add_argument(
        "--metrics-dir",
        default="metrics",
        help="Base directory for metrics outputs (default: %(default)s).",
    )

    # --- Smoke test escape hatch ---
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="If > 0, cap training set size. Use small values (50-200) for smoke tests (default: %(default)s).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading: base + SFT adapter
# ---------------------------------------------------------------------------
def load_base_model_with_sft_adapter(
    base_model_name: str,
    sft_adapter_path: str,
    device: torch.device,
) -> tuple[Any, Any]:
    """
    Load Qwen3-VL-2B and apply the SFT-trained LoRA adapter on top.

    GRPO will continue training from this state. We DON'T merge the adapter
    into the base weights — we leave it as a PEFT model so that:
      - GRPO can keep training the same LoRA parameters, OR
      - GRPO can stack a new LoRA on top (configurable later).
    """
    from peft import PeftModel

    dtype = pick_dtype(device)
    print(f"Loading base model: {base_model_name} (dtype={dtype})")

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_name,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    print(f"Applying SFT adapter from: {sft_adapter_path}")
    model = PeftModel.from_pretrained(base, sft_adapter_path, is_trainable=True)

    # Try to load the processor saved alongside the adapter — falls back to
    # the base model's processor if the SFT run didn't save one.
    try:
        processor = AutoProcessor.from_pretrained(sft_adapter_path, trust_remote_code=True)
        print(f"Loaded processor from adapter directory")
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        print(f"Adapter dir had no processor, fell back to base model's")

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        # GRPO does generation, which needs left padding for batched rollouts.
        processor.tokenizer.padding_side = "left"

    model = model.to(device)
    model.train()
    return model, processor


def import_lora_config() -> Any:
    """Mirror SFTTrain.import_lora_config so GRPO doesn't depend on it."""
    try:
        from peft import LoraConfig
        return LoraConfig
    except ImportError as exc:
        raise ImportError("peft is required for LoRA. Install with: uv add peft") from exc


def build_lora_config(args: argparse.Namespace) -> Any:
    """
    Build a fresh LoRA config for GRPO.

    Note: passing this to GRPOTrainer when the model already has a PEFT
    adapter from SFT is delicate. trl 1.x supports two paths:
      (a) reuse the existing adapter — pass peft_config=None
      (b) stack a new adapter on top — pass peft_config=this
    For our project (a) is what we want: continue training the SFT adapter.
    We build the config anyway in case we want to switch.
    """
    LoraConfig = import_lora_config()
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


# ---------------------------------------------------------------------------
# Placeholder main — we'll fill this in subsequent sub-sections
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("GRPO TRAINING — sanity prints")
    print("=" * 70)
    print(f"Dataset:      {args.dataset_path}")
    print(f"Images dir:   {args.images_dir}")
    print(f"Base model:   {args.base_model}")
    print(f"SFT adapter:  {args.sft_adapter_path}")
    print(f"K:            {args.num_generations}")
    print(f"β (KL):       {args.beta}")
    print(f"Temperature:  {args.temperature}")
    print(f"LR:           {args.lr}")
    print(f"Reward mode:  {'weighted' if args.reward_weighted else 'unweighted ±1'}")
    print()

    device = pick_device()
    print(f"Device: {device}")

    # Quick smoke test: can we load the SFT-adapted model?
    model, processor = load_base_model_with_sft_adapter(
        args.base_model, args.sft_adapter_path, device
    )

    # Print trainable parameter count to confirm LoRA is engaged
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    print("\nModel + adapter loaded successfully. (Training loop coming in the next sub-section.)")


if __name__ == "__main__":
    main()