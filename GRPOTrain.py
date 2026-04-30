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
# Dataset pipeline
# ---------------------------------------------------------------------------
def prepare_grpo_datasets(args: argparse.Namespace) -> tuple:
    """
    Build train/val/test datasets in GRPOTrainer format, using the exact
    same split logic as SFT (so test set is the same set of images).
    """
    print(f"Loading dataset from {args.dataset_path}")
    raw = load_dataset("json", data_files=args.dataset_path, num_proc=1)["train"]

    print("Exploding to claim-level rows...")
    exploded = explode_claim_level_rows(raw, args.images_dir)
    print(f"  Total claim-level rows: {len(exploded)}")

    print("Splitting train/val/test (image-level)...")
    splits = split_full_dataset(
        dataset=exploded,
        split_ratios=list(args.train_val_test_split),
        seed=args.train_seed,
    )
    for name, ds in splits.items():
        print(f"  {name}: {len(ds)}")

    print("Adapting splits to GRPO format...")
    grpo_train = to_grpo_format(splits["train"], load_images=True)
    grpo_val = to_grpo_format(splits["validation"], load_images=True)

    # Optional cap for smoke testing — only on train. Eval splits stay full.
    if args.max_train_samples and args.max_train_samples > 0:
        cap = min(args.max_train_samples, len(grpo_train))
        grpo_train = grpo_train.select(range(cap))
        print(f"  [smoke test] Capped train set to {cap} samples")

    return grpo_train, grpo_val, splits  # raw splits returned for later eval


# ---------------------------------------------------------------------------
# GRPO config builder
# ---------------------------------------------------------------------------
def build_grpo_config(args: argparse.Namespace, model_output_dir: Path, device: torch.device):
    """
    Translate our argparse args into a trl GRPOConfig.

    A note on import: trl loads its config via `pyproject.toml` reads, which
    on some Windows setups picks up a non-UTF-8 default encoding. We patch
    the same way SFTTrain does — temporarily forcing UTF-8 — to match.
    """
    original_getpreferredencoding = locale.getpreferredencoding
    original_read_text = Path.read_text

    def _utf8_preferred_encoding(do_setlocale: bool = True) -> str:
        return "utf-8"

    def _read_text_utf8(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        return original_read_text(self, encoding=encoding or "utf-8", errors=errors)

    locale.getpreferredencoding = _utf8_preferred_encoding
    Path.read_text = _read_text_utf8
    try:
        from trl import GRPOConfig
    finally:
        locale.getpreferredencoding = original_getpreferredencoding
        Path.read_text = original_read_text

    config = GRPOConfig(
        output_dir=model_output_dir.as_posix(),
        use_cpu=(device.type == "cpu"),
        # --- GRPO core ---
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        #max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        beta=args.beta,
        # --- Optimisation ---
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # --- Logging / saving ---
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to="none",
        log_completions=True,  # writes a few example completions to logs — useful for debugging
        # --- Reproducibility ---
        seed=args.train_seed,
        data_seed=args.train_seed,
        # --- Misc ---
        remove_unused_columns=False,  # we need label_text/hal_type passed to reward_func
    )
    return config

# ---------------------------------------------------------------------------
# GRPOTrainer importer (same UTF-8 trick as GRPOConfig)
# ---------------------------------------------------------------------------
def import_grpo_trainer() -> Any:
    """Import trl.GRPOTrainer with a UTF-8 locale patch — same trick as
    build_grpo_config. trl's import touches pyproject.toml via Path.read_text,
    which on Windows defaults to cp1252 and chokes on non-ASCII bytes."""
    original_getpreferredencoding = locale.getpreferredencoding
    original_read_text = Path.read_text

    def _utf8_preferred_encoding(do_setlocale: bool = True) -> str:
        return "utf-8"

    def _read_text_utf8(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        return original_read_text(self, encoding=encoding or "utf-8", errors=errors)

    locale.getpreferredencoding = _utf8_preferred_encoding
    Path.read_text = _read_text_utf8
    try:
        from trl import GRPOTrainer
    finally:
        locale.getpreferredencoding = original_getpreferredencoding
        Path.read_text = original_read_text
    return GRPOTrainer

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

    # --- Output paths ---
    run_name = (
        args.model_name.strip() or f"grpo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    model_output_dir = Path(args.model_output) / run_name
    metrics_dir = Path(args.metrics_dir) / run_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name:        {run_name}")
    print(f"Model output:    {model_output_dir}")
    print(f"Metrics output:  {metrics_dir}")
    print()

    # --- Datasets ---
    grpo_train, grpo_val, raw_splits = prepare_grpo_datasets(args)
    print(f"\nGRPO train sample[0] keys: {list(grpo_train[0].keys())}")
    print(f"GRPO train sample[0] label_text: {grpo_train[0]['label_text']}")
    print(f"GRPO train sample[0] hal_type:   {grpo_train[0]['hal_type']}")
    prompt_text_preview = grpo_train[0]["prompt"][0]["content"][1]["text"][:120]
    print(f"GRPO train sample[0] prompt text (first 120 chars):\n  {prompt_text_preview}...")
    print(f"GRPO train sample[0] image type: {type(grpo_train[0]['image']).__name__}")
    print(f"GRPO train sample[0] image size: {grpo_train[0]['image'].size}")
    print()

    # --- Model ---
    device = pick_device()
    print(f"Device: {device}")
    model, processor = load_base_model_with_sft_adapter(
        args.base_model, args.sft_adapter_path, device
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    print()

    # --- Reward function ---
    reward_func = make_reward_func(weighted=args.reward_weighted)

    # Quick reward function smoke test on a tiny batch
    print("Reward function smoke test...")
    fake_completions = [
        [{"role": "assistant", "content": "CORRECT"}],
        [{"role": "assistant", "content": "HALLUCINATED"}],
    ]
    fake_labels = ["CORRECT", "HALLUCINATED"]
    fake_rewards = reward_func(fake_completions, label_text=fake_labels)
    print(f"  fake_completions -> rewards: {fake_rewards}")
    print()

    # --- GRPO config ---
    grpo_config = build_grpo_config(args, model_output_dir, device)
    print("GRPOConfig built successfully.")
    print(f"  num_generations:          {grpo_config.num_generations}")
    print(f"  beta (KL coefficient):    {grpo_config.beta}")
    print(f"  learning_rate:            {grpo_config.learning_rate}")
    print(f"  per_device_train_bsz:     {grpo_config.per_device_train_batch_size}")
    print(f"  gradient_accumulation:    {grpo_config.gradient_accumulation_steps}")
    print(f"  output_dir:               {grpo_config.output_dir}")
    print()

    #print("=" * 70)
    #print("All preflight checks passed.")
    #print("(Trainer instantiation + train() coming in next sub-section.)")
    #print("=" * 70)
    # ========================================================================
    # Trainer instantiation + train() + save
    # ========================================================================
    print("=" * 70)
    print("Instantiating GRPOTrainer...")
    print("=" * 70)
    GRPOTrainer = import_grpo_trainer()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=grpo_train,
        # NOTE: we deliberately don't pass eval_dataset. GRPO eval at training
        # time is non-trivial (needs rollouts on the eval set) and would slow
        # the smoke test to a crawl. We evaluate offline using EvaluateModels.py
        # after training finishes — same protocol used for SFT, so the three
        # configurations (baseline / SFT / SFT+GRPO) stay directly comparable.
        processing_class=processor,
        # peft_config intentionally omitted (=None). The model already has the
        # SFT LoRA adapter from load_base_model_with_sft_adapter, and trl will
        # continue training that same adapter. Passing a fresh peft_config would
        # stack a new adapter on top, freezing the SFT one — that's not what
        # we want.
    )

    # Rough math so you know what to expect before kicking off train().
    effective_batch = (
        grpo_config.per_device_train_batch_size
        * grpo_config.gradient_accumulation_steps
    )
    estimated_steps = max(1, len(grpo_train) // effective_batch)
    print(f"Trainer ready.")
    print(f"  Train samples:           {len(grpo_train)}")
    print(f"  Effective batch:         {effective_batch} (= {grpo_config.per_device_train_batch_size} x {grpo_config.gradient_accumulation_steps})")
    print(f"  Estimated optimizer steps: ~{estimated_steps} per epoch")
    print(f"  Generations per step:    {effective_batch * grpo_config.num_generations}")
    print()

    # ========================================================================
    # Train
    # ========================================================================
    print("=" * 70)
    print("Starting GRPO training...")
    print("=" * 70)
    train_result = trainer.train()
    print(f"\nTraining complete.")
    print(f"  Final training loss: {train_result.training_loss:.4f}")
    print(f"  Total steps:         {train_result.global_step}")
    print()

    # ========================================================================
    # Save adapter + processor + log history
    # ========================================================================
    print("=" * 70)
    print(f"Saving artifacts to {model_output_dir}")
    print("=" * 70)

    # save_model on a PEFT-wrapped model saves only the LoRA adapter weights
    # (a few MB), not the whole 2B-param base model. Same convention as SFT.
    trainer.save_model(model_output_dir.as_posix())
    processor.save_pretrained(model_output_dir.as_posix())
    print(f"  Saved adapter + processor")

    # The full log history (per-step loss, reward stats, KL divergence,
    # completion length, etc.) is invaluable for the report — save it now
    # so we can plot reward-over-time later.
    log_history = trainer.state.log_history
    history_path = metrics_dir / "grpo_log_history.json"
    history_path.write_text(
        json.dumps(log_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Saved training log history to {history_path}")
    print()

    print("=" * 70)
    print("GRPO training run complete.")
    print(f"  Adapter: {model_output_dir}")
    print(f"  Metrics: {metrics_dir}")
    print()
    print("Next: evaluate with EvaluateModels.py --model-type sft-grpo")
    print(f"      --adapter-path {model_output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()