#!/usr/bin/env python3
"""Task 3 pipeline for VisualDebugger: generate -> verify -> correct.

This script is intentionally verifier-ready:
- Today it can run zero-shot verification with the same VLM.
- When the Task 2 verifier is ready, pass its checkpoint with --verifier-model-name
  and optionally --verifier-adapter-path.
- It writes JSONL records so evaluation code can be added without changing the
  pipeline contract.

Example:
    python task3_pipeline.py \
        --image data/benchmark/coco_subset/COCO_val2014_000000310196.jpg \
        --model-name Qwen/Qwen3-VL-2B-Instruct

No-GPU smoke test:
    python task3_pipeline.py \
        --mock-backend \
        --pope-file data/benchmark/pope/coco_pope_popular.json \
        --image-root data/benchmark/coco_subset \
        --max-samples 1 \
        --num-candidates 2 \
        --skip-missing-images \
        --output outputs/task3_mock_smoke.jsonl

Dataset run:
    python task3_pipeline.py \
        --pope-file data/benchmark/pope/coco_pope_popular.json \
        --image-root data/benchmark/coco_subset \
        --max-samples 20 \
        --num-candidates 5 \
        --skip-missing-images \
        --output outputs/task3_popular.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is a convenience only.
    tqdm = None  # type: ignore[assignment]


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT = "Describe the image in one concise paragraph. Only mention visible evidence."
CLAIM_LABEL_PROMPT = (
    "You are an expert assistant for determining whether a claim is correct or "
    "incorrect based on the provided image and claim.\n"
    "Your answer must be exactly one token: CORRECT or HALLUCINATED.\n"
    "Claim: {claim}\n"
)
LOCAL_SAMPLE_IMAGE = "data/benchmark/coco_subset/COCO_val2014_000000310196.jpg"


@dataclass
class VerificationClaim:
    id: int
    claim: str
    label: str
    reason: str = ""


@dataclass
class Task3Record:
    image: str
    prompt: str
    single_pass_response: str
    verification: list[VerificationClaim]
    corrected_response: str
    best_of_n_response: str | None = None
    best_of_n_verification: list[VerificationClaim] | None = None
    meta: dict[str, Any] | None = None


def pick_model_dtype(device_type: str) -> torch.dtype:
    """Match the existing benchmark code: fp16 on CUDA, fp32 elsewhere."""
    import torch

    return torch.float16 if device_type == "cuda" else torch.float32


def build_message(image_path: str, text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
    ]


def load_vlm(
    model_name: str,
    device,
    dtype,
    adapter_path: str | None = None,
):
    """Load a base VLM and, optionally, a PEFT/LoRA adapter."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "--verifier-adapter-path was set, but peft is not installed. "
                "Install it with `uv add peft` or merge the adapter into a model directory."
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_path).to(device)

    model.eval()
    return model, processor


def generate_text(
    model,
    processor,
    device,
    image_path: str,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
) -> str:
    if isinstance(model, MockVLM):
        return model.generate(image_path, prompt, do_sample=do_sample)

    import torch

    messages = build_message(image_path, prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_len = len(inputs.input_ids[0])
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    text = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
    )[0]
    del inputs, generated_ids
    return text.strip()


def split_response_into_claims(response: str) -> list[str]:
    """Lightweight fallback claim splitter for the Task 2 claim-label verifier."""
    cleaned = re.sub(r"\s+", " ", response).strip()
    if not cleaned:
        return []

    claims: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        sentence = sentence.strip(" -*\t\n")
        if not sentence:
            continue
        # Keep this deliberately conservative: the trained verifier can judge
        # sentence-sized claims, and over-splitting captions creates fragments.
        for part in re.split(r"\s*;\s*", sentence):
            part = part.strip(" ,")
            if part:
                claims.append(part)
    return claims


def verification_prompt(response: str) -> str:
    return f"""You are the VisualDebugger verifier.

Look at the image again and check the model response claim by claim.
Return ONLY valid JSON with this exact schema:
{{
  "claims": [
    {{
      "id": 1,
      "claim": "atomic factual claim from the response",
      "label": "CORRECT or HALLUCINATED",
      "reason": "brief visual evidence"
    }}
  ]
}}

Rules:
- Split compound sentences into atomic visual claims.
- Mark a claim HALLUCINATED if it is not supported by the image.
- Do not add claims that are not in the response.
- Keep labels exactly CORRECT or HALLUCINATED.

Response to verify:
{response}
"""


def claim_verification_prompt(claim: str) -> str:
    return CLAIM_LABEL_PROMPT.format(claim=claim.strip())


def correction_prompt(response: str, verification: list[VerificationClaim]) -> str:
    flagged = [claim for claim in verification if is_hallucinated(claim)]
    feedback = "\n".join(
        f"- Claim {claim.id}: {claim.claim} | reason: {claim.reason}"
        for claim in flagged
    )
    if not feedback:
        feedback = "- No hallucinated claims were flagged."

    return f"""You are the VisualDebugger corrector.

Rewrite the original response using the image and the verifier feedback.
Preserve claims that are supported by the image.
Remove or replace only the flagged hallucinated claims.
Do not invent new details. Do not mention the verifier.

Original response:
{response}

Flagged hallucinated claims:
{feedback}

Corrected response:
"""


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse JSON even when the model wraps it in ```json fences or prose."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        text = fence_match.group(1)

    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def normalize_verifier_label(text: str) -> str:
    normalized = text.strip().upper()
    if "HALLUCINATED" in normalized or "HALLUCINATION" in normalized:
        return "HALLUCINATED"
    if "CORRECT" in normalized:
        return "CORRECT"
    return "HALLUCINATED"


def parse_verification(text: str) -> list[VerificationClaim]:
    parsed = extract_json_object(text)
    if not parsed:
        return [
            VerificationClaim(
                id=1,
                claim="UNPARSEABLE_VERIFICATION_OUTPUT",
                label="HALLUCINATED",
                reason=f"Verifier did not return JSON: {text[:500]}",
            )
        ]

    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        claims = []

    out: list[VerificationClaim] = []
    for i, item in enumerate(claims, start=1):
        if not isinstance(item, dict):
            continue
        label = normalize_verifier_label(str(item.get("label", "")))
        raw_id = item.get("id", i) or i
        try:
            claim_id = int(raw_id)
        except (TypeError, ValueError):
            claim_id = i
        out.append(
            VerificationClaim(
                id=claim_id,
                claim=str(item.get("claim", "")).strip(),
                label=label,
                reason=str(item.get("reason", "")).strip(),
            )
        )

    return out


def is_hallucinated(claim: VerificationClaim) -> bool:
    return claim.label.strip().upper() == "HALLUCINATED"


def verify_response(
    verifier_model,
    verifier_processor,
    device,
    image_path: str,
    response: str,
    *,
    verifier_mode: str = "json",
    max_new_tokens: int = 512,
) -> list[VerificationClaim]:
    if verifier_mode == "claim":
        claims = split_response_into_claims(response)
        if not claims:
            return [
                VerificationClaim(
                    id=1,
                    claim="EMPTY_RESPONSE",
                    label="HALLUCINATED",
                    reason="Generator returned no verifiable claims.",
                )
            ]

        verification: list[VerificationClaim] = []
        for i, claim in enumerate(claims, start=1):
            raw = generate_text(
                verifier_model,
                verifier_processor,
                device,
                image_path,
                claim_verification_prompt(claim),
                max_new_tokens=min(max_new_tokens, 16),
                do_sample=False,
            )
            verification.append(
                VerificationClaim(
                    id=i,
                    claim=claim,
                    label=normalize_verifier_label(raw),
                    reason=f"claim-verifier output: {raw[:120]}",
                )
            )
        return verification

    raw = generate_text(
        verifier_model,
        verifier_processor,
        device,
        image_path,
        verification_prompt(response),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return parse_verification(raw)


def correct_response(
    generator_model,
    generator_processor,
    device,
    image_path: str,
    response: str,
    verification: list[VerificationClaim],
    *,
    max_new_tokens: int = 256,
) -> str:
    if not any(is_hallucinated(claim) for claim in verification):
        return response

    return generate_text(
        generator_model,
        generator_processor,
        device,
        image_path,
        correction_prompt(response, verification),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def run_loop(
    generator_model,
    generator_processor,
    verifier_model,
    verifier_processor,
    device,
    image_path: str,
    prompt: str,
    *,
    verifier_mode: str,
    max_new_tokens: int,
    verifier_max_new_tokens: int,
) -> tuple[str, list[VerificationClaim], str]:
    response = generate_text(
        generator_model,
        generator_processor,
        device,
        image_path,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    verification = verify_response(
        verifier_model,
        verifier_processor,
        device,
        image_path,
        response,
        verifier_mode=verifier_mode,
        max_new_tokens=verifier_max_new_tokens,
    )
    corrected = correct_response(
        generator_model,
        generator_processor,
        device,
        image_path,
        response,
        verification,
        max_new_tokens=max_new_tokens,
    )
    return response, verification, corrected


def run_best_of_n(
    generator_model,
    generator_processor,
    verifier_model,
    verifier_processor,
    device,
    image_path: str,
    prompt: str,
    *,
    n: int,
    verifier_mode: str,
    max_new_tokens: int,
    verifier_max_new_tokens: int,
) -> tuple[str, list[VerificationClaim]]:
    """Generate N sampled answers, verify them, and choose the least flagged one."""
    candidates: list[tuple[str, list[VerificationClaim]]] = []
    for _ in range(n):
        response = generate_text(
            generator_model,
            generator_processor,
            device,
            image_path,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        verification = verify_response(
            verifier_model,
            verifier_processor,
            device,
            image_path,
            response,
            verifier_mode=verifier_mode,
            max_new_tokens=verifier_max_new_tokens,
        )
        candidates.append((response, verification))

    def score(item: tuple[str, list[VerificationClaim]]) -> tuple[int, int]:
        response, verification = item
        hallucinated = sum(is_hallucinated(claim) for claim in verification)
        return hallucinated, len(response)

    return min(candidates, key=score)


class MockVLM:
    """Deterministic stand-in used by smoke tests before GPU/verifier access."""

    def __init__(self) -> None:
        self.sample_idx = 0

    def generate(self, image_path: str, prompt: str, *, do_sample: bool = False) -> str:
        if "Response to verify:" in prompt:
            response = prompt.split("Response to verify:", 1)[1].strip()
            hallucinated = "red car" in response.lower() or "dining table" in response.lower()
            label = "HALLUCINATED" if hallucinated else "CORRECT"
            reason = (
                "The mock verifier treats this as unsupported visual evidence."
                if hallucinated
                else "The mock verifier treats this as supported visual evidence."
            )
            return json.dumps(
                {
                    "claims": [
                        {
                            "id": 1,
                            "claim": response,
                            "label": label,
                            "reason": reason,
                        }
                    ]
                }
            )

        if "Flagged hallucinated claims:" in prompt:
            return "A person is skiing on snow."

        if "Your answer must be exactly one token: CORRECT or HALLUCINATED." in prompt:
            claim = prompt.rsplit("Claim:", 1)[-1].strip().lower()
            if "red car" in claim or "dining table" in claim:
                return "HALLUCINATED"
            return "CORRECT"

        if do_sample:
            self.sample_idx += 1
            if self.sample_idx == 1:
                return "A person is skiing on snow near a red car."
            return "A person is skiing on snow."

        return "A person is skiing on snow near a red car."


def load_pope_rows(pope_file: str, image_root: str, max_samples: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(pope_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            image_name = raw["image"]
            rows.append(
                {
                    "image": os.path.join(image_root, image_name),
                    "prompt": DEFAULT_PROMPT,
                    "pope_question": raw.get("text", ""),
                    "pope_label": raw.get("label", ""),
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def iter_inputs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.image:
        return [{"image": args.image, "prompt": args.prompt}]
    if args.pope_file:
        return load_pope_rows(args.pope_file, args.image_root, args.max_samples)
    raise ValueError("Provide either --image or --pope-file.")


def record_to_json(record: Task3Record) -> str:
    payload = asdict(record)
    return json.dumps(payload, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--verifier-model-name",
        default=None,
        help=(
            "Task 2 verifier checkpoint/model. Defaults to --model-name for "
            "zero-shot self-verification."
        ),
    )
    parser.add_argument(
        "--verifier-adapter-path",
        default=None,
        help="Optional PEFT/LoRA adapter path for the verifier once Task 2 is ready.",
    )
    parser.add_argument("--image", default=None, help="Single image path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--pope-file", default=None, help="POPE JSONL file.")
    parser.add_argument("--image-root", default="data/benchmark/coco_subset")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--verifier-mode",
        choices=["json", "claim"],
        default="json",
        help=(
            "Use 'json' for zero-shot claim extraction+verification. Use 'claim' "
            "for the Task 2 SFT/GRPO verifier that classifies one claim at a time."
        ),
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Set to 5 for the Task 3 best-of-N baseline.",
    )
    parser.add_argument("--output", default=None, help="JSONL output path.")
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip POPE rows whose image is not present under --image-root.",
    )
    parser.add_argument(
        "--mock-backend",
        action="store_true",
        help=(
            "Run a deterministic no-dependency smoke backend. This validates the "
            "pipeline contract without downloading a VLM."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verifier_model_name = args.verifier_model_name or args.model_name

    if args.mock_backend:
        device = "mock"
        generator_model = MockVLM()
        generator_processor = None
        verifier_model = generator_model
        verifier_processor = None
    else:
        from accelerate import Accelerator

        accelerator = Accelerator()
        device = accelerator.device
        dtype = pick_model_dtype(device.type)

        generator_model, generator_processor = load_vlm(args.model_name, device, dtype)

        if verifier_model_name == args.model_name and not args.verifier_adapter_path:
            # Same VLM, same weights: reuse the generator in memory.
            verifier_model = generator_model
            verifier_processor = generator_processor
        else:
            verifier_model, verifier_processor = load_vlm(
                verifier_model_name,
                device,
                dtype,
                adapter_path=args.verifier_adapter_path,
            )

    rows = iter_inputs(args)
    iterator: Iterable[dict[str, str]] = rows
    if tqdm is not None:
        iterator = tqdm(rows, desc="Task 3 pipeline")

    output_f = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output_f = open(args.output, "w", encoding="utf-8")

    try:
        for row in iterator:
            image_path = row["image"]
            if not os.path.exists(image_path):
                if args.skip_missing_images:
                    if tqdm is None:
                        print(f"Skipping missing image: {image_path}")
                    continue
                raise FileNotFoundError(f"Image not found: {image_path}")

            start = time.time()
            response, verification, corrected = run_loop(
                generator_model,
                generator_processor,
                verifier_model,
                verifier_processor,
                device,
                image_path,
                row["prompt"],
                verifier_mode=args.verifier_mode,
                max_new_tokens=args.max_new_tokens,
                verifier_max_new_tokens=args.verifier_max_new_tokens,
            )

            best_response = None
            best_verification = None
            if args.num_candidates > 1:
                best_response, best_verification = run_best_of_n(
                    generator_model,
                    generator_processor,
                    verifier_model,
                    verifier_processor,
                    device,
                    image_path,
                    row["prompt"],
                    n=args.num_candidates,
                    verifier_mode=args.verifier_mode,
                    max_new_tokens=args.max_new_tokens,
                    verifier_max_new_tokens=args.verifier_max_new_tokens,
                )

            record = Task3Record(
                image=image_path,
                prompt=row["prompt"],
                single_pass_response=response,
                verification=verification,
                corrected_response=corrected,
                best_of_n_response=best_response,
                best_of_n_verification=best_verification,
                meta={
                    "model_name": args.model_name,
                    "verifier_model_name": verifier_model_name,
                    "verifier_adapter_path": args.verifier_adapter_path,
                    "verifier_mode": args.verifier_mode,
                    "num_candidates": args.num_candidates,
                    "latency_sec": time.time() - start,
                    "pope_question": row.get("pope_question"),
                    "pope_label": row.get("pope_label"),
                },
            )

            line = record_to_json(record)
            if output_f:
                output_f.write(line + "\n")
                output_f.flush()
            else:
                print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
    finally:
        if output_f:
            output_f.close()


if __name__ == "__main__":
    main()
