"""
Build the Task-1 dataset for VisualDebugger.

Pipeline:
1) For each image in data/VG/VG_500 and each hallucination type, query base VLM.
2) Split model response into claims.
3) Extract claim-specific Visual Genome ground-truth context.
4) Ask a judge VLM to label each claim as CORRECT or HALLUCINATED.
5) Save final dataset JSON rows with:
   - img_path
   - Question
   - Response
   - hal_type
   - claims
   - claim_labels
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

HALLUCINATION_TYPES = [
    "object_existence",
    "attribute_error",
    "spatial_error",
    "counting_error",
    "ocr_text_error",
    "action_event_error",
]

TYPE_DIRECTIVES = {
    "object_existence": (
        "Describe the objects you see in the image. "
        "Write exactly one object per sentence. "
        "Each sentence must follow this format: 'There is a [object] in the image.' "
        "Write 3 sentences."
    ),
    "attribute_error": (
        "Describe the visual properties of the objects in the image. "
        "Write exactly one attribute claim per sentence. "
        "Each sentence must follow this format: 'The [object] is [color/material/size].' "
        "Write 3 sentences."
    ),
    "spatial_error": (
        "Describe where the objects are located relative to each other. "
        "Write exactly one spatial relationship per sentence. "
        "Each sentence must follow this format: 'The [object] is [left of/right of/on top of/behind/in front of] the [object].' "
        "Write 3 sentences."
    ),
    "counting_error": (
        "Count the objects visible in the image. "
        "Write exactly one count claim per sentence. "
        "Each sentence must follow this format: 'There are [number] [objects] in the image.' "
        "Write 3 sentences."
    ),
    "ocr_text_error": (
        "Describe any text, signs, labels, or writing visible in the image. "
        "Write exactly one text claim per sentence. "
        "Each sentence must follow this format: 'The [sign/label/text] reads [text].' "
        "Write 3 sentences, or fewer if there is limited text in the image."
    ),
    "action_event_error": (
        "Describe object actions/events in the image. "
        "Write exactly one action claim per sentence. "
        "Each sentence must follow this format: 'The [object] is [action].' "
        "Write 3 sentences."
    ),
}

PROMPT_TEMPLATE = (
    "You are describing an image.\n"
    "Instruction: {directive}\n"
    "Use the following QA pairs as additional context about the image:\n"
    "{reference_block}\n"
    "Important: follow the sentence format exactly. Return only the description. Don't repeat the same claim multiple times."
)

HAL_TYPE_TO_EXTRACTOR = {
    "object_existence": "existence",
    "attribute_error": "attribute",
    "spatial_error": "spatial",
    "counting_error": "counting",
    "ocr_text_error": "ocr",
    "action_event_error": "action",
}


# -----------------------------------------------------------
# Device / model helpers
# -----------------------------------------------------------


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


def vlm_generate(
    model: Any,
    processor: Any,
    device: torch.device,
    image_path: str,
    text_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    model_input = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_len = len(model_input.input_ids[0])
    with torch.inference_mode():
        output_ids = model.generate(
            **model_input,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    text = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[
        0
    ]
    del output_ids
    del model_input
    return text.strip()


# -----------------------------------------------------------
# Data loading
# -----------------------------------------------------------


def load_question_answers(
    data_dir: str, qa_json_path: str | None = None
) -> dict[int, list[dict[str, Any]]]:
    qa_path = (
        Path(qa_json_path) if qa_json_path else Path(data_dir) / "question_answers.json"
    )
    if not qa_path.exists():
        return {}

    with qa_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    qa_by_image: dict[int, list[dict[str, Any]]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            image_id = item.get("id", item.get("image_id"))
            qas = item.get("qas")
            if image_id is None or not isinstance(qas, list):
                continue
            qa_by_image[int(image_id)] = qas
    return qa_by_image


def load_vg_data(data_dir: str, qa_json_path: str | None = None) -> dict[str, Any]:
    with open(f"{data_dir}/objects.json", encoding="utf-8") as f:
        objects_by_image = {img["image_id"]: img for img in json.load(f)}
    with open(f"{data_dir}/attributes.json", encoding="utf-8") as f:
        attributes_by_image = {img["image_id"]: img for img in json.load(f)}
    with open(f"{data_dir}/relationships.json", encoding="utf-8") as f:
        relationships_by_image = {img["image_id"]: img for img in json.load(f)}
    with open(f"{data_dir}/region_descriptions.json", encoding="utf-8") as f:
        regions = json.load(f)
        regions_by_image = {
            img.get("image_id", img.get("id")): img
            for img in regions
            if img.get("image_id", img.get("id")) is not None
        }

    question_answers_by_image = load_question_answers(data_dir, qa_json_path)

    return {
        "objects": objects_by_image,
        "attributes": attributes_by_image,
        "relationships": relationships_by_image,
        "regions": regions_by_image,
        "question_answers": question_answers_by_image,
    }


def get_image_paths(images_dir: str, max_images: int | None) -> list[Path]:
    paths = sorted(
        Path(images_dir).glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if max_images is not None:
        paths = paths[:max_images]
    return paths


# -----------------------------------------------------------
# Claim extraction helpers
# -----------------------------------------------------------


def split_claims(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    claims = [seg.strip() for seg in raw if seg.strip()]
    return claims if claims else ([text.strip()] if text.strip() else [])


def get_object_names(obj: dict[str, Any]) -> list[str]:
    names = obj.get("names")
    if isinstance(names, list):
        return [n.lower().strip() for n in names if isinstance(n, str) and n.strip()]

    name = obj.get("name")
    if isinstance(name, str) and name.strip():
        return [name.lower().strip()]

    return []


def get_primary_name(obj: dict[str, Any]) -> str:
    names = get_object_names(obj)
    return names[0] if names else ""


def normalize_token(token: str) -> str:
    t = re.sub(r"[^a-z0-9\- ]", "", token.lower()).strip()
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("sses", "xes", "ches", "shes", "zes", "oes")) and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


def get_mentioned_objects(
    claim: str, vg_objects: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    claim_lower = claim.lower()
    mentioned = []
    for obj in vg_objects:
        obj_names = get_object_names(obj)
        if any(name in claim_lower for name in obj_names):
            mentioned.append(obj)
    return mentioned


def extract_claimed_existence_objects(claim: str) -> list[str]:
    claim_lower = claim.lower()
    patterns = [
        r"\bthere\s+(?:is|are)\s+(?:an?|the|some)?\s*([a-z][a-z\s\-]*?)(?:\s+(?:in|on|at|near|by|with|of)\b|[.!?,]|$)",
        r"\bcontains?\s+(?:an?|the|some)?\s*([a-z][a-z\s\-]*?)(?:\s+(?:in|on|at|near|by|with|of)\b|[.!?,]|$)",
    ]

    stopwords = {"image", "picture", "photo", "scene", "frame", "the", "a", "an"}
    claimed = []

    for pattern in patterns:
        for phrase in re.findall(pattern, claim_lower):
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if not phrase:
                continue

            tokens = [normalize_token(tok) for tok in phrase.split()]
            tokens = [tok for tok in tokens if tok and tok not in stopwords]
            if not tokens:
                continue

            claimed.append(" ".join(tokens))
            claimed.append(tokens[-1])

    deduped = []
    seen = set()
    for item in claimed:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped


# -----------------------------------------------------------
# Ground-truth extractors per hallucination type
# -----------------------------------------------------------


def extract_gt_existence(
    claim: str, image_id: int, vg: dict[str, Any]
) -> dict[str, Any]:
    all_objects = vg["objects"][image_id]["objects"]
    vg_object_names = {
        normalize_token(name)
        for obj in all_objects
        for name in get_object_names(obj)
        if normalize_token(name)
    }

    mentioned = get_mentioned_objects(claim, all_objects)
    mentioned_names = [
        get_primary_name(obj) for obj in mentioned if get_primary_name(obj)
    ]

    claimed_candidates = extract_claimed_existence_objects(claim)
    missing_claimed_objects = [
        cand
        for cand in claimed_candidates
        if normalize_token(cand) not in vg_object_names
    ]

    return {
        "vg_object_names": sorted(vg_object_names),
        "mentioned_in_claim": mentioned_names,
        "claimed_objects": claimed_candidates,
        "missing_claimed_objects": missing_claimed_objects,
        "needs_llm_judge": True,
    }


def extract_gt_attribute(
    claim: str, image_id: int, vg: dict[str, Any]
) -> dict[str, Any]:
    all_objects = vg["attributes"][image_id]["attributes"]
    mentioned = get_mentioned_objects(claim, all_objects)

    relevant_attrs = [
        {"object": get_primary_name(obj), "attributes": obj.get("attributes", [])}
        for obj in mentioned
        if get_primary_name(obj)
    ]

    return {
        "relevant_attributes": relevant_attrs,
        "needs_llm_judge": True,
    }


def extract_gt_spatial(claim: str, image_id: int, vg: dict[str, Any]) -> dict[str, Any]:
    all_objects = vg["objects"][image_id]["objects"]
    mentioned = get_mentioned_objects(claim, all_objects)
    mentioned_ids = {obj["object_id"] for obj in mentioned}

    all_rels = vg["relationships"][image_id]["relationships"]
    relevant_rels = [
        {
            "subject": rel["subject"]["name"],
            "predicate": rel["predicate"],
            "object": rel["object"]["name"],
        }
        for rel in all_rels
        if rel["subject"]["object_id"] in mentioned_ids
        and rel["object"]["object_id"] in mentioned_ids
    ]

    return {
        "relevant_relationships": relevant_rels,
        "needs_llm_judge": True,
    }


def extract_gt_counting(
    claim: str, image_id: int, vg: dict[str, Any]
) -> dict[str, Any]:
    all_objects = vg["objects"][image_id]["objects"]
    mentioned = get_mentioned_objects(claim, all_objects)

    counts = {}
    for obj in all_objects:
        for name in get_object_names(obj):
            counts[name] = counts.get(name, 0) + 1

    mentioned_counts = {
        get_primary_name(obj): counts.get(get_primary_name(obj), 0)
        for obj in mentioned
        if get_primary_name(obj)
    }

    numbers_in_claim = re.findall(r"\b\d+\b", claim)

    return {
        "vg_counts": mentioned_counts,
        "numbers_in_claim": numbers_in_claim,
        "needs_llm_judge": False,
    }


def extract_gt_action(claim: str, image_id: int, vg: dict[str, Any]) -> dict[str, Any]:
    all_regions = vg["regions"][image_id]["regions"]
    region_phrases = [region["phrase"] for region in all_regions]

    all_objects = vg["objects"][image_id]["objects"]
    mentioned = get_mentioned_objects(claim, all_objects)
    mentioned_names = {
        get_primary_name(obj) for obj in mentioned if get_primary_name(obj)
    }

    relevant_phrases = [
        phrase
        for phrase in region_phrases
        if any(name in phrase.lower() for name in mentioned_names)
    ] or region_phrases

    return {
        "relevant_region_descriptions": relevant_phrases,
        "needs_llm_judge": True,
    }


def extract_gt_ocr(claim: str, image_id: int, vg: dict[str, Any]) -> dict[str, Any]:
    all_regions = vg["regions"][image_id]["regions"]
    text_keywords = {"text", "sign", "word", "letter", "written", "reads", "says"}
    text_regions = [
        region["phrase"]
        for region in all_regions
        if any(kw in region["phrase"].lower() for kw in text_keywords)
    ]

    return {
        "text_region_descriptions": text_regions,
        "needs_llm_judge": True,
        "warning": "VG has limited OCR annotations; consider TextVQA for this type.",
    }


EXTRACTORS = {
    "existence": extract_gt_existence,
    "attribute": extract_gt_attribute,
    "spatial": extract_gt_spatial,
    "counting": extract_gt_counting,
    "action": extract_gt_action,
    "ocr": extract_gt_ocr,
}


def extract_ground_truth(
    claim: str, image_id: int, hal_type: str, vg: dict[str, Any]
) -> dict[str, Any]:
    if hal_type not in EXTRACTORS:
        raise ValueError(
            f"Unknown hallucination type: {hal_type}. Choose from {list(EXTRACTORS.keys())}"
        )
    return EXTRACTORS[hal_type](claim, image_id, vg)


# -----------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------


def build_reference_block(image_qas: list[dict[str, Any]]) -> str:
    if not image_qas:
        return "- No QA context available for this image."

    lines = []
    for qa in image_qas[:8]:
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if question or answer:
            lines.append(f"- Q: {question} | A: {answer}")

    return "\n".join(lines) if lines else "- No usable QA context available."


def build_generation_prompt(hal_type: str, image_qas: list[dict[str, Any]]) -> str:
    directive = TYPE_DIRECTIVES[hal_type]
    reference_block = build_reference_block(image_qas)
    return PROMPT_TEMPLATE.format(directive=directive, reference_block=reference_block)


def build_judge_prompt(
    hal_type: str,
    claims: list[str],
    gt_contexts: list[dict[str, Any]],
) -> str:
    payload = {
        "hallucination_type": hal_type,
        "claims": claims,
        "ground_truth_context_per_claim": gt_contexts,
    }

    return (
        "You are a strict hallucination judge for a vision-language model.\n"
        "Use ONLY the provided image and the structured ground-truth context.\n"
        "Label each claim as CORRECT or HALLUCINATED.\n"
        "Return valid JSON only with this schema:\n"
        '{"claim_labels": ["CORRECT" | "HALLUCINATED", ...]}\n'
        "The number of labels must equal the number of claims.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_judge_labels(raw_text: str, n_claims: int) -> list[str]:
    label_pattern = re.compile(r"\b(CORRECT|HALLUCINATED)\b", re.IGNORECASE)

    try:
        parsed = json.loads(raw_text)
        labels = parsed.get("claim_labels", []) if isinstance(parsed, dict) else []
        if isinstance(labels, list):
            norm = []
            for item in labels:
                s = str(item).strip().upper()
                if s not in {"CORRECT", "HALLUCINATED"}:
                    s = "HALLUCINATED"
                norm.append(s)
            if len(norm) == n_claims:
                return norm
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            labels = parsed.get("claim_labels", []) if isinstance(parsed, dict) else []
            if isinstance(labels, list) and len(labels) == n_claims:
                return [
                    (
                        str(x).strip().upper()
                        if str(x).strip().upper() in {"CORRECT", "HALLUCINATED"}
                        else "HALLUCINATED"
                    )
                    for x in labels
                ]
        except Exception:
            pass

    fallback = [m.group(1).upper() for m in label_pattern.finditer(raw_text)]
    if len(fallback) >= n_claims:
        return fallback[:n_claims]

    return ["HALLUCINATED"] * n_claims


# -----------------------------------------------------------
# Main dataset construction
# -----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VisualDebugger Task-1 dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "generate-only", "judge-only"],
        default="full",
        help="full: generate+judge, generate-only: base VLM only, judge-only: judge existing JSON",
    )
    parser.add_argument(
        "--vg-dir", default="data/VG", help="Directory containing VG JSON files"
    )
    parser.add_argument(
        "--images-dir",
        default="data/VG/VG_500",
        help="Directory containing downloaded JPG images",
    )
    parser.add_argument(
        "--qa-json", default=None, help="Optional QA context JSON override"
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="Input dataset JSON used in judge-only mode (defaults to --output-json)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base VLM for response generation",
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Judge VLM for claim labeling",
    )
    parser.add_argument(
        "--output-json",
        default="generated_dataset.json",
        help="Final dataset output file",
    )
    parser.add_argument(
        "--num-generations-per-type",
        type=int,
        default=1,
        help="Generations per image per hallucination type",
    )
    parser.add_argument(
        "--max-images", type=int, default=None, help="Optional cap on number of images"
    )
    parser.add_argument("--base-max-new-tokens", type=int, default=220)
    parser.add_argument("--judge-max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip judge stage and leave claim labels empty",
    )
    parser.add_argument(
        "--save-every", type=int, default=100, help="Checkpoint save interval"
    )
    return parser.parse_args()


def save_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_rows_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(data).__name__}")

    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def infer_image_id_from_path(img_path: str) -> int:
    return int(Path(img_path).stem)


def run_judging_stage(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    vg: dict[str, dict[int, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    print(f"Loading judge model: {args.judge_model}")
    judge_model, judge_processor = load_vlm(args.judge_model, device)

    for idx, row in enumerate(tqdm(rows, desc="Judging claims", unit="row")):
        claims = row.get("claims") or split_claims(str(row.get("Response", "")))
        row["claims"] = claims

        if not claims:
            row["claim_labels"] = []
            continue

        image_id = infer_image_id_from_path(str(row["img_path"]))
        if image_id not in vg["objects"]:
            row["claim_labels"] = ["HALLUCINATED"] * len(claims)
            continue

        extractor_key = HAL_TYPE_TO_EXTRACTOR[row["hal_type"]]
        gt_contexts = [
            extract_ground_truth(c, image_id, extractor_key, vg) for c in claims
        ]

        judge_prompt = build_judge_prompt(str(row["hal_type"]), claims, gt_contexts)
        judge_raw = vlm_generate(
            model=judge_model,
            processor=judge_processor,
            device=device,
            image_path=str(row["img_path"]),
            text_prompt=judge_prompt,
            max_new_tokens=args.judge_max_new_tokens,
            temperature=0.1,
            top_p=0.9,
        )

        row["claim_labels"] = parse_judge_labels(judge_raw, len(claims))

        if idx % args.save_every == 0:
            save_json(output_path, rows)

    unload_vlm(judge_model, judge_processor, device)
    return rows


def main() -> None:
    args = parse_args()

    device = pick_device()
    print(f"Device: {device.type}")
    print(f"Using dtype: {pick_dtype(device)}")

    output_path = Path(args.output_json)
    vg = load_vg_data(args.vg_dir, args.qa_json)

    rows: list[dict[str, Any]] = []

    if args.mode in {"full", "generate-only"}:
        image_paths = get_image_paths(args.images_dir, args.max_images)
        qa_map = vg["question_answers"]
        if not image_paths:
            raise ValueError(f"No images found in {args.images_dir}")

        print(f"Loading base model: {args.base_model}")
        base_model, base_processor = load_vlm(args.base_model, device)

        total_jobs = (
            len(image_paths) * len(HALLUCINATION_TYPES) * args.num_generations_per_type
        )

        gen_bar = tqdm(total=total_jobs, desc="Generating responses", unit="gen")

        for image_path in image_paths:
            image_id = int(image_path.stem)

            if image_id not in vg["objects"]:
                continue

            for hal_type in HALLUCINATION_TYPES:
                question = TYPE_DIRECTIVES[hal_type]
                image_qas = qa_map.get(image_id, [])
                prompt = build_generation_prompt(hal_type, image_qas)

                for _ in range(args.num_generations_per_type):
                    response = vlm_generate(
                        model=base_model,
                        processor=base_processor,
                        device=device,
                        image_path=str(image_path),
                        text_prompt=prompt,
                        max_new_tokens=args.base_max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                    claims = split_claims(response)
                    rows.append(
                        {
                            "image_id": image_id,
                            "img_path": str(image_path).replace("\\\\", "/"),
                            "Question": question,
                            "Response": response,
                            "hal_type": hal_type,
                            "claims": claims,
                            "claim_labels": [],
                        }
                    )

                    gen_bar.update(1)

                    if len(rows) % args.save_every == 0:
                        save_json(output_path, rows)

        gen_bar.close()
        unload_vlm(base_model, base_processor, device)

    if args.mode == "judge-only":
        input_path = Path(args.input_json) if args.input_json else output_path
        if not input_path.exists():
            raise ValueError(
                f"Judge-only mode requires existing input JSON: {input_path}"
            )
        rows = load_rows_json(input_path)

    if args.mode in {"full", "judge-only"} and not args.skip_judge:
        rows = run_judging_stage(rows, args, device, vg, output_path)

    save_json(output_path, rows)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
