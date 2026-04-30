#!/usr/bin/env python3
"""Evaluate Task 3 JSONL outputs on verifier and POPE-consistency metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


RESPONSE_FIELDS = {
    "single_pass": "single_pass_response",
    "corrected": "corrected_response",
    "best_of_n": "best_of_n_response",
}

VERIFICATION_FIELDS = {
    "single_pass": "verification",
    "best_of_n": "best_of_n_verification",
}


def load_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return records


def is_hallucinated_claim(claim: dict[str, Any]) -> bool:
    return str(claim.get("label", "")).strip().upper() == "HALLUCINATED"


def summarize_verification(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    rows = [r for r in records if isinstance(r.get(field), list)]
    claim_counts = [len(r[field]) for r in rows]
    hallucinated_counts = [
        sum(is_hallucinated_claim(claim) for claim in r[field]) for r in rows
    ]
    total_claims = sum(claim_counts)
    total_hallucinated = sum(hallucinated_counts)

    return {
        "samples_evaluated": len(rows),
        "total_claims": total_claims,
        "total_flagged_hallucinated_claims": total_hallucinated,
        "avg_claims_per_sample": mean(claim_counts) if claim_counts else 0.0,
        "avg_flagged_hallucinated_claims_per_sample": (
            mean(hallucinated_counts) if hallucinated_counts else 0.0
        ),
        "samples_with_any_flagged_hallucination": sum(
            count > 0 for count in hallucinated_counts
        ),
        "sample_flagged_hallucination_rate": (
            sum(count > 0 for count in hallucinated_counts) / len(rows) if rows else 0.0
        ),
        "claim_flagged_hallucination_rate": (
            total_hallucinated / total_claims if total_claims else 0.0
        ),
    }


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize(noun: str) -> str:
    if noun.endswith("ies") and len(noun) > 3:
        return noun[:-3] + "y"
    if noun.endswith("ses"):
        return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss") and len(noun) > 3:
        return noun[:-1]
    return noun


def extract_pope_object(question: str) -> str | None:
    normalized = question.strip().rstrip("?")
    patterns = [
        r"^is there an? (.+?) in the image$",
        r"^are there (?:any )?(.+?) in the image$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(1))
    return None


def object_terms(obj: str) -> set[str]:
    terms = {obj}
    tokens = obj.split()
    if tokens:
        singular_last = singularize(tokens[-1])
        if singular_last != tokens[-1]:
            terms.add(" ".join(tokens[:-1] + [singular_last]))
    return terms


def has_negated_object(text: str, term: str) -> bool:
    escaped = re.escape(term)
    negation_patterns = [
        rf"\bno\s+(?:visible\s+)?{escaped}\b",
        rf"\bnot\s+(?:a\s+|an\s+|any\s+)?{escaped}\b",
        rf"\bwithout\s+(?:a\s+|an\s+|any\s+)?{escaped}\b",
    ]
    return any(re.search(pattern, text) for pattern in negation_patterns)


def infer_pope_presence(response: str | None, question: str | None) -> str:
    if not response or not question:
        return "unknown"
    obj = extract_pope_object(question)
    if not obj:
        return "unknown"

    text = normalize_text(response)
    for term in object_terms(obj):
        if has_negated_object(text, term):
            return "no"
    for term in object_terms(obj):
        if re.search(rf"\b{re.escape(term)}\b", text):
            return "yes"
    return "no"


def compute_binary_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    tp = fp = tn = fn = unknown = 0
    for pred, gt in pairs:
        if pred not in {"yes", "no"}:
            unknown += 1
            if gt == "yes":
                fn += 1
            elif gt == "no":
                tn += 1
            continue

        if gt == "yes":
            if pred == "yes":
                tp += 1
            else:
                fn += 1
        elif gt == "no":
            if pred == "yes":
                fp += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "samples_evaluated": total,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "hallucination_rate": fp / max(fp + tn, 1),
        "unknown_rate": unknown / max(total, 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def summarize_pope_consistency(
    records: list[dict[str, Any]], response_field: str
) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    missing_response = 0
    missing_meta = 0

    for record in records:
        meta = record.get("meta") or {}
        question = meta.get("pope_question")
        gt = str(meta.get("pope_label", "")).strip().lower()
        response = record.get(response_field)

        if not question or gt not in {"yes", "no"}:
            missing_meta += 1
            continue
        if not response:
            missing_response += 1
            continue
        pairs.append((infer_pope_presence(str(response), str(question)), gt))

    metrics = compute_binary_metrics(pairs)
    metrics.update(
        {
            "response_field": response_field,
            "samples_missing_response": missing_response,
            "samples_missing_pope_metadata": missing_meta,
        }
    )
    return metrics


def correction_rate(records: list[dict[str, Any]]) -> float:
    comparable = [
        r
        for r in records
        if r.get("single_pass_response") is not None and r.get("corrected_response") is not None
    ]
    if not comparable:
        return 0.0
    changed = sum(
        str(r.get("single_pass_response", "")).strip()
        != str(r.get("corrected_response", "")).strip()
        for r in comparable
    )
    return changed / len(comparable)


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    verifier_metrics = {
        name: summarize_verification(records, field)
        for name, field in VERIFICATION_FIELDS.items()
    }
    pope_metrics = {
        name: summarize_pope_consistency(records, field)
        for name, field in RESPONSE_FIELDS.items()
    }
    latencies = [
        float((r.get("meta") or {}).get("latency_sec"))
        for r in records
        if isinstance((r.get("meta") or {}).get("latency_sec"), (int, float))
    ]
    return {
        "samples": len(records),
        "correction_rate": correction_rate(records),
        "avg_latency_sec": mean(latencies) if latencies else 0.0,
        "verifier_flag_metrics": verifier_metrics,
        "pope_consistency_metrics": pope_metrics,
    }


def format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def print_text_report(metrics: dict[str, Any]) -> None:
    print("Task 3 POPE Evaluation")
    print(f"Samples: {metrics['samples']}")
    print(f"Correction rate: {format_percent(metrics['correction_rate'])}")
    print(f"Average latency: {metrics['avg_latency_sec']:.3f}s")
    print()

    print("Verifier-Flag Metrics")
    for name, row in metrics["verifier_flag_metrics"].items():
        if row["samples_evaluated"] == 0:
            print(f"  {name}: no verification records")
            continue
        print(
            f"  {name}: avg flagged/sample="
            f"{row['avg_flagged_hallucinated_claims_per_sample']:.3f}, "
            f"sample flagged rate={format_percent(row['sample_flagged_hallucination_rate'])}, "
            f"claim flagged rate={format_percent(row['claim_flagged_hallucination_rate'])}"
        )
    print()

    print("POPE Object-Consistency Metrics")
    print("  split        acc      prec     recall   f1       halluc   unknown")
    for name, row in metrics["pope_consistency_metrics"].items():
        if row["samples_evaluated"] == 0:
            print(f"  {name:<11} no scored responses")
            continue
        print(
            f"  {name:<11} "
            f"{row['accuracy']:.4f}  {row['precision']:.4f}  "
            f"{row['recall']:.4f}  {row['f1']:.4f}  "
            f"{row['hallucination_rate']:.4f}  {row['unknown_rate']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Task 3 JSONL output file.")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Report format (default: %(default)s).",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON metrics path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    metrics = evaluate(records)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if args.format == "json":
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        print_text_report(metrics)


if __name__ == "__main__":
    main()
