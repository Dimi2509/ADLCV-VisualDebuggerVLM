import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def norm_label(x: Any) -> str:
    return str(x).strip().upper()


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def summarize_counter(counter: Counter, positive: str, negative: str) -> dict[str, Any]:
    total = sum(counter.values())
    pos = counter.get(positive, 0)
    neg = counter.get(negative, 0)

    other = {
        k: v for k, v in sorted(counter.items())
        if k not in {positive, negative}
    }

    return {
        "total": total,
        positive: pos,
        negative: neg,
        f"{positive}_pct": pct(pos, total),
        f"{negative}_pct": pct(neg, total),
        "other_labels": other,
    }


def compute_stats(
    rows: list[dict[str, Any]],
    positive: str = "HALLUCINATED",
    negative: str = "CORRECT",
) -> dict[str, Any]:
    overall = Counter()
    per_type = defaultdict(Counter)

    malformed_rows = 0
    usable_pairs = 0

    for row in rows:
        hal_type = str(row.get("hal_type", "unknown"))
        claims = row.get("claims", [])
        labels = row.get("claim_labels", [])

        if not isinstance(claims, list) or not isinstance(labels, list):
            malformed_rows += 1
            continue

        if len(claims) != len(labels):
            malformed_rows += 1

        # Important: multiple claims per item are handled here.
        n = min(len(claims), len(labels))
        usable_pairs += n

        for i in range(n):
            label = norm_label(labels[i])
            overall[label] += 1
            per_type[hal_type][label] += 1

    return {
        "meta": {
            "rows_total": len(rows),
            "rows_malformed_or_mismatch": malformed_rows,
            "usable_claim_label_pairs": usable_pairs,
            "positive_label": positive,
            "negative_label": negative,
        },
        "overall": summarize_counter(overall, positive, negative),
        "per_hallucination_type": {
            hal_type: summarize_counter(cnt, positive, negative)
            for hal_type, cnt in sorted(per_type.items(), key=lambda kv: kv[0])
        },
    }


def print_report(stats: dict[str, Any]) -> None:
    meta = stats["meta"]
    overall = stats["overall"]
    per_type = stats["per_hallucination_type"]
    pos = meta["positive_label"]
    neg = meta["negative_label"]

    print("=== Dataset Distribution ===")
    print(f"Rows total: {meta['rows_total']}")
    print(f"Rows malformed/mismatch: {meta['rows_malformed_or_mismatch']}")
    print(f"Usable claim-label pairs: {meta['usable_claim_label_pairs']}")
    print()

    print("=== Overall ===")
    print(f"Total claims: {overall['total']}")
    print(f"{pos}: {overall[pos]} ({overall[f'{pos}_pct']:.2f}%)")
    print(f"{neg}: {overall[neg]} ({overall[f'{neg}_pct']:.2f}%)")
    if overall["other_labels"]:
        print(f"Other labels: {overall['other_labels']}")
    print()

    print("=== Per hallucination type ===")
    for hal_type, s in per_type.items():
        print(f"- {hal_type}")
        print(f"  Total: {s['total']}")
        print(f"  {pos}: {s[pos]} ({s[f'{pos}_pct']:.2f}%)")
        print(f"  {neg}: {s[neg]} ({s[f'{neg}_pct']:.2f}%)")
        if s["other_labels"]:
            print(f"  Other labels: {s['other_labels']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute overall and per-hallucination-type CORRECT/HALLUCINATED stats."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to dataset JSON with fields: hal_type, claims, claim_labels",
    )
    parser.add_argument(
        "--positive-label",
        default="HALLUCINATED",
        help="Positive class label (default: HALLUCINATED)",
    )
    parser.add_argument(
        "--negative-label",
        default="CORRECT",
        help="Negative class label (default: CORRECT)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write stats JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items.")

    pos = norm_label(args.positive_label)
    neg = norm_label(args.negative_label)

    stats = compute_stats(data, positive=pos, negative=neg)
    print_report(stats)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print()
        print(f"Saved report JSON to: {out_path.as_posix()}")


if __name__ == "__main__":
    main()