import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("precision", "recall", "f1")


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid or empty JSON metrics file: {path}")

    return data


def get_hallucination_types(experiment_metrics: list[dict]) -> list[str]:
    # Use the first file as canonical ordering, then append any missing types from others.
    ordered = list(experiment_metrics[0].keys())
    seen = set(ordered)

    for metrics in experiment_metrics[1:]:
        for hal_type in metrics.keys():
            if hal_type not in seen:
                ordered.append(hal_type)
                seen.add(hal_type)

    return ordered


def extract_values(metrics: dict, hall_types: list[str], metric_name: str) -> np.ndarray:
    values = []
    for hal_type in hall_types:
        metric_value = metrics.get(hal_type, {}).get(metric_name, np.nan)
        values.append(metric_value)
    return np.array(values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot precision, recall, and F1 per hallucination type using bar plots."
    )
    parser.add_argument(
        "--metrics-files",
        nargs="+",
        default=["baseline_evaluate_metrics.json", "sft_evaluate_metrics.json"],
        help="One or more evaluation metrics JSON files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels corresponding to --metrics-files. Defaults to file stem names.",
    )
    parser.add_argument(
        "--output",
        default="hallucination_metrics_barplots.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI for saved figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metric_paths = [Path(p) for p in args.metrics_files]
    for p in metric_paths:
        if not p.exists():
            raise FileNotFoundError(f"Metrics file not found: {p}")

    if args.labels is None:
        labels = [p.stem for p in metric_paths]
    else:
        labels = args.labels
        if len(labels) != len(metric_paths):
            raise ValueError("Number of --labels must match number of --metrics-files")

    all_metrics = [load_metrics(p) for p in metric_paths]
    hall_types = get_hallucination_types(all_metrics)

    x = np.arange(len(hall_types))
    n_experiments = len(all_metrics)
    width = 0.8 / max(n_experiments, 1)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)

    for ax, metric_name in zip(axes, METRICS, strict=True):
        for i, (exp_metrics, label) in enumerate(zip(all_metrics, labels, strict=True)):
            values = extract_values(exp_metrics, hall_types, metric_name)
            offsets = x - 0.4 + width / 2 + i * width
            ax.bar(offsets, values, width=width, label=label)

        ax.set_title(metric_name.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(hall_types, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Score")
    fig.suptitle("Precision / Recall / F1 per Hallucination Type", y=1.02)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncols=max(len(labels), 1), frameon=False)
    fig.tight_layout()

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
