import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def list_numeric_keys(log_history: list[dict[str, Any]]) -> list[str]:
    keys = set()

    for row in log_history:
        for k, v in row.items():
            if is_number(v):
                keys.add(k)

    return sorted(keys)


def find_best_key(
    numeric_keys: list[str],
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
) -> str | None:
    exclude_patterns = exclude_patterns or []

    candidates = []

    for key in numeric_keys:
        key_lower = key.lower()

        if all(pattern.lower() in key_lower for pattern in include_patterns):
            if not any(pattern.lower() in key_lower for pattern in exclude_patterns):
                candidates.append(key)

    if not candidates:
        return None

    # Prefer shorter / simpler key names.
    candidates = sorted(candidates, key=lambda x: (len(x), x))
    return candidates[0]


def extract_curve(
    log_history: list[dict[str, Any]],
    key: str,
    window: int,
) -> pd.DataFrame:
    rows = []

    for i, row in enumerate(log_history):
        if key not in row:
            continue

        value = row[key]
        if not is_number(value):
            continue

        step = row.get("step", i + 1)
        if not is_number(step):
            step = i + 1

        rows.append(
            {
                "step": float(step),
                "value": float(value),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values("step").reset_index(drop=True)
    df["smoothed"] = df["value"].rolling(
        window=window,
        min_periods=1,
        center=True,
    ).mean()

    return df


def plot_single_curve(
    df: pd.DataFrame,
    key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    show_raw: bool = True,
) -> None:
    if df.empty:
        print(f"Skipping {key}: no data.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    if show_raw:
        plt.plot(
            df["step"],
            df["value"],
            alpha=0.25,
            linewidth=1,
            label=f"raw {key}",
        )

    plt.plot(
        df["step"],
        df["smoothed"],
        linewidth=2.5,
        label=f"smoothed {key}",
    )

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-path",
        required=True,
        help="Path to metrics/<run_name>/grpo_log_history.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots. Default: same folder as log file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Moving average window size.",
    )
    parser.add_argument(
        "--reward-key",
        default=None,
        help="Optional manual reward key.",
    )
    parser.add_argument(
        "--kl-key",
        default=None,
        help="Optional manual KL key.",
    )
    parser.add_argument(
        "--reward-std-key",
        default=None,
        help="Optional manual reward std key.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Only plot smoothed curves, without raw values.",
    )

    args = parser.parse_args()

    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir) if args.output_dir else log_path.parent

    log_history = json.loads(log_path.read_text(encoding="utf-8"))

    if not isinstance(log_history, list):
        raise ValueError("Expected log history to be a list of dictionaries.")

    numeric_keys = list_numeric_keys(log_history)

    print("\nAvailable numeric keys:")
    for key in numeric_keys:
        print(f"  - {key}")

    # Auto-detect keys.
    reward_key = args.reward_key or find_best_key(
        numeric_keys,
        include_patterns=["reward"],
        exclude_patterns=["std", "standard", "kl"],
    )

    kl_key = args.kl_key or find_best_key(
        numeric_keys,
        include_patterns=["kl"],
    )

    reward_std_key = args.reward_std_key or find_best_key(
        numeric_keys,
        include_patterns=["reward", "std"],
    )

    print("\nSelected keys:")
    print(f"  reward_key     = {reward_key}")
    print(f"  kl_key         = {kl_key}")
    print(f"  reward_std_key = {reward_std_key}")

    show_raw = not args.no_raw

    if reward_key:
        reward_df = extract_curve(log_history, reward_key, args.window)
        plot_single_curve(
            reward_df,
            key=reward_key,
            ylabel="Reward",
            title=f"GRPO Reward Progression, moving average window={args.window}",
            output_path=output_dir / "grpo_reward_smoothed.png",
            show_raw=show_raw,
        )
    else:
        print("No reward key found. Skipping reward plot.")

    if kl_key:
        kl_df = extract_curve(log_history, kl_key, args.window)
        plot_single_curve(
            kl_df,
            key=kl_key,
            ylabel="KL",
            title=f"GRPO KL Progression, moving average window={args.window}",
            output_path=output_dir / "grpo_kl_smoothed.png",
            show_raw=show_raw,
        )
    else:
        print("No KL key found. Skipping KL plot.")

    if reward_std_key:
        reward_std_df = extract_curve(log_history, reward_std_key, args.window)
        plot_single_curve(
            reward_std_df,
            key=reward_std_key,
            ylabel="Reward standard deviation",
            title=f"GRPO Reward Std Progression, moving average window={args.window}",
            output_path=output_dir / "grpo_reward_std_smoothed.png",
            show_raw=show_raw,
        )
    else:
        print("No reward_std key found. Skipping reward std plot.")


if __name__ == "__main__":
    main()