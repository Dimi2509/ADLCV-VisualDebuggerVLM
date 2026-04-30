import json

import matplotlib.pyplot as plt


def plot_benchmark_results(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in results:
        name = model["model"].split("/")[-1]
        avg_latency_seconds = model["avg_latency_sec"]
        ax.bar(name, avg_latency_seconds, label=name, alpha=0.7, edgecolor="black")

    ax.set_xlabel("Model")
    ax.set_ylabel("Average Latency (seconds)")
    ax.set_title("Benchmark Results: Average Latency per Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig("benchmark_results.png")


if __name__ == "__main__":
    with open("benchmark_results.json", "r") as f:
        results = json.load(f)

    plot_benchmark_results(results)
