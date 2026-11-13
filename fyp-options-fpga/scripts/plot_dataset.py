"""Plot training relationships from a generated Black-Scholes dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(dataset_path: str | Path, output_path: str | Path) -> Path:
    """Create call-price, moneyness, and normalized-call plots."""
    dataset = Path(dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(dataset) as data:
        calls = data["y_call_train"]
        moneyness = data["moneyness_train"]
        call_over_k = data["call_over_K_train"]

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(calls, bins=100, color="steelblue", edgecolor="white")
    axes[0].set(title="Training call prices", xlabel="Call price", ylabel="Count")
    axes[1].scatter(
        moneyness, calls, s=2, alpha=0.15, color="darkorange", rasterized=True
    )
    axes[1].set(title="Call price vs moneyness", xlabel="S/K", ylabel="Call price")
    axes[2].scatter(
        moneyness, call_over_k, s=2, alpha=0.15, color="seagreen", rasterized=True
    )
    axes[2].set(title="Call/K vs moneyness", xlabel="S/K", ylabel="Call/K")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/black_scholes_dataset.npz",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="data/black_scholes_training_plots.png",
        type=Path,
    )
    args = parser.parse_args()
    print(f"Saved plot to {plot_dataset(args.dataset, args.output)}")


if __name__ == "__main__":
    main()
