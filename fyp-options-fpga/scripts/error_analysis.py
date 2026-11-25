"""Plot test-set absolute call/K error for the selected NN and polynomial baseline."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from optfpga.nn.baseline import fit_polynomial_regression, polynomial_features
from optfpga.nn.model import MLP


def _network_predictions(
    run_dir: Path,
    reduced_inputs: np.ndarray,
) -> np.ndarray:
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = MLP(
        checkpoint["input_size"],
        checkpoint["hidden_sizes"],
        checkpoint["activation"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with np.load(run_dir / "normalization.npz") as normalization:
        mean = normalization["X_reduced_mean"].astype(np.float32)
        std = normalization["X_reduced_std"].astype(np.float32)
    inputs = ((reduced_inputs - mean) / std).astype(np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(inputs)).numpy().reshape(-1).astype(np.float64)


def create_error_analysis(
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
    network_run: str | Path = "results/b2_selected_seed_42",
    output_dir: str | Path = "docs/figures",
) -> list[Path]:
    """Create four test-only plots comparing network and polynomial errors."""
    dataset_path = Path(dataset_path)
    network_run = Path(network_run)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(dataset_path) as data:
        reduced = data["X_reduced_test"].astype(np.float64)
        true_call_over_k = data["call_over_K_test"].astype(np.float64)
        full_inputs = data["X_test"].astype(np.float64)
    network_prediction = _network_predictions(network_run, reduced)
    polynomial = fit_polynomial_regression(dataset_path)
    normalized = (reduced - polynomial["mean"]) / polynomial["std"]
    polynomial_prediction = (
        polynomial_features(normalized, polynomial["chosen_degree"])
        @ polynomial["chosen_coefficients"]
    )

    errors = {
        "Neural network": np.abs(network_prediction - true_call_over_k),
        "Polynomial": np.abs(polynomial_prediction - true_call_over_k),
    }

    moneyness = full_inputs[:, 0] / full_inputs[:, 1]
    sigma_sqrt_t = full_inputs[:, 4] * np.sqrt(full_inputs[:, 2])
    standardized_moneyness = np.log(moneyness) / sigma_sqrt_t
    true_price = true_call_over_k * full_inputs[:, 1]
    plots = [
        ("moneyness", moneyness, "Moneyness S/K", False),
        ("sigma_sqrt_T", sigma_sqrt_t, "Volatility-time sigma*sqrt(T)", False),
        (
            "standardized_moneyness",
            standardized_moneyness,
            "Standardized moneyness ln(S/K)/(sigma*sqrt(T))",
            False,
        ),
        ("true_price", true_price, "True call price", True),
    ]

    output_paths = []
    for suffix, x_values, xlabel, log_x in plots:
        figure, axis = plt.subplots(figsize=(8, 5))
        for label, absolute_error in errors.items():
            valid = np.isfinite(x_values) & np.isfinite(absolute_error)
            if log_x:
                valid &= x_values > 0
            axis.scatter(
                x_values[valid],
                absolute_error[valid],
                s=7,
                alpha=0.25,
                label=label,
                rasterized=True,
            )
        if log_x:
            axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Absolute call/K error")
        axis.set_title(f"Test-set absolute call/K error vs {xlabel}")
        axis.legend()
        figure.tight_layout()
        output_path = output_dir / f"absolute_call_over_k_error_{suffix}.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/black_scholes_dataset.npz"))
    parser.add_argument("--network-run", type=Path, default=Path("results/b2_selected_seed_42"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    paths = create_error_analysis(args.dataset, args.network_run, args.output_dir)
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
