"""Polynomial-regression baseline for Black-Scholes call/K approximation."""

import csv
import json
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optfpga.nn.evaluate import compute_metrics


def polynomial_features(
    inputs: NDArray[np.floating],
    degree: int,
) -> NDArray[np.float64]:
    """Expand normalized 4-column inputs into all monomials through degree d."""
    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != 4:
        raise ValueError("inputs must have shape (N, 4)")
    if not 1 <= degree <= 6:
        raise ValueError("degree must be between 1 and 6")

    columns = [np.ones(inputs.shape[0], dtype=np.float64)]
    for current_degree in range(1, degree + 1):
        for variables in combinations_with_replacement(range(4), current_degree):
            column = np.ones(inputs.shape[0], dtype=np.float64)
            for variable in variables:
                column *= inputs[:, variable]
            columns.append(column)
    return np.column_stack(columns)


def fit_polynomial_degree(
    train_inputs: NDArray[np.floating],
    train_targets: NDArray[np.floating],
    degree: int,
    mean: NDArray[np.floating] | None = None,
    std: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float | int]]:
    """Fit one polynomial degree with lstsq and return coefficients and train data."""
    train_inputs = np.asarray(train_inputs, dtype=np.float64)
    train_targets = np.asarray(train_targets, dtype=np.float64).reshape(-1)
    if train_inputs.ndim != 2 or train_inputs.shape[1] != 4:
        raise ValueError("train_inputs must have shape (N, 4)")
    if train_targets.shape[0] != train_inputs.shape[0]:
        raise ValueError("train inputs and targets must have equal lengths")

    if mean is None:
        mean = train_inputs.mean(axis=0)
    if std is None:
        std = train_inputs.std(axis=0)
    std = np.asarray(std, dtype=np.float64)
    if np.any(std == 0):
        raise ValueError("normalisation standard deviations must be non-zero")

    normalized = (train_inputs - np.asarray(mean, dtype=np.float64)) / std
    design = polynomial_features(normalized, degree)
    coefficients, _, _, _ = np.linalg.lstsq(design, train_targets, rcond=None)
    predictions = design @ coefficients
    train_mse = float(np.mean((predictions - train_targets) ** 2))
    return (
        coefficients,
        np.asarray([*mean, *std], dtype=np.float64),
        {"n_coefficients": int(design.shape[1]), "train_mse": train_mse},
    )


def fit_polynomial_baseline(
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
) -> dict[str, Any]:
    """Alias for fit_polynomial_regression for a discoverable baseline API."""
    return fit_polynomial_regression(dataset_path)


def fit_polynomial_regression(
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
) -> dict[str, Any]:
    """Fit degrees 1-6 using train data and choose by validation MSE."""
    models: list[dict[str, Any]] = []
    with np.load(dataset_path) as data:
        mean = data["X_reduced_mean"].astype(np.float64)
        std = data["X_reduced_std"].astype(np.float64)
        train_x = data["X_reduced_train"].astype(np.float64)
        train_y = data["call_over_K_train"].astype(np.float64).reshape(-1)
        val_x = data["X_reduced_val"].astype(np.float64)
        val_y = data["call_over_K_val"].astype(np.float64).reshape(-1)

    val_normalized = (val_x - mean) / std
    for degree in range(1, 7):
        val_design = polynomial_features(val_normalized, degree)
        coefficients, _, info = fit_polynomial_degree(
            train_x,
            train_y,
            degree,
            mean,
            std,
        )
        val_predictions = val_design @ coefficients
        val_mse = float(np.mean((val_predictions - val_y) ** 2))
        models.append({
            "degree": degree,
            "n_coefficients": int(info["n_coefficients"]),
            "coefficients": coefficients,
            "validation_mse": val_mse,
        })

    chosen = min(models, key=lambda model: model["validation_mse"])
    return {
        "chosen_degree": int(chosen["degree"]),
        "chosen_coefficients": chosen["coefficients"],
        "degrees": models,
        "mean": mean,
        "std": std,
    }


def evaluate_polynomial(
    result: dict[str, Any],
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
) -> dict[str, Any]:
    """Evaluate the chosen polynomial with the same metrics as evaluate.py."""
    degree = int(result["chosen_degree"])
    coefficients = result["chosen_coefficients"]
    mean = result["mean"]
    std = result["std"]

    with np.load(dataset_path) as data:
        test_normalized = (
            data["X_reduced_test"].astype(np.float64) - mean
        ) / std
        true_call_over_k = data["call_over_K_test"].astype(np.float64).reshape(-1)
        test_full = data["X_test"].astype(np.float64)

    predicted_call_over_k = polynomial_features(test_normalized, degree) @ coefficients
    metrics = compute_metrics(predicted_call_over_k, true_call_over_k, test_full)
    metrics["degree"] = degree
    metrics["n_coefficients"] = int(polynomial_features(test_normalized, degree).shape[1])
    return metrics


def run_baseline(
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
    output_dir: str | Path = "results/baseline",
) -> dict[str, Any]:
    """Fit degrees 1-6, select on validation MSE, and save baseline artifacts."""
    fit = fit_polynomial_regression(dataset_path)
    metrics = evaluate_polynomial(fit, dataset_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output / "validation_mse_by_degree.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["degree", "n_coefficients", "validation_mse"],
        )
        writer.writeheader()
        for model in fit["degrees"]:
            writer.writerow({
                "degree": model["degree"],
                "n_coefficients": model["n_coefficients"],
                "validation_mse": model["validation_mse"],
            })
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_baseline(), indent=2))
