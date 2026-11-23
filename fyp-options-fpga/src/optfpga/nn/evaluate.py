"""Test-set evaluation for the baseline neural network."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optfpga.nn.model import MLP


def _metrics(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "max_abs_error": 0.0, "count": 0.0}
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "count": float(errors.size),
    }


def _finite_metrics(errors: np.ndarray) -> dict[str, float]:
    result = _metrics(errors)
    return {key: float(value) for key, value in result.items()}


def evaluate_test(
    checkpoint_path: str | Path,
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
    normalization_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a trained checkpoint on the test split only."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    run_dir = checkpoint_path.parent

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = MLP(
        checkpoint["input_size"],
        checkpoint["hidden_sizes"],
        checkpoint["activation"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if normalization_path is None:
        normalization_path = run_dir / "normalization.npz"
    if Path(normalization_path).exists():
        with np.load(normalization_path) as normal:
            mean = normal["X_reduced_mean"].astype(np.float32)
            std = normal["X_reduced_std"].astype(np.float32)
    else:
        with np.load(dataset_path) as data:
            mean = data["X_reduced_mean"].astype(np.float32)
            std = data["X_reduced_std"].astype(np.float32)

    with np.load(dataset_path) as data:
        test_x = ((data["X_reduced_test"] - mean) / std).astype(np.float32)
        true_call_over_k = data["call_over_K_test"].reshape(-1).astype(np.float64)
        test_full = data["X_test"].astype(np.float64)

    with torch.no_grad():
        predicted_call_over_k = model(torch.from_numpy(test_x)).numpy().reshape(-1).astype(np.float64)

    call_over_k_error = predicted_call_over_k - true_call_over_k
    strike = test_full[:, 1]
    true_price = true_call_over_k * strike
    predicted_price = predicted_call_over_k * strike
    price_error = predicted_price - true_price

    moneyness = test_full[:, 0] / strike
    moneyness_bins = {
        "below_0.9": moneyness < 0.9,
        "0.9_to_1.1": (moneyness >= 0.9) & (moneyness <= 1.1),
        "above_1.1": moneyness > 1.1,
    }

    sigma = test_full[:, 4]
    maturity = test_full[:, 2]
    volatility_time = sigma * np.sqrt(maturity)
    tercile_edges = np.quantile(volatility_time, [1.0 / 3.0, 2.0 / 3.0])
    volatility_terciles = {
        "tercile_1": volatility_time <= tercile_edges[0],
        "tercile_2": (volatility_time > tercile_edges[0]) & (volatility_time <= tercile_edges[1]),
        "tercile_3": volatility_time > tercile_edges[1],
    }

    def unit_metrics(mask: np.ndarray) -> dict[str, dict[str, float]]:
        return {
            "call_over_K": _finite_metrics(call_over_k_error[mask]),
            "price": _finite_metrics(price_error[mask]),
        }

    moneyness_metrics = {
        name: unit_metrics(mask)
        for name, mask in moneyness_bins.items()
    }
    volatility_metrics = {
        name: unit_metrics(mask)
        for name, mask in volatility_terciles.items()
    }

    price_abs_error = np.abs(price_error)
    low_price_mask = true_price < 0.1
    reliable_price_mask = true_price > 0.1
    low_price_metrics = _finite_metrics(price_error[low_price_mask])
    relative_error = (
        price_abs_error[reliable_price_mask] / np.abs(true_price[reliable_price_mask])
        if np.any(reliable_price_mask)
        else np.array([], dtype=np.float64)
    )

    result: dict[str, Any] = {
        "split": "test",
        "sample_count": int(true_price.size),
        "overall": unit_metrics(np.ones(true_price.size, dtype=bool)),
        "moneyness_bins": moneyness_metrics,
        "sigma_sqrt_T_terciles": volatility_metrics,
        "sigma_sqrt_T_tercile_edges": {
            "lower": float(tercile_edges[0]),
            "upper": float(tercile_edges[1]),
        },
        "fraction_price_error_above_0.01": float(np.mean(price_abs_error > 0.01)),
        "fraction_price_error_above_0.1": float(np.mean(price_abs_error > 0.1)),
        "true_price_below_0.1": {
            "mae": low_price_metrics["mae"],
            "rmse": low_price_metrics["rmse"],
            "max_abs_error": low_price_metrics["max_abs_error"],
            "count": low_price_metrics["count"],
        },
        "relative_error_price_above_0.1": {
            "mae": float(np.mean(relative_error)) if relative_error.size else 0.0,
            "rmse": float(np.sqrt(np.mean(relative_error ** 2))) if relative_error.size else 0.0,
            "max_abs_error": float(np.max(relative_error)) if relative_error.size else 0.0,
            "count": float(relative_error.size),
        },
    }

    overall_call = result["overall"]["call_over_K"]
    overall_price = result["overall"]["price"]
    result.update({
        "mae_call_over_K": overall_call["mae"],
        "rmse_call_over_K": overall_call["rmse"],
        "max_abs_error_call_over_K": overall_call["max_abs_error"],
        "mae_price": overall_price["mae"],
        "rmse_price": overall_price["rmse"],
        "max_abs_error_price": overall_price["max_abs_error"],
    })
    return result


def evaluate(
    checkpoint_path: str | Path,
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate test data and optionally write metrics JSON."""
    checkpoint_path = Path(checkpoint_path)
    result = evaluate_test(checkpoint_path, dataset_path)
    if output_path is None:
        output_path = checkpoint_path.parent / "metrics.json" if checkpoint_path.is_file() else checkpoint_path / "metrics.json"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_summary(result)
    return result


def print_summary(result: dict[str, Any]) -> None:
    print("Test split metrics")
    print(f"  Samples: {result['sample_count']}")
    print(
        "  Call/K: "
        f"MAE={result['overall']['call_over_K']['mae']:.8f}, "
        f"RMSE={result['overall']['call_over_K']['rmse']:.8f}, "
        f"max={result['overall']['call_over_K']['max_abs_error']:.8f}"
    )
    print(
        "  Price: "
        f"MAE={result['overall']['price']['mae']:.8f}, "
        f"RMSE={result['overall']['price']['rmse']:.8f}, "
        f"max={result['overall']['price']['max_abs_error']:.8f}"
    )
    print(f"  Price error > 0.01: {result['fraction_price_error_above_0.01']:.6f}")
    print(f"  Price error > 0.1: {result['fraction_price_error_above_0.1']:.6f}")
    for name, metrics in result["moneyness_bins"].items():
        print(f"  Moneyness {name}: call/K MAE={metrics['call_over_K']['mae']:.8f}, price MAE={metrics['price']['mae']:.8f}")
    for name, metrics in result["sigma_sqrt_T_terciles"].items():
        print(f"  {name}: call/K MAE={metrics['call_over_K']['mae']:.8f}, price MAE={metrics['price']['mae']:.8f}")
    print(
        "  True price below 0.1: "
        f"MAE={result['true_price_below_0.1']['mae']:.8f}, "
        f"RMSE={result['true_price_below_0.1']['rmse']:.8f}"
    )
    print(
        "  Relative error (true price > 0.1): "
        f"MAE={result['relative_error_price_above_0.1']['mae']:.8f}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/black_scholes_dataset.npz"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    evaluate(args.run, args.dataset, args.output)


if __name__ == "__main__":
    main()
