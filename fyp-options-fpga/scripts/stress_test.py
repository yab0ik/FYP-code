"""Stress-test the selected neural network on random and boundary inputs."""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optfpga.config import load_config
from optfpga.nn.evaluate import compute_metrics
from optfpga.nn.model import MLP
from optfpga.pricing.black_scholes import call_price

STRESS_SEED = 20_261_125
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _network_predict(
    run_dir: Path,
    reduced_inputs: np.ndarray,
    chunk_size: int = 65_536,
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
    normalized = ((reduced_inputs - mean) / std).astype(np.float32)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, normalized.shape[0], chunk_size):
            batch = torch.from_numpy(normalized[start : start + chunk_size])
            predictions.append(model(batch).numpy().reshape(-1).astype(np.float64))
    return np.concatenate(predictions)


def _full_and_reduced(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spot, strike, maturity, rate, sigma = inputs.T
    target_call_over_k = np.asarray(
        call_price(spot, strike, maturity, rate, sigma),
        dtype=np.float64,
    ) / strike
    reduced = np.column_stack((spot / strike, maturity, rate, sigma))
    return inputs, reduced, target_call_over_k


def _corner_and_edge_grid(bounds: dict[str, Any], points_per_edge: int = 33) -> np.ndarray:
    names = ("S", "K", "T", "r", "sigma")
    ranges = [bounds[name] for name in names]
    points = []
    for endpoints in itertools.product((0, 1), repeat=len(names)):
        points.append([ranges[index][endpoint] for index, endpoint in enumerate(endpoints)])
    for varying_axis in range(len(names)):
        fixed_axes = [index for index in range(len(names)) if index != varying_axis]
        for fixed_endpoints in itertools.product((0, 1), repeat=len(fixed_axes)):
            edge = np.zeros((points_per_edge, len(names)), dtype=np.float64)
            for index, endpoint in zip(fixed_axes, fixed_endpoints):
                edge[:, index] = ranges[index][endpoint]
            edge[:, varying_axis] = np.linspace(
                ranges[varying_axis][0],
                ranges[varying_axis][1],
                points_per_edge,
            )
            points.extend(edge)
    return np.unique(np.asarray(points, dtype=np.float64), axis=0)


def _evaluate_inputs(
    inputs: np.ndarray,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    full_inputs, reduced, true_call_over_k = _full_and_reduced(inputs)
    predicted = _network_predict(run_dir, reduced)
    metrics = compute_metrics(predicted, true_call_over_k, full_inputs)
    call_k_abs_error = np.abs(predicted - true_call_over_k)
    price_abs_error = call_k_abs_error * full_inputs[:, 1]
    percentiles = {
        "call_over_K_p99_9_abs_error": float(np.percentile(call_k_abs_error, 99.9)),
        "price_p99_9_abs_error": float(np.percentile(price_abs_error, 99.9)),
        "call_over_K_max_abs_error": float(np.max(call_k_abs_error)),
        "price_max_abs_error": float(np.max(price_abs_error)),
    }
    return metrics, percentiles


def run_stress_test(
    dataset_path: str | Path = "data/black_scholes_dataset.npz",
    config_path: str | Path = "configs/default.yaml",
    network_run: str | Path = "results/b2_selected_seed_42",
    output_path: str | Path = "results/b2_stress/metrics.json",
    seed: int = STRESS_SEED,
    sample_count: int = 1_000_000,
) -> dict[str, Any]:
    """Evaluate one million seeded random samples and the input-box edges."""
    config = load_config(config_path)
    network_config = load_config(Path(network_run) / "config.yaml")
    existing_seeds = {
        int(config.get("random_seed", 42)),
        int(network_config.get("nn", {}).get("seed", 42)),
    }
    metadata_path = Path(dataset_path).with_name("black_scholes_dataset_metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_seeds.add(int(metadata["seed"]))
    if seed in existing_seeds:
        raise ValueError(f"stress seed {seed} must differ from existing seeds {sorted(existing_seeds)}")

    ranges = config.get("input_ranges")
    if not isinstance(ranges, dict):
        raise ValueError("config must define input_ranges")
    names = ("S", "K", "T", "r", "sigma")
    rng = np.random.default_rng(seed)
    sampled = np.column_stack([
        rng.uniform(float(ranges[name][0]), float(ranges[name][1]), sample_count)
        for name in names
    ])
    edges = _corner_and_edge_grid(ranges)

    run_dir = Path(network_run)
    random_metrics, random_percentiles = _evaluate_inputs(sampled, run_dir)
    edge_metrics, edge_percentiles = _evaluate_inputs(edges, run_dir)
    result = {
        "network_run": str(run_dir),
        "random_seed": seed,
        "random_sample_count": int(sample_count),
        "corner_edge_sample_count": int(edges.shape[0]),
        "random_samples": {
            "metrics": random_metrics,
            "absolute_error_summary": random_percentiles,
        },
        "corners_and_edges": {
            "metrics": edge_metrics,
            "absolute_error_summary": edge_percentiles,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/black_scholes_dataset.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--network-run", type=Path, default=Path("results/b2_selected_seed_42"))
    parser.add_argument("--output", type=Path, default=Path("results/b2_stress/metrics.json"))
    parser.add_argument("--seed", type=int, default=STRESS_SEED)
    parser.add_argument("--samples", type=int, default=1_000_000)
    args = parser.parse_args()
    run_stress_test(
        _project_path(args.dataset),
        _project_path(args.config),
        _project_path(args.network_run),
        _project_path(args.output),
        args.seed,
        args.samples,
    )


if __name__ == "__main__":
    main()
