"""Generate reproducible train, validation, and test datasets for Black-Scholes pricing."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray

from optfpga.config import load_config
from optfpga.pricing.black_scholes import call_price, put_price

FEATURE_NAMES: tuple[str, ...] = ("S", "K", "T", "r", "sigma")
REDUCED_FEATURE_NAMES: tuple[str, ...] = ("moneyness", "T", "r", "sigma")


@dataclass(frozen=True)
class ParameterBounds:
    """Sampling bounds loaded from the project configuration."""

    S_min: float
    S_max: float
    K_min: float
    K_max: float
    T_min: float
    T_max: float
    r_min: float
    r_max: float
    sigma_min: float
    sigma_max: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ParameterBounds":
        ranges = config.get("input_ranges")
        if not isinstance(ranges, dict):
            raise ValueError("Configuration must contain input_ranges")
        values: dict[str, float] = {}
        for name in FEATURE_NAMES:
            bounds = ranges.get(name)
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"input_ranges.{name} must contain two values")
            minimum, maximum = (float(value) for value in bounds)
            if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
                raise ValueError(f"{name} minimum must be less than maximum")
            values[f"{name}_min"] = minimum
            values[f"{name}_max"] = maximum
        return cls(**values)

    @classmethod
    def from_config_file(cls, config_path: str | Path | None = None) -> "ParameterBounds":
        return cls.from_config(load_config(config_path))

    def as_dict(self) -> dict[str, list[float]]:
        return {
            name: [getattr(self, f"{name}_min"), getattr(self, f"{name}_max")]
            for name in FEATURE_NAMES
        }


def _config_value(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sample_features(
    n_samples: int,
    rng: np.random.Generator,
    bounds: ParameterBounds,
) -> NDArray[np.float64]:
    """Draw uniform random samples of (S, K, T, r, sigma)."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return np.column_stack(
        [
            rng.uniform(bounds.S_min, bounds.S_max, n_samples),
            rng.uniform(bounds.K_min, bounds.K_max, n_samples),
            rng.uniform(bounds.T_min, bounds.T_max, n_samples),
            rng.uniform(bounds.r_min, bounds.r_max, n_samples),
            rng.uniform(bounds.sigma_min, bounds.sigma_max, n_samples),
        ]
    )


def compute_labels(
    features: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute exact Black-Scholes call and put prices for an (N, 5) matrix."""
    S, K, T, r, sigma = features.T
    return (
        np.asarray(call_price(S, K, T, r, sigma), dtype=np.float64),
        np.asarray(put_price(S, K, T, r, sigma), dtype=np.float64),
    )


def _split_arrays(
    features: NDArray[np.float64],
    calls: NDArray[np.float64],
    puts: NDArray[np.float64],
    name: str,
) -> dict[str, NDArray[np.float64]]:
    S, K, _, _, _ = features.T
    moneyness = S / K
    return {
        f"X_{name}": features,
        f"y_call_{name}": calls,
        f"y_put_{name}": puts,
        f"moneyness_{name}": moneyness,
        f"call_over_K_{name}": calls / K,
        f"put_over_K_{name}": puts / K,
        f"X_reduced_{name}": np.column_stack([moneyness, features[:, 2], features[:, 3], features[:, 4]]),
    }


def generate_and_save_dataset(
    output_path: str | Path = "data/black_scholes_dataset.npz",
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    seed: int | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Generate configured splits, labels, derived data, and metadata."""
    config = load_config(config_path)
    bounds = ParameterBounds.from_config(config)
    sizes = {
        "train": _config_value(config, "n_train") if n_train is None else n_train,
        "val": _config_value(config, "n_val") if n_val is None else n_val,
        "test": _config_value(config, "n_test") if n_test is None else n_test,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in sizes.values()):
        raise ValueError("split sizes must be positive integers")
    base_seed = config.get("random_seed") if seed is None else seed
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("seed must be an integer")
    split_names = ("train", "val", "test")
    split_sequences = dict(zip(split_names, np.random.SeedSequence(base_seed).spawn(3)))

    arrays: dict[str, NDArray[np.float64]] = {}
    for split in split_names:
        features = sample_features(
            sizes[split], np.random.default_rng(split_sequences[split]), bounds
        )
        calls, puts = compute_labels(features)
        arrays.update(_split_arrays(features, calls, puts, split))

    training_reduced = arrays["X_reduced_train"]
    arrays["X_reduced_mean"] = training_reduced.mean(axis=0)
    arrays["X_reduced_std"] = training_reduced.std(axis=0)
    arrays["X_mean"] = arrays["X_train"].mean(axis=0)
    arrays["X_std"] = arrays["X_train"].std(axis=0)
    arrays["X_min"] = np.array([bounds.S_min, bounds.K_min, bounds.T_min, bounds.r_min, bounds.sigma_min])
    arrays["X_max"] = np.array([bounds.S_max, bounds.K_max, bounds.T_max, bounds.r_max, bounds.sigma_max])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feature_names=np.array(FEATURE_NAMES), reduced_feature_names=np.array(REDUCED_FEATURE_NAMES), **arrays)
    metadata = {
        "bounds": bounds.as_dict(),
        "seed": base_seed,
        "split_seeds": {
            split: list(split_sequences[split].spawn_key) for split in split_names
        },
        "split_sizes": sizes,
        "columns": {
            "features": list(FEATURE_NAMES),
            "reduced_features": list(REDUCED_FEATURE_NAMES),
            "derived": ["moneyness", "call_over_K", "put_over_K"],
        },
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out.with_name(f"{out.stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    save_test_split_preview(
        out.with_name(f"{out.stem}_test_split_preview.csv"),
        arrays["X_test"],
        arrays["y_call_test"],
        arrays["y_put_test"],
    )
    return out


def save_test_split_preview(
    output_path: str | Path,
    features: NDArray[np.float64],
    call_prices: NDArray[np.float64],
    put_prices: NDArray[np.float64],
) -> Path:
    """Save test features and prices as an Excel-friendly CSV file."""
    out = Path(output_path)
    np.savetxt(
        out,
        np.column_stack([features, call_prices, put_prices]),
        delimiter=",",
        header="S,K,T,r,sigma,call_price,put_price",
        comments="",
        fmt="%.6f",
    )
    return out


if __name__ == "__main__":
    saved_file = generate_and_save_dataset()
    with np.load(saved_file) as data:
        print(f"Saved dataset to {saved_file}")
        for name in FEATURE_NAMES:
            index = FEATURE_NAMES.index(name)
            print(f"{name}: min={data['X_train'][:, index].min():.8f}, max={data['X_train'][:, index].max():.8f}")
        calls = data["y_call_train"]
        print(f"Call prices below 0.01: {np.mean(calls < 0.01):.8f}")
        print(f"Call prices below 0.1: {np.mean(calls < 0.1):.8f}")
