"""Generate reproducible train/validation/test datasets for Black-Scholes option pricing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from optfpga.pricing.black_scholes import call_price, put_price

FEATURE_NAMES: tuple[str, ...] = ("S", "K", "T", "r", "sigma")


@dataclass(frozen=True)
class ParameterBounds:
    """Sampling bounds for Black-Scholes input parameters."""

    S_min: float = 50.0
    S_max: float = 150.0
    K_min: float = 50.0
    K_max: float = 150.0
    T_min: float = 0.05
    T_max: float = 2.0
    r_min: float = 0.0
    r_max: float = 0.10
    sigma_min: float = 0.05
    sigma_max: float = 0.60


def sample_features(
    n_samples: int,
    rng: np.random.Generator,
    bounds: ParameterBounds = ParameterBounds(),
) -> NDArray[np.float64]:
    """Draw uniform random samples of (S, K, T, r, sigma)."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    S = rng.uniform(bounds.S_min, bounds.S_max, n_samples)
    K = rng.uniform(bounds.K_min, bounds.K_max, n_samples)
    T = rng.uniform(bounds.T_min, bounds.T_max, n_samples)
    r = rng.uniform(bounds.r_min, bounds.r_max, n_samples)
    sigma = rng.uniform(bounds.sigma_min, bounds.sigma_max, n_samples)

    return np.column_stack([S, K, T, r, sigma])


def compute_labels(
    features: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute exact Black-Scholes call and put prices for an (N, 5) feature matrix."""
    S, K, T, r, sigma = features.T
    calls = np.asarray(call_price(S, K, T, r, sigma), dtype=np.float64)
    puts = np.asarray(put_price(S, K, T, r, sigma), dtype=np.float64)
    return calls, puts


def save_test_split_preview(
    output_path: str | Path,
    features: NDArray[np.float64],
    call_prices: NDArray[np.float64],
    put_prices: NDArray[np.float64],
) -> Path:
    """Save test features and prices as an Excel-friendly CSV file."""
    out = Path(output_path)
    table = np.column_stack([features, call_prices, put_prices])
    np.savetxt(
        out,
        table,
        delimiter=",",
        header="S,K,T,r,sigma,call_price,put_price",
        comments="",
        fmt="%.6f",
    )
    return out


def generate_and_save_dataset(
    output_path: str | Path = "data/black_scholes_dataset.npz",
    n_train: int = 80_000,
    n_val: int = 10_000,
    n_test: int = 10_000,
    seed: int = 42,
    bounds: ParameterBounds = ParameterBounds(),
) -> Path:
    """Generate train, validation, and test splits and save to a compressed .npz archive."""
    rng = np.random.default_rng(seed)

    X_train = sample_features(n_train, rng, bounds)
    X_val = sample_features(n_val, rng, bounds)
    X_test = sample_features(n_test, rng, bounds)

    y_call_train, y_put_train = compute_labels(X_train)
    y_call_val, y_put_val = compute_labels(X_val)
    y_call_test, y_put_test = compute_labels(X_test)

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_min = np.array(
        [bounds.S_min, bounds.K_min, bounds.T_min, bounds.r_min, bounds.sigma_min]
    )
    X_max = np.array(
        [bounds.S_max, bounds.K_max, bounds.T_max, bounds.r_max, bounds.sigma_max]
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        feature_names=np.array(FEATURE_NAMES),
        seed=np.array(seed),
        X_train=X_train,
        y_call_train=y_call_train,
        y_put_train=y_put_train,
        X_val=X_val,
        y_call_val=y_call_val,
        y_put_val=y_put_val,
        X_test=X_test,
        y_call_test=y_call_test,
        y_put_test=y_put_test,
        X_mean=X_mean,
        X_std=X_std,
        X_min=X_min,
        X_max=X_max,
    )
    save_test_split_preview(
        out.with_name("test_split_preview.csv"),
        X_test,
        y_call_test,
        y_put_test,
    )
    return out


if __name__ == "__main__":
    saved_file = generate_and_save_dataset()
    data = np.load(saved_file)
    print(f"Saved dataset to {saved_file}")
    print(f"Saved Excel preview to {saved_file.with_name('test_split_preview.csv')}")
    print(
        f"Train samples: {len(data['X_train'])}, "
        f"Val: {len(data['X_val'])}, Test: {len(data['X_test'])}"
    )
    print(
        f"Call price range (train): "
        f"[{data['y_call_train'].min():.4f}, {data['y_call_train'].max():.4f}]"
    )
    print(
        f"Put price range  (train): "
        f"[{data['y_put_train'].min():.4f}, {data['y_put_train'].max():.4f}]"
    )
