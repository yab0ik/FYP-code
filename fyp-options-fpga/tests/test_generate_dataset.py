"""Tests for configured Black-Scholes dataset generation."""

import json
from pathlib import Path

import numpy as np
import pytest

from optfpga.config import load_config
from optfpga.data.generate_dataset import ParameterBounds, generate_and_save_dataset


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def test_dataset_config_bounds_and_invariants(tmp_path: Path) -> None:
    output = generate_and_save_dataset(tmp_path / "dataset.npz", n_train=50, n_val=30, n_test=20, seed=123)
    data = _load(output)
    bounds = ParameterBounds.from_config(load_config())
    for array in data.values():
        if np.issubdtype(array.dtype, np.number):
            assert np.all(np.isfinite(array))
    for split, size in (("train", 50), ("val", 30), ("test", 20)):
        features = data[f"X_{split}"]
        assert features.shape == (size, 5)
        for index, name in enumerate(("S", "K", "T", "r", "sigma")):
            assert np.all((features[:, index] >= getattr(bounds, f"{name}_min")) & (features[:, index] <= getattr(bounds, f"{name}_max")))
        for key in (f"y_call_{split}", f"y_put_{split}", f"moneyness_{split}", f"call_over_K_{split}", f"put_over_K_{split}"):
            assert np.all(np.isfinite(data[key]))
        S, K, T, r, _ = features.T
        assert np.all(np.maximum(S - K * np.exp(-r * T), 0) <= data[f"y_call_{split}"])
        assert np.all(data[f"y_call_{split}"] <= S)
        np.testing.assert_allclose(data[f"moneyness_{split}"], S / K, atol=1e-12)
        np.testing.assert_allclose(data[f"call_over_K_{split}"], data[f"y_call_{split}"] / K, atol=1e-12)
        np.testing.assert_allclose(data[f"put_over_K_{split}"], data[f"y_put_{split}"] / K, atol=1e-12)

    metadata = json.loads(
        (output.parent / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["seed"] == 123
    assert metadata["split_seeds"] == {
        "train": [0],
        "val": [1],
        "test": [2],
    }
    assert metadata["split_sizes"] == {"train": 50, "val": 30, "test": 20}
    assert metadata["bounds"] == bounds.as_dict()
    assert metadata["columns"]["features"] == ["S", "K", "T", "r", "sigma"]
    assert metadata["columns"]["reduced_features"] == ["moneyness", "T", "r", "sigma"]
    assert metadata["columns"]["derived"] == ["moneyness", "call_over_K", "put_over_K"]
    np.testing.assert_allclose(data["X_reduced_mean"], data["X_reduced_train"].mean(axis=0))
    np.testing.assert_allclose(data["X_reduced_std"], data["X_reduced_train"].std(axis=0))


def test_reproducibility_and_independent_splits(tmp_path: Path) -> None:
    first = generate_and_save_dataset(tmp_path / "first.npz", n_train=20, n_val=10, n_test=10, seed=99)
    second = generate_and_save_dataset(tmp_path / "second.npz", n_train=20, n_val=10, n_test=10, seed=99)
    different = generate_and_save_dataset(tmp_path / "different.npz", n_train=20, n_val=10, n_test=10, seed=100)
    changed_train = generate_and_save_dataset(tmp_path / "changed.npz", n_train=25, n_val=10, n_test=10, seed=99)
    a, b, c, d = (_load(path) for path in (first, second, different, changed_train))
    for split in ("train", "val", "test"):
        for label in ("y_call", "y_put"):
            np.testing.assert_array_equal(a[f"{label}_{split}"], b[f"{label}_{split}"])
        np.testing.assert_array_equal(a[f"X_{split}"], b[f"X_{split}"])
    assert not np.array_equal(a["X_train"], c["X_train"])
    assert not np.array_equal(a["X_val"], c["X_val"])
    assert not np.array_equal(a["X_test"], c["X_test"])
    np.testing.assert_array_equal(a["X_val"], d["X_val"])
    np.testing.assert_array_equal(a["X_test"], d["X_test"])


@pytest.mark.parametrize("bad_range", ["S: [150, 50]", "T: [2, 0.1]", "sigma: [0.6, 0.05]"])
def test_bad_config_bounds_raise_value_error(tmp_path: Path, bad_range: str) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        f"input_ranges:\n  S: [50, 150]\n  K: [50, 150]\n  T: [0.1, 2]\n  r: [0, 0.1]\n  sigma: [0.05, 0.6]\n  {bad_range}\n"
        "random_seed: 42\nn_train: 1\nn_val: 1\nn_test: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="minimum must be less than maximum"):
        generate_and_save_dataset(tmp_path / "bad.npz", config_path=config)
