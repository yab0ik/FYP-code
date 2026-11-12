"""Tests for Black-Scholes dataset generation."""

from pathlib import Path

import numpy as np

from optfpga.data.generate_dataset import ParameterBounds, generate_and_save_dataset


def test_dataset_shapes_bounds_and_parity(tmp_path: Path) -> None:
    out_file = tmp_path / "test_bs.npz"
    bounds = ParameterBounds()
    generate_and_save_dataset(
        output_path=out_file,
        n_train=500,
        n_val=100,
        n_test=100,
        seed=123,
        bounds=bounds,
    )

    data = np.load(out_file)
    assert data["X_train"].shape == (500, 5)
    assert data["X_val"].shape == (100, 5)
    assert data["X_test"].shape == (100, 5)
    assert data["y_call_test"].shape == (100,)
    assert data["y_put_test"].shape == (100,)

    preview = np.loadtxt(
        out_file.with_name("test_split_preview.csv"),
        delimiter=",",
        skiprows=1,
    )
    assert preview.shape == (100, 7)
    np.testing.assert_allclose(preview, np.column_stack([
        data["X_test"],
        data["y_call_test"],
        data["y_put_test"],
    ]), atol=0.5e-6)

    assert np.all(data["X_train"] >= data["X_min"])
    assert np.all(data["X_train"] <= data["X_max"])

    S, K, T, r, _ = data["X_test"].T
    np.testing.assert_allclose(
        data["y_call_test"] - data["y_put_test"],
        S - K * np.exp(-r * T),
        atol=1e-9,
    )


def test_dataset_reproducibility(tmp_path: Path) -> None:
    file_a = tmp_path / "a.npz"
    file_b = tmp_path / "b.npz"
    generate_and_save_dataset(file_a, n_train=200, n_val=50, n_test=50, seed=99)
    generate_and_save_dataset(file_b, n_train=200, n_val=50, n_test=50, seed=99)

    data_a = np.load(file_a)
    data_b = np.load(file_b)
    np.testing.assert_array_equal(data_a["X_train"], data_b["X_train"])
    np.testing.assert_array_equal(data_a["y_call_train"], data_b["y_call_train"])
