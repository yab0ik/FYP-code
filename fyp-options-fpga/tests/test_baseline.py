import numpy as np
from math import comb

from optfpga.nn.baseline import (
    fit_polynomial_degree,
    fit_polynomial_regression,
    polynomial_features,
)
from optfpga.nn.evaluate import compute_metrics


def test_polynomial_fit_recovers_exact_target() -> None:
    rng = np.random.default_rng(123)
    train_x = rng.normal(size=(200, 4))
    target_coefficients = np.array([0.2, -0.3, 0.1, 0.05, 0.07])
    train_y = (
        target_coefficients[0]
        + target_coefficients[1] * train_x[:, 0]
        + target_coefficients[2] * train_x[:, 1]
        + target_coefficients[3] * train_x[:, 2]
        + target_coefficients[4] * train_x[:, 3]
    )
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    coefficients, _, info = fit_polynomial_degree(
        train_x,
        train_y,
        1,
        mean,
        std,
    )
    normalized = (train_x - mean) / std
    predictions = polynomial_features(normalized, 1) @ coefficients
    np.testing.assert_allclose(predictions, train_y, rtol=1e-8, atol=1e-8)
    assert info["n_coefficients"] == 5


def test_degree_one_equals_normal_equation() -> None:
    rng = np.random.default_rng(456)
    train_x = rng.normal(size=(100, 4))
    train_y = 0.4 + train_x @ np.array([0.2, -0.1, 0.3, 0.05])
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    coefficients, _, _ = fit_polynomial_degree(
        train_x,
        train_y,
        1,
        mean,
        std,
    )
    normalized = (train_x - mean) / std
    design = polynomial_features(normalized, 1)
    normal_solution = np.linalg.inv(design.T @ design) @ design.T @ train_y
    np.testing.assert_allclose(coefficients, normal_solution, rtol=1e-10, atol=1e-10)


def test_test_split_is_not_used_for_degree_selection(tmp_path) -> None:
    rng = np.random.default_rng(789)
    train_x = rng.normal(size=(80, 4))
    val_x = rng.normal(size=(20, 4))
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    np.savez(
        tmp_path / "train_val_only.npz",
        X_reduced_train=train_x,
        X_reduced_val=val_x,
        X_reduced_mean=mean,
        X_reduced_std=std,
        call_over_K_train=0.1 + 0.2 * train_x[:, 0],
        call_over_K_val=0.1 + 0.2 * val_x[:, 0],
    )

    result = fit_polynomial_regression(tmp_path / "train_val_only.npz")
    assert 1 <= result["chosen_degree"] <= 8
    for row in result["degrees"]:
        assert row["n_coefficients"] == comb(4 + row["degree"], row["degree"])


def test_exact_degree_two_polynomial_selects_degree_at_least_two(tmp_path) -> None:
    rng = np.random.default_rng(2024)
    train_x = rng.normal(size=(300, 4))
    val_x = rng.normal(size=(80, 4))
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)

    train_y = (
        0.15
        + 0.3 * train_x[:, 0]
        - 0.2 * train_x[:, 1]
        + 0.25 * train_x[:, 0] ** 2
        - 0.1 * train_x[:, 0] * train_x[:, 1]
        + 0.15 * train_x[:, 2] ** 2
    )
    val_y = (
        0.15
        + 0.3 * val_x[:, 0]
        - 0.2 * val_x[:, 1]
        + 0.25 * val_x[:, 0] ** 2
        - 0.1 * val_x[:, 0] * val_x[:, 1]
        + 0.15 * val_x[:, 2] ** 2
    )
    np.savez(
        tmp_path / "degree_two.npz",
        X_reduced_train=train_x,
        X_reduced_val=val_x,
        X_reduced_mean=mean,
        X_reduced_std=std,
        call_over_K_train=train_y,
        call_over_K_val=val_y,
    )

    result = fit_polynomial_regression(tmp_path / "degree_two.npz")
    assert result["chosen_degree"] >= 2
    degree_rows = {row["degree"]: row for row in result["degrees"]}
    assert degree_rows[2]["validation_mse"] < 1e-8
    assert len(degree_rows) == 8
    for degree, row in degree_rows.items():
        assert row["n_coefficients"] == comb(4 + degree, degree)


def test_compute_metrics_perfect_predictions_and_identical_paths() -> None:
    predicted = np.array([0.1, 0.2, 0.3])
    true = predicted.copy()
    full_inputs = np.array([
        [0.8, 100.0, 1.0, 0.05, 0.2],
        [1.0, 100.0, 1.0, 0.05, 0.3],
        [1.2, 100.0, 1.0, 0.05, 0.4],
    ])

    network_metrics = compute_metrics(predicted, true, full_inputs)
    baseline_metrics = compute_metrics(predicted, true, full_inputs)

    assert network_metrics["overall"]["call_over_K"]["mae"] == 0.0
    assert network_metrics["overall"]["call_over_K"]["rmse"] == 0.0
    assert network_metrics["overall"]["call_over_K"]["max_abs_error"] == 0.0
    assert network_metrics["overall"]["price"]["mae"] == 0.0
    assert network_metrics["overall"]["price"]["rmse"] == 0.0
    assert network_metrics["overall"]["price"]["max_abs_error"] == 0.0

    def assert_metrics_equal(left, right) -> None:
        assert left.keys() == right.keys()
        for key in left:
            left_value = left[key]
            right_value = right[key]
            if isinstance(left_value, dict):
                assert_metrics_equal(left_value, right_value)
            else:
                if isinstance(left_value, float) and np.isnan(left_value):
                    assert np.isnan(right_value)
                else:
                    assert left_value == right_value

    assert_metrics_equal(network_metrics, baseline_metrics)


def test_compute_metrics_empty_relative_error_is_nan() -> None:
    predicted = np.array([0.0005, 0.0008])
    true = np.array([0.0005, 0.0008])
    full_inputs = np.array([
        [0.8, 100.0, 1.0, 0.05, 0.2],
        [1.0, 100.0, 1.0, 0.05, 0.3],
    ])

    metrics = compute_metrics(predicted, true, full_inputs)
    relative = metrics["relative_error_price_above_0.1"]
    assert relative["count"] == 0
    assert np.isnan(relative["mae"])
    assert np.isnan(relative["rmse"])
    assert np.isnan(relative["max_abs_error"])


def test_compute_metrics_hand_calculated_errors_and_bin_edges() -> None:
    true = np.array([0.20, 0.20, 0.20, 0.20])
    predicted = np.array([0.21, 0.18, 0.23, 0.16])
    full_inputs = np.array([
        [89.9, 100.0, 1.0, 0.05, 0.2],
        [90.0, 100.0, 1.0, 0.05, 0.2],
        [110.0, 100.0, 1.0, 0.05, 0.2],
        [110.1, 100.0, 1.0, 0.05, 0.2],
    ])

    metrics = compute_metrics(predicted, true, full_inputs)

    overall = metrics["overall"]["call_over_K"]
    np.testing.assert_allclose(overall["mae"], 0.025, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(overall["rmse"], np.sqrt(0.00075), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(overall["max_abs_error"], 0.04, rtol=0.0, atol=1e-12)

    below = metrics["moneyness_bins"]["below_0.9"]["call_over_K"]
    middle = metrics["moneyness_bins"]["0.9_to_1.1"]["call_over_K"]
    above = metrics["moneyness_bins"]["above_1.1"]["call_over_K"]
    assert below["count"] == 1
    assert middle["count"] == 2
    assert above["count"] == 1

    np.testing.assert_allclose(below["mae"], 0.01, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(middle["mae"], 0.025, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(above["mae"], 0.04, rtol=0.0, atol=1e-12)
