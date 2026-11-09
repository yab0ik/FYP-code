#Tests for the Black-Scholes reference implementation.

import numpy as np
import pytest

from optfpga.pricing.black_scholes import call_price, put_price


def test_known_value() -> None:
    assert call_price(100, 100, 1, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-3)
    assert put_price(100, 100, 1, 0.05, 0.2) == pytest.approx(5.5735, abs=1e-3)


def test_hull_example() -> None:
    assert call_price(42, 40, 0.5, 0.1, 0.2) == pytest.approx(4.76, abs=1e-2)
    assert put_price(42, 40, 0.5, 0.1, 0.2) == pytest.approx(0.81, abs=1e-2)


def test_sigma_point_three() -> None:
    assert call_price(100, 100, 1, 0.05, 0.3) == pytest.approx(14.2313, abs=1e-3)
    assert put_price(100, 100, 1, 0.05, 0.3) == pytest.approx(9.3542, abs=1e-3)


def test_deep_in_the_money_call() -> None:
    assert call_price(150, 100, 1, 0.05, 0.2) == pytest.approx(54.9701, abs=1e-3)


def test_put_call_parity_on_random_grid() -> None:
    rng = np.random.default_rng(42)
    S = rng.uniform(50, 150, 100)
    K = rng.uniform(50, 150, 100)
    T = rng.uniform(0.1, 2, 100)
    r = rng.uniform(0, 0.1, 100)
    sigma = rng.uniform(0.05, 0.6, 100)
    np.testing.assert_allclose(
        call_price(S, K, T, r, sigma) - put_price(S, K, T, r, sigma),
        S - K * np.exp(-r * T),
        atol=1e-9,
    )


def test_vectorised_call_matches_scalar_results() -> None:
    inputs = (
        np.array([80.0, 100.0, 120.0]),
        np.array([90.0, 100.0, 110.0]),
        np.array([0.5, 1.0, 1.5]),
        np.array([0.01, 0.05, 0.08]),
        np.array([0.15, 0.2, 0.35]),
    )
    vectorised = call_price(*inputs)
    scalar = np.array([call_price(*(values[index] for values in inputs)) for index in range(3)])
    np.testing.assert_allclose(vectorised, scalar)


def test_call_price_monotonicity() -> None:
    assert call_price(100, 100, 1, 0.05, 0.3) > call_price(100, 100, 1, 0.05, 0.2)
    assert call_price(110, 100, 1, 0.05, 0.2) > call_price(100, 100, 1, 0.05, 0.2)


def test_put_price_monotonicity() -> None:
    assert put_price(100, 100, 1, 0.05, 0.3) > put_price(100, 100, 1, 0.05, 0.2)
    assert put_price(100, 110, 1, 0.05, 0.2) > put_price(100, 100, 1, 0.05, 0.2)


@pytest.mark.parametrize("name,index", [("S", 0), ("K", 1), ("T", 2), ("sigma", 4)])
def test_invalid_inputs_raise_value_error(name: str, index: int) -> None:
    values = [100.0, 100.0, 1.0, 0.05, 0.2]
    values[index] = 0.0
    with pytest.raises(ValueError, match=name):
        call_price(*values)


def test_nan_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="S"):
        call_price(np.nan, 100, 1, 0.05, 0.2)


def test_matches_py_vollib_if_installed() -> None:
    py_vollib = pytest.importorskip("py_vollib.black_scholes")
    flag = "c"
    expected = py_vollib.black_scholes(flag, 100, 100, 1, 0.05, 0.2)
    assert call_price(100, 100, 1, 0.05, 0.2) == pytest.approx(expected, abs=1e-9)
