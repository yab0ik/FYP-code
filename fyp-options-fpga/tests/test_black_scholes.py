#Tests for the Black-Scholes reference implementation.


import numpy as np
import pytest

from optfpga.pricing.black_scholes import call_price, put_price

#John Hull's Options, Futures, and Other Derivatives example where $S=42, K=40, T=0.5, r=0.1, \sigma=0.2 \rightarrow C \approx 4.76, P \approx 0.81$) 

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

# Generates 100 random option parameter sets using a seeded random generator (default_rng(42)) and verifies that put-call parity holds to within $10^{-9}$
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

#Confirms basic financial monotonicity: higher volatility ($\sigma$) increases both call and put prices; higher spot price ($S$) increases call value; higher strike price ($K$) increases put value.
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


#Uses @pytest.mark.parametrize to confirm that 0.0, NaN, or inf in restricted parameters trigger the expected ValueError messages, while 0.0 and -0.01 interest rates execute cleanly.
@pytest.mark.parametrize("name,index", [("S", 0), ("K", 1), ("T", 2), ("sigma", 4)])
def test_invalid_inputs_raise_value_error(name: str, index: int) -> None:
    values = [100.0, 100.0, 1.0, 0.05, 0.2]
    values[index] = 0.0
    with pytest.raises(ValueError, match=f"{name} must be"):
        call_price(*values)


def test_nan_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="S must be"):
        call_price(np.nan, 100, 1, 0.05, 0.2)


@pytest.mark.parametrize("rate", [np.nan, np.inf])
def test_non_finite_rate_raises_value_error(rate: float) -> None:
    with pytest.raises(ValueError, match="r must be finite"):
        call_price(100, 100, 1, rate, 0.2)


@pytest.mark.parametrize("rate", [0.0, -0.01])
def test_non_positive_finite_rate_is_valid(rate: float) -> None:
    assert np.isfinite(call_price(100, 100, 1, rate, 0.2))
    assert np.isfinite(put_price(100, 100, 1, rate, 0.2))

#checks 200 random call and put configurations against py_vollib's established Black-Scholes implementation to within $10^{-8}$.
@pytest.mark.parametrize("flag,fn", [("c", call_price), ("p", put_price)])
def test_matches_py_vollib_on_random_inputs(flag, fn) -> None:
    bs = pytest.importorskip("py_vollib.black_scholes").black_scholes
    rng = np.random.default_rng(42)
    S = rng.uniform(50, 150, 200)
    K = rng.uniform(50, 150, 200)
    T = rng.uniform(0.1, 2, 200)
    r = rng.uniform(0, 0.1, 200)
    sigma = rng.uniform(0.05, 0.6, 200)
    for values in zip(S, K, T, r, sigma):
        expected = bs(flag, *values)
        assert fn(*values) == pytest.approx(expected, abs=1e-8)
