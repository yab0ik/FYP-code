"""Vectorised Black-Scholes pricing for European options."""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

Numeric: TypeAlias = ArrayLike
NumericResult: TypeAlias = float | NDArray[np.float64]


def _validate_positive(name: str, value: Numeric) -> NDArray[np.float64]:
    """Convert a value to an array and require finite, positive values."""
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and greater than zero")
    return array


def d1_d2(
    S: Numeric, K: Numeric, T: Numeric, r: Numeric, sigma: Numeric
) -> tuple[NumericResult, NumericResult]:
    """Return Black-Scholes ``d1`` and ``d2``.

    T in years; r continuously compounded; no dividends; constant sigma and r.
    Formula: d1 = (ln(S/K) + (r + sigma^2/2)T)/(sigma*sqrt(T)); d2 = d1 - sigma*sqrt(T).
    """
    d1, d2, _, _, _, _ = _calculate_d1_d2(S, K, T, r, sigma)
    return d1, d2


def _calculate_d1_d2(
    S: Numeric, K: Numeric, T: Numeric, r: Numeric, sigma: Numeric
) -> tuple[
    NumericResult,
    NumericResult,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Calculate ``d1`` and ``d2`` along with validated pricing inputs."""
    spot = _validate_positive("S", S)
    strike = _validate_positive("K", K)
    time = _validate_positive("T", T)
    volatility = _validate_positive("sigma", sigma)
    rate = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(rate)):
        raise ValueError("r must be finite")

    volatility_sqrt_time = volatility * np.sqrt(time)
    d1 = (
        np.log(spot / strike) + (rate + volatility**2 / 2) * time
    ) / volatility_sqrt_time
    d2 = d1 - volatility_sqrt_time
    return d1, d2, spot, strike, time, rate


def call_price(S: Numeric, K: Numeric, T: Numeric, r: Numeric, sigma: Numeric) -> NumericResult:
    """Return a European call price using Black-Scholes.

    T in years; r continuously compounded; no dividends; constant sigma and r.
    Formula: S*N(d1) - K*exp(-rT)*N(d2).
    """

    d1, d2, spot, strike, time, rate = _calculate_d1_d2(S, K, T, r, sigma)
    result = spot * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2)
    return result


def put_price(S: Numeric, K: Numeric, T: Numeric, r: Numeric, sigma: Numeric) -> NumericResult:
    """Return a European put price using Black-Scholes.

    T in years; r continuously compounded; no dividends; constant sigma and r.
    Formula: K*exp(-rT)*N(-d2) - S*N(-d1).
    """

    d1, d2, spot, strike, time, rate = _calculate_d1_d2(S, K, T, r, sigma)
    result = strike * np.exp(-rate * time) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return result
