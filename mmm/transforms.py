"""Media transformations: geometric Adstock (carry-over) and Hill saturation.

Adstock is *normalized* (weights sum to 1), so a constant weekly spend ``x``
has a steady-state adstock of exactly ``x``. That keeps the Hill
half-saturation point ``K`` in interpretable spend units and lets the
optimizer and the Power BI DAX measures evaluate steady-state response
without simulating carry-over.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """Normalized geometric adstock along axis 0 (time).

    a[t] = theta * a[t-1] + (1 - theta) * x[t], with a[-1] = x[0]
    (the series is assumed to start in steady state).
    """
    x = np.asarray(x, dtype=float)
    if not 0.0 <= theta < 1.0:
        raise ValueError(f"theta must be in [0, 1), got {theta}")
    if theta == 0.0:
        return x.copy()
    zi = theta * x[:1]                       # initial filter state = theta * a[-1]
    out, _ = lfilter([1.0 - theta], [1.0, -theta], x, axis=0, zi=zi)
    return out


def adstock_half_life(theta: float) -> float:
    """Weeks until a single impulse decays to half its initial effect."""
    if theta <= 0.0:
        return 0.0
    return float(np.log(0.5) / np.log(theta))


def hill(u: np.ndarray, k: float, s: float) -> np.ndarray:
    """Hill saturation h(u) = u^s / (u^s + k^s), bounded in [0, 1)."""
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    us = u ** s
    return us / (us + k ** s)


def hill_derivative(u: np.ndarray, k: float, s: float) -> np.ndarray:
    """dh/du = s * k^s * u^(s-1) / (u^s + k^s)^2."""
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    k, s = np.asarray(k, dtype=float), np.asarray(s, dtype=float)
    ks = k ** s
    with np.errstate(divide="ignore", invalid="ignore"):
        d = s * ks * u ** (s - 1.0) / (u ** s + ks) ** 2
    # limit at u = 0: infinite for s < 1, 1/k for s = 1, zero for s > 1
    at_zero = np.where(s < 1.0, np.inf, np.where(s == 1.0, 1.0 / k, 0.0))
    return np.where(u == 0.0, at_zero, d)
