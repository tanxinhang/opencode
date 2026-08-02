from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import norm
from numpy.typing import ArrayLike, NDArray

from .linalg import stable_solve


def _indices(indices: Iterable[int]) -> NDArray[np.int64]:
    result = np.asarray(sorted(set(indices)), dtype=int)
    if result.size == 0:
        raise ValueError("fusion set cannot be empty")
    return result


def optimal_deflection(delta: ArrayLike, sigma0: ArrayLike, indices: Iterable[int]) -> float:
    idx = _indices(indices)
    d = np.asarray(delta, dtype=float)[idx]
    cov = np.asarray(sigma0, dtype=float)[np.ix_(idx, idx)]
    return float(d @ stable_solve(cov, d))


def optimal_weights(delta: ArrayLike, sigma0: ArrayLike, indices: Iterable[int]) -> NDArray[np.float64]:
    idx = _indices(indices)
    d = np.asarray(delta, dtype=float)[idx]
    cov = np.asarray(sigma0, dtype=float)[np.ix_(idx, idx)]
    direction = stable_solve(cov, d)
    normalization = float(np.sqrt(max(d @ direction, 0.0)))
    if normalization <= 1e-14:
        return np.zeros_like(direction)
    return direction / normalization


def conditional_marginal_deflection(
    delta: ArrayLike,
    sigma0: ArrayLike,
    selected: Iterable[int],
    candidate: int,
    epsilon: float = 1e-10,
) -> float:
    idx = _indices(selected)
    if candidate in idx:
        return 0.0
    d = np.asarray(delta, dtype=float)
    cov = np.asarray(sigma0, dtype=float)
    cov_s = cov[np.ix_(idx, idx)]
    c = cov[candidate, idx]
    inv_delta = stable_solve(cov_s, d[idx])
    inv_c = stable_solve(cov_s, c)
    residual_mean = d[candidate] - c @ inv_delta
    conditional_variance = cov[candidate, candidate] - c @ inv_c
    if conditional_variance <= epsilon:
        return 0.0
    return float(residual_mean**2 / conditional_variance)


def gaussian_detection_probability(
    mu0: ArrayLike,
    mu1: ArrayLike,
    sigma0: ArrayLike,
    sigma1: ArrayLike,
    indices: Iterable[int],
    false_alarm_rate: float,
) -> float:
    """Moment-matched Gaussian P_D for the deflection-optimal linear score."""
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must lie in (0, 1)")
    idx = _indices(indices)
    m0 = np.asarray(mu0, dtype=float)[idx]
    m1 = np.asarray(mu1, dtype=float)[idx]
    cov0 = np.asarray(sigma0, dtype=float)[np.ix_(idx, idx)]
    cov1 = np.asarray(sigma1, dtype=float)[np.ix_(idx, idx)]
    weights = optimal_weights(np.asarray(mu1) - np.asarray(mu0), sigma0, idx)
    mean0 = float(weights @ m0)
    mean1 = float(weights @ m1)
    variance0 = float(weights @ cov0 @ weights)
    variance1 = float(weights @ cov1 @ weights)
    if variance0 <= 1e-14 or variance1 <= 1e-14:
        return float(mean1 > mean0)
    threshold = mean0 + np.sqrt(variance0) * norm.ppf(1.0 - false_alarm_rate)
    return float(norm.sf((threshold - mean1) / np.sqrt(variance1)))
