from __future__ import annotations

from collections.abc import Iterable

import numpy as np
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

