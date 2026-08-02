from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def nearest_psd(matrix: ArrayLike, epsilon: float = 1e-10) -> NDArray[np.float64]:
    """Project a real covariance estimate onto the positive-semidefinite cone."""
    a = np.asarray(matrix, dtype=float)
    sym = 0.5 * (a + a.T)
    values, vectors = np.linalg.eigh(sym)
    values = np.maximum(values, epsilon)
    return (vectors * values) @ vectors.T


def regularize_covariance(
    covariance: ArrayLike,
    shrinkage: float = 0.0,
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
    cov = nearest_psd(covariance, epsilon=0.0)
    diagonal = np.diag(np.diag(cov))
    result = (1.0 - shrinkage) * cov + shrinkage * diagonal
    return nearest_psd(result + epsilon * np.eye(cov.shape[0]), epsilon=epsilon)


def stable_solve(covariance: ArrayLike, vector: ArrayLike) -> NDArray[np.float64]:
    cov = np.asarray(covariance, dtype=float)
    vec = np.asarray(vector, dtype=float)
    try:
        return np.linalg.solve(cov, vec)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov, hermitian=True) @ vec

