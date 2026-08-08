from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.optimize import minimize_scalar
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
        raise FloatingPointError(
            "Gaussian detection probability is undefined for a degenerate score variance"
        )
    threshold = mean0 + np.sqrt(variance0) * norm.ppf(1.0 - false_alarm_rate)
    return float(norm.sf((threshold - mean1) / np.sqrt(variance1)))


def gaussian_pd_closed_form(
    deflection: float,
    variance_ratio: float,
    false_alarm_rate: float,
) -> float:
    """Closed-form P_D in the proportional-covariance regime.

    When ``Sigma1 = variance_ratio * Sigma0``, the deflection-optimal linear
    score also maximizes P_D, and

    ``P_D = Phi((sqrt(D) - Phi^-1(1 - P_FA)) / sqrt(variance_ratio))``

    with ``D = delta_S^T Sigma0_SS^-1 delta_S``.
    """
    if deflection < 0.0:
        raise ValueError("deflection must be nonnegative")
    if variance_ratio <= 0.0:
        raise ValueError("variance_ratio must be positive")
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must lie in (0, 1)")
    threshold = norm.ppf(1.0 - false_alarm_rate)
    return float(norm.cdf(
        (np.sqrt(deflection) - threshold) / np.sqrt(variance_ratio)
    ))


def _pd_optimal_score_components(
    mu0: ArrayLike,
    mu1: ArrayLike,
    sigma0: ArrayLike,
    sigma1: ArrayLike,
    indices: Iterable[int],
    false_alarm_rate: float,
    *,
    grid: int = 4096,
) -> tuple[float, NDArray[np.float64], float]:
    """Return (shift, unit-null-variance weights, mu) of the P_D-optimal score.

    For a linear score ``s = w^T x`` with threshold
    ``tau = w^T mu0 + Phi^-1(1 - P_FA) sqrt(w^T Sigma0 w)``,

    ``P_D(w) = Phi((w^T delta - z sqrt(w^T Sigma0 w)) / sqrt(w^T Sigma1 w))``.

    The score is scale invariant.  In whitened coordinates
    ``y = Sigma0^1/2 w``, ``a = Sigma0^-1/2 delta`` and
    ``Q = Sigma0^-1/2 Sigma1 Sigma0^-1/2`` the shift is

    ``f(y) = (a^T y - z ||y||) / sqrt(y^T Q y)``.

    A KKT analysis shows that every positive-shift stationary point has
    direction ``y(mu) = (Q + mu I)^-1 a`` for ``mu >= 0``, so maximizing
    P_D over the one-parameter family attains the global optimum over linear
    scores whenever the optimum has P_D >= 0.5.  Since the zero-extension of
    any subset-optimal weight stays feasible for a superset, the resulting
    P_D is set-monotone at these operating points.
    """
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must lie in (0, 1)")
    if grid < 8:
        raise ValueError("grid must be at least 8")
    idx = _indices(indices)
    m0 = np.asarray(mu0, dtype=float)[idx]
    m1 = np.asarray(mu1, dtype=float)[idx]
    delta = m1 - m0
    cov0 = np.asarray(sigma0, dtype=float)[np.ix_(idx, idx)]
    cov1 = np.asarray(sigma1, dtype=float)[np.ix_(idx, idx)]
    if np.linalg.eigvalsh(cov0).min() <= 0.0:
        raise ValueError("sigma0 must be positive definite")
    if np.linalg.eigvalsh(cov1).min() <= 0.0:
        raise ValueError("sigma1 must be positive definite")
    z = norm.ppf(1.0 - false_alarm_rate)
    cholesky0 = np.linalg.cholesky(cov0)
    inverse0 = stable_solve(cholesky0, np.eye(cov0.shape[0]))
    a = inverse0 @ delta
    q = inverse0 @ cov1 @ inverse0.T
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    projected = eigenvectors.T @ a
    mu_grid = np.concatenate((
        np.linspace(0.0, 3.0, grid // 2),
        np.geomspace(3.0 + 1e-3, 1e6, grid // 2),
    ))
    denom = eigenvalues[None, :] + mu_grid[:, None]
    numerator = np.sum(
        (projected * projected)[None, :] / denom, axis=1
    )
    null_norm2 = np.sum(
        (projected * projected)[None, :] / denom**2, axis=1
    )
    h1_norm2 = np.sum(
        (eigenvalues * projected * projected)[None, :] / denom**2,
        axis=1,
    )
    shifts = (
        numerator - z * np.sqrt(np.maximum(null_norm2, 0.0))
    ) / np.sqrt(np.maximum(h1_norm2, 1e-30))

    def shift_at(mu: float) -> float:
        denom = eigenvalues + mu
        num = float(np.sum((projected * projected) / denom))
        null2 = float(np.sum((projected * projected) / denom**2))
        h1_2 = float(np.sum((eigenvalues * projected * projected) / denom**2))
        return (
            num - z * np.sqrt(max(null2, 0.0))
        ) / np.sqrt(max(h1_2, 1e-30))

    # Include the mu -> infinity limit, i.e. the deflection-optimal direction
    # y = a, so the family is closed at both ends of the parameter range.
    deflection_limit = (
        float(a @ a) - z * float(np.linalg.norm(a))
    ) / np.sqrt(max(float(a @ q @ a), 1e-30))
    shifts = np.concatenate((shifts, [deflection_limit]))
    best_index = int(np.argmax(shifts))
    best_shift = float(shifts[best_index])
    if best_index == shifts.size - 1:
        direction = a
        mu_best = np.inf
    else:
        mu_best = float(mu_grid[best_index])
        if best_index > 0 or best_index < mu_grid.size - 1:
            lo_index = max(best_index - 1, 0)
            hi_index = min(best_index + 1, mu_grid.size - 1)
            refined = minimize_scalar(
                lambda mu: -shift_at(mu),
                bounds=(mu_grid[lo_index], mu_grid[hi_index]),
                method="bounded",
                options={"xatol": 1e-11, "maxiter": 200},
            )
            if -refined.fun > shifts[best_index]:
                mu_best = float(refined.x)
                best_shift = float(-refined.fun)
        direction = eigenvectors @ (
            projected / (eigenvalues + mu_best)
        )
    weights = stable_solve(cholesky0.T, direction)
    null_variance = float(np.sqrt(max(weights @ cov0 @ weights, 1e-30)))
    if null_variance > 1e-14:
        weights = weights / null_variance
    return best_shift, weights, mu_best


def pd_shift_upper_bound(
    mu0: ArrayLike,
    mu1: ArrayLike,
    sigma0: ArrayLike,
    sigma1: ArrayLike,
    indices: Iterable[int],
    false_alarm_rate: float,
) -> float:
    """Tight dual upper bound on the P_D-optimal linear-score shift.

    In whitened coordinates ``y = L^T w``, ``a = L^-1 delta`` and
    ``Q = L^-1 Sigma1 L^-T``, every linear score has shift

    ``s(y) = (a^T y - z ||y||) / sqrt(y^T Q y)``

    with ``z = Phi^-1(1 - P_FA)``.  For any ``mu >= 0``,

    ``y^T(Q + mu I)y = 1 + mu ||y||^2``

    and Cauchy-Schwarz gives

    ``a^T y <= sqrt(a^T(Q+mu I)^-1 a) sqrt(1 + mu ||y||^2)``.

    Since ``y^T Q y = 1``, Rayleigh's quotient gives
    ``||y|| in [1/sqrt(lambda_max(Q)), 1/sqrt(lambda_min(Q))]``, and the
    one-dimensional majorant ``g(t)`` is convex, so its maximum over the
    interval is attained at one of the two endpoints.  Minimizing the
    resulting bound over ``mu >= 0`` yields the upper bound used here.  The
    earlier Cauchy bound is the ``mu = 0`` member of this family:

    ``s(y) <= sqrt(a^T Q^-1 a) - z / sqrt(lambda_max(Q))``.

    The bound is valid for every linear score, including ``z < 0``, and is
    used as a pruning bound that does not require ``P_D >= 0.5``.
    """
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must lie in (0, 1)")
    idx = _indices(indices)
    m0 = np.asarray(mu0, dtype=float)[idx]
    m1 = np.asarray(mu1, dtype=float)[idx]
    delta = m1 - m0
    cov0 = np.asarray(sigma0, dtype=float)[np.ix_(idx, idx)]
    cov1 = np.asarray(sigma1, dtype=float)[np.ix_(idx, idx)]
    if np.linalg.eigvalsh(cov0).min() <= 0.0:
        raise ValueError("sigma0 must be positive definite")
    if np.linalg.eigvalsh(cov1).min() <= 0.0:
        raise ValueError("sigma1 must be positive definite")
    cholesky0 = np.linalg.cholesky(cov0)
    inverse0 = stable_solve(cholesky0, np.eye(cov0.shape[0]))
    a = inverse0 @ delta
    q = inverse0 @ cov1 @ inverse0.T
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    z = norm.ppf(1.0 - false_alarm_rate)
    projected = eigenvectors.T @ a
    t_lo = 1.0 / float(np.sqrt(max(eigenvalues.max(), 1e-30)))
    t_hi = 1.0 / float(np.sqrt(max(eigenvalues.min(), 1e-30)))

    def endpoint(mu: float) -> float:
        energy = float(np.sum(projected**2 / (eigenvalues + mu)))
        amplitude = float(np.sqrt(max(energy, 0.0)))
        return max(
            amplitude * np.sqrt(1.0 + mu * t_lo**2) - z * t_lo,
            amplitude * np.sqrt(1.0 + mu * t_hi**2) - z * t_hi,
        )

    mu_grid = np.concatenate(([0.0], np.geomspace(1e-8, 1e8, 32)))
    values = np.asarray([endpoint(mu) for mu in mu_grid])
    best_index = int(np.argmin(values))
    lo = 0.0 if best_index == 0 else mu_grid[best_index - 1]
    hi = 1e9 if best_index == mu_grid.size - 1 else mu_grid[best_index + 1]
    refined = minimize_scalar(
        endpoint,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 100},
    )
    return float(min(refined.fun, values.min()))


def optimal_gaussian_weights(
    mu0: ArrayLike,
    mu1: ArrayLike,
    sigma0: ArrayLike,
    sigma1: ArrayLike,
    indices: Iterable[int],
    false_alarm_rate: float,
    *,
    grid: int = 4096,
) -> NDArray[np.float64]:
    """Return P_D-optimal linear weights normalized to unit H0 variance."""
    _, weights, _ = _pd_optimal_score_components(
        mu0, mu1, sigma0, sigma1, indices, false_alarm_rate, grid=grid
    )
    return weights


def optimal_gaussian_detection_probability(
    mu0: ArrayLike,
    mu1: ArrayLike,
    sigma0: ArrayLike,
    sigma1: ArrayLike,
    indices: Iterable[int],
    false_alarm_rate: float,
    *,
    grid: int = 4096,
) -> float:
    """Max P_D over the one-parameter family of linear fusion scores.

    The deflection-optimal score is the ``mu -> infinity`` limit of the
    family, so the returned value is never below
    :func:`gaussian_detection_probability`.  At operating points with
    P_D >= 0.5 the family contains the global linear-score optimum, which
    makes the set function monotone under report addition.
    """
    shift, _, _ = _pd_optimal_score_components(
        mu0, mu1, sigma0, sigma1, indices, false_alarm_rate, grid=grid
    )
    return float(norm.cdf(shift))
