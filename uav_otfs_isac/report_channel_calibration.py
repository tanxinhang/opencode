"""G1-B exact versus Monte Carlo moments for the report channel.

The evidence law is: Gaussian source moments -> scalar quantization -> BSC
bit errors -> detectable erasure.  This module computes exact joint moments of
the received report values (value times received indicator) and compares them
with Monte Carlo estimates, closing the loop required by G1-B.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

from .reporting import (
    bsc_transition,
    gaussian_bin_probabilities,
    quantize,
    transmit_indices,
)


def _bsc_reconstruction_profiles(
    edges: np.ndarray,
    values: np.ndarray,
    bits: int,
    bit_flip_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-source-bin expected received value and expected square."""
    transition = bsc_transition(bits, bit_flip_probability)
    mean_profile = transition @ values
    square_profile = transition @ (values ** 2)
    return mean_profile, square_profile


def exact_received_moments(
    mu: np.ndarray,
    covariance: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    bits: int,
    bit_flip_probability: float,
    success_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact unconditional mean and covariance of received report values."""
    mu = np.asarray(mu, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    success = np.asarray(success_probabilities, dtype=float)
    n = len(mu)
    if covariance.shape != (n, n) or success.shape != (n,):
        raise ValueError("moment inputs must match the number of reports")
    mean_profile, square_profile = _bsc_reconstruction_profiles(
        edges, values, bits, bit_flip_probability
    )
    source_probabilities = [
        gaussian_bin_probabilities(mu[i], covariance[i, i], edges)
        for i in range(n)
    ]
    exact_mean = np.zeros(n)
    exact_second = np.zeros((n, n))
    for i in range(n):
        exact_mean[i] = success[i] * float(
            source_probabilities[i] @ mean_profile
        )
        exact_second[i, i] = success[i] * float(
            source_probabilities[i] @ square_profile
        )
    for i in range(n):
        for j in range(i + 1, n):
            cross = covariance[i, j]
            joint = np.zeros((len(edges) - 1, len(edges) - 1))
            mean = np.asarray((mu[i], mu[j]))
            joint_covariance = np.asarray((
                (covariance[i, i], cross),
                (cross, covariance[j, j]),
            ))
            distribution = multivariate_normal(mean=mean, cov=joint_covariance)
            bin_edges = np.asarray(edges, dtype=float)
            for ki in range(bin_edges.size - 1):
                for kj in range(bin_edges.size - 1):
                    joint[ki, kj] = distribution.cdf((
                        bin_edges[ki + 1], bin_edges[kj + 1]
                    )) - distribution.cdf((
                        bin_edges[ki], bin_edges[kj + 1]
                    )) - distribution.cdf((
                        bin_edges[ki + 1], bin_edges[kj]
                    )) + distribution.cdf((
                        bin_edges[ki], bin_edges[kj]
                    ))
            # The product term is sum_{k,l} P[k,l] * r_i[k] * r_j[l].
            expected_product = float(
                np.sum(joint * np.outer(mean_profile, mean_profile))
            )
            exact_second[i, j] = exact_second[j, i] = (
                success[i] * success[j] * expected_product
            )
    exact_mean = np.asarray(exact_mean, dtype=float)
    exact_covariance = exact_second - np.outer(exact_mean, exact_mean)
    return {"mean": exact_mean, "covariance": exact_covariance}


def simulate_received_moments(
    mu: np.ndarray,
    covariance: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    bits: int,
    bit_flip_probability: float,
    success_probabilities: np.ndarray,
    trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Monte Carlo mean/covariance with erasures mapped to zero indicators."""
    rng = np.random.default_rng(seed)
    received = np.zeros((trials, len(mu)))
    for trial in range(trials):
        z = rng.multivariate_normal(mu, covariance)
        sent = quantize(z, edges)
        decoded = transmit_indices(
            sent, bits, np.full(len(mu), bit_flip_probability), rng
        )
        kept = rng.random(len(mu)) < success_probabilities
        received[trial] = np.where(
            kept, np.asarray(values)[decoded], 0.0
        )
    return {
        "mean": np.mean(received, axis=0),
        "covariance": np.cov(received, rowvar=False, ddof=1),
    }


def relative_errors(
    exact: dict[str, np.ndarray],
    simulated: dict[str, np.ndarray],
    *,
    main_pairs: tuple[tuple[int, int], ...] = ((0, 1),),
    mean_floor: float = 0.05,
    covariance_floor_scale: float = 0.1,
) -> dict:
    """Relative errors for nonzero means, diagonal and main cross-covariances."""
    exact_mean = np.asarray(exact["mean"], dtype=float)
    simulated_mean = np.asarray(simulated["mean"], dtype=float)
    exact_covariance = np.asarray(exact["covariance"], dtype=float)
    simulated_covariance = np.asarray(simulated["covariance"], dtype=float)
    mean_errors = []
    mean_errors_per_report = []
    for i in range(len(exact_mean)):
        denominator = max(abs(exact_mean[i]), mean_floor)
        error = abs(simulated_mean[i] - exact_mean[i]) / denominator
        mean_errors_per_report.append(float(error))
        if abs(exact_mean[i]) >= mean_floor:
            mean_errors.append(error)
    max_diagonal = float(np.max(np.abs(np.diag(exact_covariance))))
    covariance_errors = []
    covariance_errors_matrix = np.zeros_like(exact_covariance)
    entries = [(i, i) for i in range(len(exact_mean))] + list(main_pairs)
    for i, j in entries:
        denominator = max(
            abs(exact_covariance[i, j]),
            covariance_floor_scale * max_diagonal,
        )
        error = abs(
            simulated_covariance[i, j] - exact_covariance[i, j]
        ) / denominator
        covariance_errors.append(float(error))
        covariance_errors_matrix[i, j] = float(error)
        covariance_errors_matrix[j, i] = float(error)
    return {
        "mean_relative_error": float(np.max(mean_errors)) if mean_errors else 0.0,
        "mean_relative_error_per_report": mean_errors_per_report,
        "covariance_relative_error": float(np.max(covariance_errors)),
        "covariance_relative_error_matrix": covariance_errors_matrix.tolist(),
    }
