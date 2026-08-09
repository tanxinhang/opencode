"""Winner-take-all allocation under coefficient error with feedback.

The algorithm starts with noisy per-report gain estimates.  Each round it
explores a small number of top-ranked candidates, corrects their estimates
toward the observed true gains, and allocates all power to the current best
estimate.  With enough rounds the true winner is explored and chosen, so the
allocation converges to winner-take-all under distinct coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def wta_feedback_allocator(
    true_coefficients: np.ndarray,
    noisy_coefficients: np.ndarray,
    budget: float,
    *,
    rounds: int,
    learning_rate: float,
    explore: int = 1,
) -> dict:
    """Iterative feedback WTA allocation under coefficient error."""
    true = np.asarray(true_coefficients, dtype=float)
    estimates = np.asarray(noisy_coefficients, dtype=float).copy()
    trace = []
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        order = np.argsort(-estimates, kind="stable")
        explored = order[:explore]
        if explore < estimates.size:
            candidates = [
                int(i) for i in range(estimates.size)
                if int(i) not in explored
            ]
            explored = np.concatenate((
                explored,
                [int(rng.choice(candidates))],
            ))
        estimates[explored] += learning_rate * (
            true[explored] - estimates[explored]
        )
        best = int(np.argmax(estimates))
        allocation = np.zeros_like(estimates)
        allocation[best] = float(budget)
        trace.append({
            "allocation": allocation.tolist(),
            "true_deflection": float(true @ allocation),
            "estimates": estimates.tolist(),
        })
    return {
        "allocation": trace[-1]["allocation"],
        "true_deflection": trace[-1]["true_deflection"],
        "best_report": int(np.argmax(estimates)),
        "trace": trace,
    }


def one_shot_wta(
    true_coefficients: np.ndarray,
    noisy_coefficients: np.ndarray,
    budget: float,
) -> dict:
    """One-shot allocation from noisy estimates without feedback."""
    true = np.asarray(true_coefficients, dtype=float)
    noisy = np.asarray(noisy_coefficients, dtype=float)
    best = int(np.argmax(noisy))
    allocation = np.zeros_like(true)
    allocation[best] = float(budget)
    return {
        "allocation": allocation.tolist(),
        "true_deflection": float(true @ allocation),
        "best_report": best,
    }


def evaluate_feedback_gain(
    true_coefficients: np.ndarray,
    noise_scale: float,
    *,
    budget: float,
    rounds: int,
    learning_rate: float,
    explore: int,
    seed: int,
) -> dict:
    """Compare one-shot and multi-round feedback on one noisy draw."""
    rng = np.random.default_rng(seed)
    noisy = true_coefficients + noise_scale * rng.standard_normal(
        true_coefficients.size
    )
    one = one_shot_wta(true_coefficients, noisy, budget)
    feedback = wta_feedback_allocator(
        true_coefficients,
        noisy,
        budget,
        rounds=rounds,
        learning_rate=learning_rate,
        explore=explore,
    )
    return {
        "one_shot_deflection": one["true_deflection"],
        "feedback_deflection": feedback["true_deflection"],
        "feedback_improvement": (
            feedback["true_deflection"] - one["true_deflection"]
        ),
        "one_shot_best": one["best_report"],
        "feedback_best": feedback["best_report"],
    }


def ucb_wta_feedback_allocator(
    true_coefficients: np.ndarray,
    initial_estimates: np.ndarray,
    budget: float,
    *,
    observation_noise_scale: float,
    max_rounds: int,
    confidence: float = 0.05,
    explore: int = 1,
    prior_noise_scale: float | None = None,
    seed: int = 0,
) -> dict:
    """UCB-style WTA allocation with a certificate-based stopping rule.

    Every report keeps a running mean and an uncertainty width
    ``beta * sigma / sqrt(n)``.  Exploration observes the highest-UCB
    reports; allocation goes to the best UCB.  The loop stops when the best
    report's lower confidence bound exceeds the second-best upper confidence
    bound, or when ``max_rounds`` is reached.
    """
    true = np.asarray(true_coefficients, dtype=float)
    means = np.asarray(initial_estimates, dtype=float).copy()
    counts = np.ones(true.size, dtype=float)
    sigma = float(observation_noise_scale)
    if prior_noise_scale is None:
        prior_noise_scale = max(3.0 * sigma, 0.3)
    prior_sigma = float(prior_noise_scale)
    beta = float(norm.ppf(1.0 - confidence / (2.0 * true.size)))
    rng = np.random.default_rng(seed)
    stopped_by_certificate = False
    rounds_used = 0
    for _ in range(max_rounds):
        rounds_used += 1
        uncertainty = beta * prior_sigma / np.sqrt(counts)
        ucb = means + uncertainty
        order = np.argsort(-ucb, kind="stable")
        explored = order[:explore]
        if explore < true.size:
            candidates = [
                int(i) for i in range(true.size)
                if int(i) not in explored
            ]
            explored = np.concatenate((
                explored,
                [int(rng.choice(candidates))],
            ))
        noise = rng.normal(0.0, sigma, explored.size)
        for index, observation_noise in zip(explored, noise):
            observed = true[index] + observation_noise
            means[index] = (
                means[index] * counts[index] + observed
            ) / (counts[index] + 1.0)
            counts[index] += 1.0
        uncertainty = beta * prior_sigma / np.sqrt(counts)
        ucb = means + uncertainty
        order = np.argsort(-ucb, kind="stable")
        best = int(order[0])
        second = int(order[1])
        lcb_best = means[best] - uncertainty[best]
        ucb_second = means[second] + uncertainty[second]
        if lcb_best > ucb_second:
            stopped_by_certificate = True
            break
    allocation = np.zeros_like(true)
    allocation[best] = float(budget)
    return {
        "allocation": allocation.tolist(),
        "true_deflection": float(true @ allocation),
        "best_report": best,
        "rounds_used": rounds_used,
        "stopped_by_certificate": stopped_by_certificate,
        "final_means": means.tolist(),
        "final_counts": counts.tolist(),
    }
