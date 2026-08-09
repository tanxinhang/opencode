"""Winner-take-all allocation under coefficient error with feedback.

The algorithm starts with noisy per-report gain estimates.  Each round it
explores a small number of top-ranked candidates, corrects their estimates
toward the observed true gains, and allocates all power to the current best
estimate.  With enough rounds the true winner is explored and chosen, so the
allocation converges to winner-take-all under distinct coefficients.
"""

from __future__ import annotations

import numpy as np


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
