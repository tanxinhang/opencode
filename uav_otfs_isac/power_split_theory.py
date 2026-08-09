"""Winner-take-all sensing power allocation under proportional covariance.

For a diagonal proportional-covariance target model with deterministic
reception, the deflection of a power allocation is linear in each report's
power:

``D(p) = sum_i p_i * J_i``

with ``J_i = s_i * delta_i(b_i)^2 / v_i(b_i)``.  Since P_D is monotone in
deflection, the optimal power allocation puts all budget on the report with
the largest ``J_i``.
"""

from __future__ import annotations

import itertools

import numpy as np

from .joint_allocation import moments


def power_gain_coefficient(
    delta: float,
    bits: int,
    flip_probability: float,
    success_probability: float,
) -> float:
    """Per-unit-power communication-aware sensing gain."""
    m0, m1, v0, v1 = moments(float(delta), int(bits), float(flip_probability))
    return float(
        success_probability
        * (m1 - m0) ** 2
        / max(v0, 1e-12)
    )


def winner_take_all_allocation(
    coefficients: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Allocate all power to the report with the largest coefficient."""
    coefficients = np.asarray(coefficients, dtype=float)
    allocation = np.zeros_like(coefficients)
    allocation[int(np.argmax(coefficients))] = float(budget)
    return allocation


def verify_winner_take_all(
    owner_delta: float,
    deltas: np.ndarray,
    bits: np.ndarray,
    *,
    flip_probability: float,
    success_probability: float,
    budget: float,
    power_levels: np.ndarray,
    grid: int = 32,
) -> dict:
    """Compare winner-take-all with exhaustive power allocations."""
    coefficients = np.asarray([
        power_gain_coefficient(
            delta, int(bit_count), flip_probability, success_probability
        )
        for delta, bit_count in zip(deltas, bits)
    ])
    winner = winner_take_all_allocation(coefficients, budget)
    best_deflection = -1.0
    best_allocation = None
    for powers in itertools.product(power_levels, repeat=deltas.size):
        powers = np.asarray(powers, dtype=float)
        if abs(float(powers.sum()) - float(budget)) > 1e-9:
            continue
        deflection = float(coefficients @ powers)
        if deflection > best_deflection + 1e-12:
            best_deflection = deflection
            best_allocation = powers
    winner_deflection = float(coefficients @ winner)
    return {
        "winner_allocation": winner.tolist(),
        "best_allocation": best_allocation.tolist(),
        "winner_deflection": winner_deflection,
        "best_deflection": best_deflection,
        "passed": winner_deflection >= best_deflection - 1e-9,
    }
