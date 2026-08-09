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
from dataclasses import replace

import numpy as np

from .fusion import optimal_gaussian_detection_probability
from .joint_allocation import model_from_bits, moments


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


def _proportional_pd(
    owner_delta: float,
    deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    grid: int,
) -> float:
    full_deltas = np.concatenate((
        [float(owner_delta)],
        np.asarray(deltas, dtype=float)
        * np.sqrt(np.maximum(np.asarray(powers, dtype=float), 0.0)),
    ))
    full_bits = np.concatenate(([0], np.asarray(bits, dtype=int)))
    model = model_from_bits(
        full_deltas, full_bits, bit_flip_probability=0.0
    )
    model = replace(
        model,
        success_prob=np.ones(model.num_uavs),
        sigma1=model.sigma0,
    )
    return float(optimal_gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1,
        set(range(model.num_uavs)), 0.05, grid=grid,
    ))


def proportional_power_bit_options(
    owner_delta: float,
    deltas: np.ndarray,
    *,
    power_levels: np.ndarray,
    bit_options: np.ndarray,
    budget: int,
    grid: int = 32,
) -> list[tuple[int, float]]:
    """Exact power-bit frontier under Sigma1 = Sigma0 and no erasure."""
    deltas = np.asarray(deltas, dtype=float)
    out = []
    for combo in itertools.product(
        itertools.product(power_levels, bit_options), repeat=deltas.size
    ):
        powers = np.asarray([item[0] for item in combo], dtype=float)
        bits = np.asarray([item[1] for item in combo], dtype=int)
        cost = int(round(float(powers.sum()) + float(bits.sum())))
        if cost > budget:
            continue
        pd = _proportional_pd(owner_delta, deltas, powers, bits, grid)
        out.append((cost, pd))
    out.sort(key=lambda item: (item[0], -item[1]))
    pareto = []
    best_value = -1.0
    last_cost = None
    for cost, pd in out:
        if cost == last_cost:
            continue
        last_cost = cost
        if pd > best_value + 1e-12:
            pareto.append((cost, pd))
            best_value = pd
    return pareto


def winner_take_all_proportional_options(
    owner_delta: float,
    deltas: np.ndarray,
    *,
    bit_options: np.ndarray,
    budget: int,
    grid: int = 32,
) -> list[tuple[int, float]]:
    """Winner-take-all power frontier under the same proportional model."""
    deltas = np.asarray(deltas, dtype=float)
    out = []
    for combo in itertools.product(bit_options, repeat=deltas.size):
        bits = np.asarray(combo, dtype=int)
        bit_cost = int(bits.sum())
        if bit_cost > budget:
            continue
        selected = [i for i in range(deltas.size) if bits[i] > 0]
        for power_budget in range(0, budget - bit_cost + 1):
            powers = np.zeros(deltas.size)
            if selected:
                coefficients = np.asarray([
                    power_gain_coefficient(
                        deltas[i], int(bits[i]), 0.0, 1.0
                    )
                    for i in selected
                ])
                powers[selected[int(np.argmax(coefficients))]] = float(
                    power_budget
                )
            pd = _proportional_pd(
                owner_delta, deltas, powers, bits, grid
            )
            out.append((bit_cost + int(power_budget), pd))
    result = sorted(out, key=lambda item: (item[0], -item[1]))
    pareto = []
    best_value = -1.0
    last_cost = None
    for cost, pd in result:
        if cost == last_cost:
            continue
        last_cost = cost
        if pd > best_value + 1e-12:
            pareto.append((cost, pd))
            best_value = pd
    return pareto
