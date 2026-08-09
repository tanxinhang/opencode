"""Joint sensing-power and communication-bit allocation.

Each report link has a sensing power level ``p`` and a quantizer bit count
``b``.  Power scales the evidence separation by ``sqrt(p)``, and bits control
the communication fidelity.  The per-target frontier enumerates all
affordable ``(power, bits)`` combinations and the global max-min problem is
the same target-separable knapsack solved by ``exact_joint_maxmin``.
"""

from __future__ import annotations

import itertools

import numpy as np

from .fusion import optimal_gaussian_detection_probability
from .joint_allocation import exact_joint_maxmin, moments


def _target_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    grid: int,
) -> float:
    mu0 = [0.0]
    mu1 = [float(owner_delta)]
    var0 = [1.0]
    var1 = [1.0]
    for delta, power, bit_count in zip(report_deltas, powers, bits):
        if int(bit_count) <= 0:
            continue
        scaled_delta = float(delta) * np.sqrt(max(float(power), 0.0))
        m0, m1, v0, v1 = moments(scaled_delta, int(bit_count))
        mu0.append(m0)
        mu1.append(m1)
        var0.append(v0)
        var1.append(v1)
    return float(optimal_gaussian_detection_probability(
        np.asarray(mu0), np.asarray(mu1),
        np.diag(var0), np.diag(var1),
        set(range(len(mu0))), 0.05, grid=grid,
    ))


def power_bit_target_options(
    owner_delta: float,
    report_deltas: np.ndarray,
    *,
    power_levels: np.ndarray,
    bit_options: np.ndarray,
    budget: int,
    power_cost: float = 1.0,
    bit_cost: float = 1.0,
    grid: int = 32,
) -> list[tuple[int, float]]:
    """Pareto frontier of (cost, P_D) over power and bit choices."""
    reports = list(range(len(report_deltas)))
    per_report_choices = list(itertools.product(power_levels, bit_options))
    out = []
    for combo in itertools.product(
        per_report_choices, repeat=len(reports)
    ):
        powers = np.asarray([item[0] for item in combo], dtype=float)
        bits = np.asarray([item[1] for item in combo], dtype=int)
        cost = int(round(
            power_cost * float(powers.sum()) + bit_cost * float(bits.sum())
        ))
        if cost > budget:
            continue
        pd = _target_pd(
            owner_delta, np.asarray(report_deltas, dtype=float),
            powers, bits, grid,
        )
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
    if not pareto:
        pareto.append((0, _target_pd(
            owner_delta, np.asarray(report_deltas, dtype=float),
            np.zeros(len(report_deltas)), np.zeros(len(report_deltas), dtype=int),
            grid,
        )))
    return pareto


def exact_joint_power_bit_maxmin(
    target_groups: list[list[tuple[int, float]]],
    budget: int,
) -> float:
    """Exact max-min over joint power-bit target frontiers."""
    return exact_joint_maxmin(target_groups, budget)
