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
from scipy.stats import norm

from .fusion import optimal_gaussian_detection_probability
from .joint_allocation import (
    exact_joint_maxmin,
    exact_joint_maxmin_selection,
    moments,
)


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
    batch_size: int = 50_000,
) -> list[tuple[int, float]]:
    """Pareto frontier of (cost, P_D) over power and bit choices."""
    return vectorized_power_bit_target_options(
        owner_delta,
        report_deltas,
        power_levels=power_levels,
        bit_options=bit_options,
        budget=budget,
        power_cost=power_cost,
        bit_cost=bit_cost,
        grid=grid,
        batch_size=batch_size,
    )


def vectorized_power_bit_target_options(
    owner_delta: float,
    report_deltas: np.ndarray,
    *,
    power_levels: np.ndarray,
    bit_options: np.ndarray,
    budget: int,
    power_cost: float = 1.0,
    bit_cost: float = 1.0,
    grid: int = 32,
    batch_size: int = 50_000,
    max_combos: int = 4_000_000,
) -> list[tuple[int, float]]:
    """Batched exact Pareto frontier over power and bit choices."""
    deltas = np.asarray(report_deltas, dtype=float)
    reports = list(range(deltas.size))
    choices = [
        (float(p), int(b))
        for p, b in itertools.product(power_levels, bit_options)
        if float(power_cost) * float(p) + float(bit_cost) * int(b) <= float(budget)
    ]
    if len(choices) == 0:
        return [(0, _target_pd(
            owner_delta, deltas,
            np.zeros(deltas.size), np.zeros(deltas.size, dtype=int), grid,
        ))]
    total_combos = len(choices) ** deltas.size
    if total_combos > max_combos:
        raise ValueError(
            f"power-bit enumeration would materialize {total_combos} "
            f"combinations (reports={deltas.size}, choices={len(choices)}); "
            f"cap is {max_combos}. Reduce report count, coarsen power_levels/"
            f"bit_options, lower budget, or raise max_combos."
        )
    pre_a = np.zeros((deltas.size, len(choices)), dtype=float)
    pre_q = np.ones((deltas.size, len(choices)), dtype=float)
    pre_cost = np.zeros((deltas.size, len(choices)), dtype=float)
    for i, delta in enumerate(deltas):
        for j, (power, bits) in enumerate(choices):
            if int(bits) <= 0:
                continue
            scaled_delta = float(delta) * np.sqrt(max(float(power), 0.0))
            m0, m1, v0, v1 = moments(scaled_delta, int(bits))
            pre_a[i, j] = (m1 - m0) / np.sqrt(max(v0, 1e-12))
            pre_q[i, j] = max(v1 / max(v0, 1e-12), 1e-12)
            pre_cost[i, j] = (
                float(power_cost) * float(power)
                + float(bit_cost) * int(bits)
            )

    all_combos = np.asarray(
        list(itertools.product(range(len(choices)), repeat=deltas.size)),
        dtype=np.int64,
    )
    costs = np.zeros(all_combos.shape[0], dtype=float)
    for i in reports:
        costs += pre_cost[i, all_combos[:, i]]
    mask = costs <= float(budget)
    all_combos = all_combos[mask]
    costs = np.round(costs[mask]).astype(int)
    if all_combos.shape[0] == 0:
        return [(0, _target_pd(
            owner_delta, deltas,
            np.zeros(deltas.size), np.zeros(deltas.size, dtype=int), grid,
        ))]

    z = float(norm.ppf(0.95))
    owner_a2 = float(owner_delta ** 2)
    mu_grid = np.concatenate((
        np.linspace(0.0, 3.0, grid // 2),
        np.geomspace(3.0 + 1e-3, 1e6, grid // 2),
    ))
    frontier: dict[int, float] = {}
    report_index = np.arange(deltas.size)[None, :]

    def evaluate_batch(batch: np.ndarray):
        selected = pre_a[report_index, batch] != 0.0
        a = np.where(
            selected,
            pre_a[report_index, batch],
            0.0,
        )
        q = np.where(
            selected,
            pre_q[report_index, batch],
            1.0,
        )
        a2 = a * a
        denom = q[None, :, :] + mu_grid[:, None, None]
        a2_den = a2[None, :, :] / denom
        a2_den2 = a2[None, :, :] / (denom * denom)
        q_a2_den2 = q[None, :, :] * a2[None, :, :] / (denom * denom)
        A2 = a2_den.sum(axis=2) + owner_a2 / (1.0 + mu_grid[:, None])
        N2 = a2_den2.sum(axis=2) + owner_a2 / ((1.0 + mu_grid[:, None]) ** 2)
        H2 = q_a2_den2.sum(axis=2) + owner_a2 / ((1.0 + mu_grid[:, None]) ** 2)
        shifts = (
            A2 - z * np.sqrt(np.maximum(N2, 0.0))
        ) / np.sqrt(np.maximum(H2, 1e-30))
        num_a = a2.sum(axis=1) + owner_a2
        h_a = (q * a2).sum(axis=1) + owner_a2
        deflection_limit = (
            num_a - z * np.sqrt(np.maximum(num_a, 0.0))
        ) / np.sqrt(np.maximum(h_a, 1e-30))
        return np.maximum(shifts.max(axis=0), deflection_limit)

    for start in range(0, all_combos.shape[0], batch_size):
        batch = all_combos[start:start + batch_size]
        batch_costs = costs[start:start + batch_size]
        pd_values = norm.cdf(evaluate_batch(batch))
        for cost, pd in zip(batch_costs, pd_values):
            key = int(cost)
            if key not in frontier or float(pd) > frontier[key]:
                frontier[key] = float(pd)

    result = sorted(frontier.items())
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


def exact_joint_power_bit_maxmin(
    target_groups: list[list[tuple[int, float]]],
    budget: int,
) -> float:
    """Exact max-min over joint power-bit target frontiers."""
    return exact_joint_maxmin(target_groups, budget)


def exact_joint_power_bit_maxmin_selection(
    target_groups: list[list[tuple[int, float]]],
    budget: int,
) -> tuple[float, list[tuple[int, float]]]:
    """Exact max-min value and the cost-minimal schedule attaining it."""
    return exact_joint_maxmin_selection(target_groups, budget)
