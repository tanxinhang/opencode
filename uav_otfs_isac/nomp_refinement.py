"""NOMP-inspired online max-min refinement for joint power-bit allocation.

The greedy phase follows the winner-take-all marginal rule with an optional
mandatory per-target minimum cover, so every target keeps at least one active
communication/sensing link.  A discrete Newton-style refinement then searches
single power/bit exchanges, within-target atom merges, and redundant-atom
transfers.  A move is accepted only when it improves the lexicographic max-min
vector, so the worst target value never decreases and the loop terminates at a
finite local optimum or at a hard round cap.
"""

from __future__ import annotations

import numpy as np

from .power_split_theory import (
    power_gain_coefficient,
    proportional_target_pd,
)


def leximin_improves(old_values, new_values) -> bool:
    """True when the sorted target vector improves lexicographically."""
    a = np.sort(np.asarray(old_values, dtype=float))
    b = np.sort(np.asarray(new_values, dtype=float))
    for x, y in zip(a, b):
        if not np.isclose(x, y, atol=1e-12, rtol=0.0):
            return y > x
    return False


def initial_min_cover(scenario, budget, *, max_bits: int = 2):
    """Activate one best report per target when the budget allows it."""
    reports = len(scenario[0]) - 1
    powers = [np.zeros(reports, dtype=int) for _ in scenario]
    bits = [np.zeros(reports, dtype=int) for _ in scenario]
    used = 0
    if 2 * len(scenario) > budget:
        return powers, bits, used
    for q, target in enumerate(scenario):
        deltas = np.asarray(target[1:], dtype=float)
        coefficients = np.asarray([
            power_gain_coefficient(float(delta), 1, 0.0, 1.0)
            for delta in deltas
        ])
        winner = int(np.argmax(coefficients))
        powers[q][winner] = 1
        bits[q][winner] = 1
        used += 2
    return powers, bits, used


def target_scores(scenario, powers, bits, grid: int = 16):
    """Per-target P_D under the current allocation."""
    return [
        float(proportional_target_pd(
            float(target[0]),
            target[1:],
            powers[q],
            bits[q],
            grid,
        ))
        for q, target in enumerate(scenario)
    ]


def _active_reports(bits):
    return [r for r in range(len(bits)) if bits[r] > 0]


def _winner_index(target, bits):
    active = _active_reports(bits)
    if not active:
        return None
    return max(
        active,
        key=lambda r: power_gain_coefficient(
            float(target[r + 1]), int(bits[r]), 0.0, 1.0
        ),
    )


def _add_freed_units(
    power_row,
    bit_row,
    report_index,
    freed,
    *,
    max_power,
    max_bits,
):
    remaining = freed
    while remaining > 0 and power_row[report_index] < max_power:
        power_row[report_index] += 1
        remaining -= 1
    while remaining > 0 and bit_row[report_index] < max_bits:
        bit_row[report_index] += 1
        remaining -= 1
    return remaining


def _iter_candidates(
    scenario,
    powers,
    bits,
    *,
    max_power,
    max_bits,
):
    """Yield feasible single-exchange power/bit/atom moves."""
    q_count = len(scenario)
    reports = len(scenario[0]) - 1
    for q in range(q_count):
        target_q = scenario[q]
        active_q = _active_reports(bits[q])
        winner_q = _winner_index(target_q, bits[q])
        for s in range(reports):
            if winner_q is not None and s != winner_q:
                if powers[q][s] > 0 and powers[q][winner_q] < max_power:
                    new_p = [row.copy() for row in powers]
                    new_b = [row.copy() for row in bits]
                    new_p[q][s] -= 1
                    new_p[q][winner_q] += 1
                    yield new_p, new_b
                if (
                    bits[q][s] > 0
                    and bits[q][winner_q] < max_bits
                    and (len(active_q) > 1 or bits[q][s] > 1)
                ):
                    new_p = [row.copy() for row in powers]
                    new_b = [row.copy() for row in bits]
                    new_b[q][s] -= 1
                    new_b[q][winner_q] += 1
                    yield new_p, new_b
                if len(active_q) >= 2:
                    freed = int(powers[q][s] + bits[q][s])
                    new_p = [row.copy() for row in powers]
                    new_b = [row.copy() for row in bits]
                    new_p[q][s] = 0
                    new_b[q][s] = 0
                    remaining = _add_freed_units(
                        new_p[q],
                        new_b[q],
                        winner_q,
                        freed,
                        max_power=max_power,
                        max_bits=max_bits,
                    )
                    if remaining == 0:
                        yield new_p, new_b
            for d in range(q_count):
                if d == q:
                    continue
                target_d = scenario[d]
                active_d = _active_reports(bits[d])
                winner_d = _winner_index(target_d, bits[d])
                if winner_d is None:
                    continue
                if powers[q][s] > 0 and powers[d][winner_d] < max_power:
                    new_p = [row.copy() for row in powers]
                    new_b = [row.copy() for row in bits]
                    new_p[q][s] -= 1
                    new_p[d][winner_d] += 1
                    yield new_p, new_b
                if (
                    bits[q][s] > 0
                    and bits[d][winner_d] < max_bits
                    and (len(active_q) > 1 or bits[q][s] > 1)
                ):
                    new_p = [row.copy() for row in powers]
                    new_b = [row.copy() for row in bits]
                    new_b[q][s] -= 1
                    new_b[d][winner_d] += 1
                    yield new_p, new_b
                if len(active_q) >= 2:
                    freed = int(powers[q][s] + bits[q][s])
                    new_p = [row.copy() for row in powers]
                    new_b = [row.copy() for row in bits]
                    new_p[q][s] = 0
                    new_b[q][s] = 0
                    remaining = _add_freed_units(
                        new_p[d],
                        new_b[d],
                        winner_d,
                        freed,
                        max_power=max_power,
                        max_bits=max_bits,
                    )
                    if remaining == 0:
                        yield new_p, new_b


def maxmin_refine(
    scenario,
    powers,
    bits,
    *,
    max_power,
    max_bits: int = 2,
    max_rounds: int = 100,
    grid: int = 16,
):
    """NOMP-style discrete refinement with a hard iteration cap."""
    rounds_used = 0
    for _ in range(max_rounds):
        old_values = np.sort(target_scores(scenario, powers, bits, grid))
        best = None
        for candidate in _iter_candidates(
            scenario,
            powers,
            bits,
            max_power=max_power,
            max_bits=max_bits,
        ):
            new_values = np.sort(target_scores(
                scenario, candidate[0], candidate[1], grid
            ))
            if leximin_improves(old_values, new_values):
                if best is None or new_values.tolist() > best[0]:
                    best = (new_values.tolist(), candidate)
        if best is None:
            break
        _, (powers, bits) = best
        rounds_used += 1
    return powers, bits, rounds_used


def wta_greedy_joint_multi(
    scenario,
    budget,
    *,
    min_cover: bool = False,
    max_bits: int = 2,
    max_power=None,
    grid: int = 16,
):
    """Online WTA greedy allocation with optional per-target minimum cover."""
    if max_power is None:
        max_power = int(budget)
    reports = len(scenario[0]) - 1
    if min_cover:
        powers, bits, used = initial_min_cover(
            scenario, budget, max_bits=max_bits
        )
    else:
        powers = [np.zeros(reports, dtype=int) for _ in scenario]
        bits = [np.zeros(reports, dtype=int) for _ in scenario]
        used = 0
    steps = 0

    def current_mean():
        return float(np.mean(target_scores(scenario, powers, bits, grid)))

    while True:
        mean_before = current_mean()
        best = None
        for q, target in enumerate(scenario):
            active = _active_reports(bits[q])
            for r in range(reports):
                if bits[q][r] > 0 or used + 2 > budget:
                    continue
                old_b, old_p = bits[q].copy(), powers[q].copy()
                bits[q][r] = 1
                powers[q][r] = 1
                gain = current_mean() - mean_before
                bits[q], powers[q] = old_b, old_p
                if gain > 0:
                    key = (gain / 2.0, gain, q, "activate", r)
                    if best is None or key > best[0]:
                        best = (key, q, "activate", r)
            for r in active:
                if bits[q][r] >= max_bits or used + 1 > budget:
                    continue
                old_b = bits[q].copy()
                bits[q][r] += 1
                gain = current_mean() - mean_before
                bits[q] = old_b
                if gain > 0:
                    key = (gain, gain, q, "bit", r)
                    if best is None or key > best[0]:
                        best = (key, q, "bit", r)
            if active:
                coefficients = np.asarray([
                    power_gain_coefficient(
                        float(target[r + 1]), int(bits[q][r]), 0.0, 1.0
                    )
                    for r in active
                ])
                winner = active[int(np.argmax(coefficients))]
                if powers[q][winner] < max_power and used + 1 <= budget:
                    old_p = powers[q].copy()
                    powers[q][winner] += 1
                    gain = current_mean() - mean_before
                    powers[q] = old_p
                    if gain > 0:
                        key = (gain, gain, q, "power", winner)
                        if best is None or key > best[0]:
                            best = (key, q, "power", winner)
        if best is None:
            break
        _, q, action, index = best
        if action == "activate":
            bits[q][index] = 1
            powers[q][index] = 1
            used += 2
        elif action == "bit":
            bits[q][index] += 1
            used += 1
        else:
            powers[q][index] += 1
            used += 1
        steps += 1

    return {
        "powers": powers,
        "bits": bits,
        "worst_pd": float(min(target_scores(scenario, powers, bits, grid))),
        "used": int(sum(
            int(powers[q].sum()) + int(bits[q].sum())
            for q in range(len(scenario))
        )),
        "steps": steps,
    }


def nomp_wta_greedy_joint_multi(
    scenario,
    budget,
    *,
    max_bits: int = 2,
    max_power=None,
    max_rounds: int = 100,
    grid: int = 16,
):
    """WTA greedy with minimum cover, followed by NOMP-style refinement."""
    if max_power is None:
        max_power = int(budget)
    greedy = wta_greedy_joint_multi(
        scenario,
        budget,
        min_cover=True,
        max_bits=max_bits,
        max_power=max_power,
        grid=grid,
    )
    powers, bits, refine_rounds = maxmin_refine(
        scenario,
        greedy["powers"],
        greedy["bits"],
        max_power=max_power,
        max_bits=max_bits,
        max_rounds=max_rounds,
        grid=grid,
    )
    return {
        "powers": powers,
        "bits": bits,
        "worst_pd": float(min(target_scores(scenario, powers, bits, grid))),
        "used": int(sum(
            int(powers[q].sum()) + int(bits[q].sum())
            for q in range(len(scenario))
        )),
        "greedy_steps": greedy["steps"],
        "refine_rounds": refine_rounds,
    }
