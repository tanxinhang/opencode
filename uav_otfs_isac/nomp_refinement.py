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
from .robust_joint_power_bit import per_report_communication_target_pd


def leximin_improves(old_values, new_values) -> bool:
    """True when the sorted target vector improves lexicographically."""
    a = np.sort(np.asarray(old_values, dtype=float))
    b = np.sort(np.asarray(new_values, dtype=float))
    for x, y in zip(a, b):
        if not np.isclose(x, y, atol=1e-12, rtol=0.0):
            return y > x
    return False


def _parse_target(target, flip_probability=0.0, success_probability=1.0):
    """Return (owner, deltas, flips, successes) for either scenario format."""
    if isinstance(target, tuple) and len(target) == 4:
        return (
            float(target[0]),
            np.asarray(target[1], dtype=float),
            np.asarray(target[2], dtype=float),
            np.asarray(target[3], dtype=float),
        )
    row = np.asarray(target, dtype=float)
    owner = float(row[0])
    deltas = row[1:]
    flips = np.full(deltas.size, float(flip_probability))
    successes = np.full(deltas.size, float(success_probability))
    return owner, deltas, flips, successes


def _report_count(target):
    return int(_parse_target(target)[1].size)


def initial_min_cover(
    scenario,
    budget,
    *,
    max_bits: int = 2,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    grid: int = 16,
):
    """Activate one best report per target when the budget allows it."""
    reports = _report_count(scenario[0])
    powers = [np.zeros(reports, dtype=int) for _ in scenario]
    bits = [np.zeros(reports, dtype=int) for _ in scenario]
    used = 0
    if 2 * len(scenario) > budget:
        return powers, bits, used
    for q, target in enumerate(scenario):
        zero_powers = np.zeros(reports, dtype=float)
        zero_bits = np.zeros(reports, dtype=int)
        baseline = _target_pd(
            target,
            zero_powers,
            zero_bits,
            grid,
            flip_probability,
            success_probability,
        )
        best_value = baseline
        best_winner = None
        for r in range(reports):
            candidate_powers = zero_powers.copy()
            candidate_bits = zero_bits.copy()
            candidate_powers[r] = 1
            candidate_bits[r] = 1
            candidate = _target_pd(
                target,
                candidate_powers,
                candidate_bits,
                grid,
                flip_probability,
                success_probability,
            )
            if candidate > best_value:
                best_value = candidate
                best_winner = r
        if best_winner is not None and best_value > baseline + 1e-12:
            winner = best_winner
            powers[q][winner] = 1
            bits[q][winner] = 1
            used += 2
    return powers, bits, used


def target_scores(
    scenario,
    powers,
    bits,
    grid: int = 16,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
):
    """Per-target P_D under the current allocation."""
    return [
        float(_target_pd(
            target,
            powers[q],
            bits[q],
            grid,
            flip_probability,
            success_probability,
        ))
        for q, target in enumerate(scenario)
    ]


def _target_pd(
    target,
    powers,
    bits,
    grid,
    flip_probability,
    success_probability,
):
    owner_delta, deltas, flips, successes = _parse_target(
        target, flip_probability, success_probability
    )
    if np.all(flips == 0.0) and np.all(successes == 1.0):
        return proportional_target_pd(owner_delta, deltas, powers, bits, grid)
    return per_report_communication_target_pd(
        owner_delta,
        deltas,
        powers,
        bits,
        flips,
        successes,
        grid,
    )


def _active_reports(bits):
    return [r for r in range(len(bits)) if bits[r] > 0]


def _winner_index(
    target,
    bits,
    powers,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    grid: int = 16,
):
    active = _active_reports(bits)
    if not active:
        return None
    owner, deltas, flips, successes = _parse_target(
        target, flip_probability, success_probability
    )
    if not (np.all(flips == 0.0) and np.all(successes == 1.0)):
        base = _target_pd(
            (owner, deltas, flips, successes),
            powers,
            bits,
            grid,
            flip_probability,
            success_probability,
        )
        best = None
        for r in active:
            candidate_p = np.asarray(powers, dtype=float).copy()
            candidate_b = np.asarray(bits, dtype=int).copy()
            candidate_p[r] += 1
            value = _target_pd(
                (owner, deltas, flips, successes),
                candidate_p,
                candidate_b,
                grid,
                flip_probability,
                success_probability,
            ) - base
            if best is None or value > best[0]:
                best = (float(value), r)
        return int(best[1])
    return max(
        active,
        key=lambda r: power_gain_coefficient(
            float(deltas[r]),
            int(bits[r]),
            float(flips[r]),
            float(successes[r]),
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
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    grid: int = 16,
):
    """Yield feasible single-exchange power/bit/atom moves."""
    q_count = len(scenario)
    reports = _report_count(scenario[0])
    for q in range(q_count):
        target_q = scenario[q]
        active_q = _active_reports(bits[q])
        winner_q = _winner_index(
            target_q,
            bits[q],
            powers[q],
            flip_probability,
            success_probability,
            grid,
        )
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
                active_d = _active_reports(bits[d])
                for dd in active_d:
                    if powers[q][s] > 0 and powers[d][dd] < max_power:
                        new_p = [row.copy() for row in powers]
                        new_b = [row.copy() for row in bits]
                        new_p[q][s] -= 1
                        new_p[d][dd] += 1
                        yield new_p, new_b
                    if (
                        bits[q][s] > 0
                        and bits[d][dd] < max_bits
                        and (len(active_q) > 1 or bits[q][s] > 1)
                    ):
                        new_p = [row.copy() for row in powers]
                        new_b = [row.copy() for row in bits]
                        new_b[q][s] -= 1
                        new_b[d][dd] += 1
                        yield new_p, new_b
                if len(active_q) >= 2:
                    freed = int(powers[q][s] + bits[q][s])
                    for dd in active_d:
                        new_p = [row.copy() for row in powers]
                        new_b = [row.copy() for row in bits]
                        new_p[q][s] = 0
                        new_b[q][s] = 0
                        remaining = _add_freed_units(
                            new_p[d],
                            new_b[d],
                            dd,
                            freed,
                            max_power=max_power,
                            max_bits=max_bits,
                        )
                        if remaining == 0:
                            yield new_p, new_b
                    for dd in range(reports):
                        if bits[d][dd] > 0:
                            continue
                        new_p = [row.copy() for row in powers]
                        new_b = [row.copy() for row in bits]
                        new_p[q][s] = 0
                        new_b[q][s] = 0
                        if freed >= 2:
                            new_p[d][dd] = 1
                            new_b[d][dd] = 1
                            remaining = freed - 2
                            remaining = _add_freed_units(
                                new_p[d],
                                new_b[d],
                                dd,
                                remaining,
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
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
):
    """NOMP-style discrete refinement with a hard iteration cap."""
    rounds_used = 0
    for _ in range(max_rounds):
        old_values = np.sort(target_scores(
            scenario,
            powers,
            bits,
            grid,
            flip_probability,
            success_probability,
        ))
        best = None
        for candidate in _iter_candidates(
            scenario,
            powers,
            bits,
            max_power=max_power,
            max_bits=max_bits,
            flip_probability=flip_probability,
            success_probability=success_probability,
            grid=grid,
        ):
            new_values = np.sort(target_scores(
                scenario,
                candidate[0],
                candidate[1],
                grid,
                flip_probability,
                success_probability,
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
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
):
    """Online WTA greedy allocation with optional per-target minimum cover."""
    if max_power is None:
        max_power = int(budget)
    reports = _report_count(scenario[0])
    if min_cover:
        powers, bits, used = initial_min_cover(
            scenario,
            budget,
            max_bits=max_bits,
            flip_probability=flip_probability,
            success_probability=success_probability,
            grid=grid,
        )
    else:
        powers = [np.zeros(reports, dtype=int) for _ in scenario]
        bits = [np.zeros(reports, dtype=int) for _ in scenario]
        used = 0
    steps = 0

    def current_mean():
        return float(np.mean(target_scores(
            scenario,
            powers,
            bits,
            grid,
            flip_probability,
            success_probability,
        )))

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
                winner = _winner_index(
                    target,
                    bits[q],
                    powers[q],
                    flip_probability,
                    success_probability,
                    grid,
                )
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
        "worst_pd": float(min(target_scores(
            scenario,
            powers,
            bits,
            grid,
            flip_probability,
            success_probability,
        ))),
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
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
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
        flip_probability=flip_probability,
        success_probability=success_probability,
    )
    powers, bits, refine_rounds = maxmin_refine(
        scenario,
        greedy["powers"],
        greedy["bits"],
        max_power=max_power,
        max_bits=max_bits,
        max_rounds=max_rounds,
        grid=grid,
        flip_probability=flip_probability,
        success_probability=success_probability,
    )
    return {
        "powers": powers,
        "bits": bits,
        "worst_pd": float(min(target_scores(
            scenario,
            powers,
            bits,
            grid,
            flip_probability,
            success_probability,
        ))),
        "used": int(sum(
            int(powers[q].sum()) + int(bits[q].sum())
            for q in range(len(scenario))
        )),
        "greedy_steps": greedy["steps"],
        "refine_rounds": refine_rounds,
    }
