"""Exact centralized/distributed architecture switching.

Both centralized soft fusion and peer consensus are feasible detection
architectures at the same global false-alarm rate.  The centralized branch
spends report bits; the peer branch spends zero report bits but requires
local decisions and a fully connected voting layer.  The mode selector
chooses the branch with the larger exact worst-target P_D, or uses a fixed
report-budget threshold as a practical substitute.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .expected_pd import expected_gaussian_detection_probability
from .models import TargetEvidenceModel


def exact_architecture_switch(
    soft_worst_pd: float,
    peer_worst_pd: float,
    tolerance: float = 1e-12,
) -> str:
    """Return ``"peer"`` when peer majority has the higher exact P_D."""
    if peer_worst_pd > soft_worst_pd + tolerance:
        return "peer"
    return "soft"


def fixed_budget_architecture_switch(
    report_budget_bits: int,
    threshold_bits: int = 10,
) -> str:
    """Return the practical mode for a report budget.

    The default threshold is the empirically observed crossover in the
    N=128, interference audit: with fewer than 10 report bits the soft branch
    cannot beat peer consensus, while with 10 or more bits centralized soft
    wins.  The threshold is a design parameter, not a universal constant.
    """
    if report_budget_bits < threshold_bits:
        return "peer"
    return "soft"


def selected_architecture_pd(
    soft_worst_pd: float,
    peer_worst_pd: float,
    mode: str,
) -> float:
    """P_D of the selected architecture for the given mode."""
    if mode == "peer":
        return float(peer_worst_pd)
    if mode == "soft":
        return float(soft_worst_pd)
    raise ValueError("mode must be 'peer' or 'soft'")


def target_wise_architecture_switch(
    soft_pds: list[float],
    peer_pds: list[float],
) -> tuple[list[str], list[float]]:
    """Select the better architecture separately for each target.

    Returns ``(modes, values)`` with
    ``values[q] = max(soft_pds[q], peer_pds[q])``.  The resulting
    worst-target P_D is never below the global-mode switch because

    ``min_q max(a_q, b_q) >= max(min_q a_q, min_q b_q)``.
    """
    if len(soft_pds) != len(peer_pds):
        raise ValueError("one soft and one peer P_D is required per target")
    modes = []
    values = []
    for soft_pd, peer_pd in zip(soft_pds, peer_pds):
        if peer_pd > soft_pd:
            modes.append("peer")
        else:
            modes.append("soft")
        values.append(max(soft_pd, peer_pd))
    return modes, values


def reallocate_soft_report_bits(
    models: Sequence[TargetEvidenceModel],
    modes: Sequence[str],
    scheduled: Sequence[Iterable[int]],
    report_budget_bits: int,
    false_alarm_rate: float,
    *,
    grid: int = 512,
    tolerance: float = 1e-12,
) -> tuple[list[frozenset[int]], list[float], int]:
    """Greedily add soft reports to centralized targets within the budget.

    Peer-selected targets spend zero report bits, so their previous soft
    schedules are freed.  This routine keeps every centralized target's
    current schedule and only adds reports while ``used <=
    report_budget_bits``.  The per-target expected P_D is therefore
    nondecreasing and the worst-target P_D cannot decrease.
    """
    if len(models) != len(modes) or len(models) != len(scheduled):
        raise ValueError("models, modes, and scheduled must have equal length")
    current = [frozenset(group) for group in scheduled]
    quality = [
        expected_gaussian_detection_probability(
            model, current[q], false_alarm_rate, grid=grid,
        )
        for q, model in enumerate(models)
    ]

    def cost(q: int, uav: int) -> int:
        return int(models[q].report_bits[uav])

    used = sum(
        cost(q, uav)
        for q in range(len(models))
        if modes[q] == "soft"
        for uav in current[q]
        if uav != models[q].owner
    )
    while True:
        remaining = report_budget_bits - used
        if remaining <= 0:
            break
        best = None
        for q, model in enumerate(models):
            if modes[q] != "soft":
                continue
            for uav in range(model.num_uavs):
                if uav == model.owner or uav in current[q]:
                    continue
                bit_cost = cost(q, uav)
                if bit_cost <= 0 or bit_cost > remaining:
                    continue
                trial_quality = expected_gaussian_detection_probability(
                    model, current[q] | {uav}, false_alarm_rate, grid=grid,
                )
                gain = trial_quality - quality[q]
                if gain <= tolerance:
                    continue
                score = gain / bit_cost
                candidate = (score, gain, -bit_cost, q, uav)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            break
        _, gain, _, q, uav = best
        current[q] = current[q] | {uav}
        quality[q] += gain
        used += cost(q, uav)
    return current, quality, used


def two_sided_mode_ascent(
    models: Sequence[TargetEvidenceModel],
    peer_pds: Sequence[float],
    scheduled: Sequence[Iterable[int]],
    report_budget_bits: int,
    false_alarm_rate: float,
    *,
    grid: int = 512,
    tolerance: float = 1e-12,
) -> tuple[list[str], list[frozenset[int]], list[float], int]:
    """Alternating soft-reallocation and peer-to-soft mode upgrade.

    Starting from any feasible target-wise mode selection, the routine (i)
    reallocates freed bits to centralized targets and (ii) tries to upgrade
    a limiting peer target back to centralized soft by spending unused
    report bits on its original schedule.  A peer target is switched only
    when it currently attains the worst P_D, no other target remains at or
    below the old worst value, and the upgraded soft P_D strictly raises the
    system minimum; otherwise the trial additions are discarded.  Every
    accepted step is monotone, so the worst-target P_D is nondecreasing and
    the number of peer-to-soft switches is at most the number of targets.
    """
    if len(models) != len(peer_pds) or len(models) != len(scheduled):
        raise ValueError("models, peer_pds, and scheduled must align")
    current = [frozenset(group) for group in scheduled]
    quality = [
        expected_gaussian_detection_probability(
            model, current[q], false_alarm_rate, grid=grid,
        )
        for q, model in enumerate(models)
    ]
    modes = [
        "peer" if peer_pds[q] > quality[q] + tolerance else "soft"
        for q in range(len(models))
    ]
    used = sum(
        int(models[q].report_bits[uav])
        for q in range(len(models))
        if modes[q] == "soft"
        for uav in current[q]
        if uav != models[q].owner
    )

    while True:
        before_used = used
        before_quality = tuple(quality)
        current, quality, used = reallocate_soft_report_bits(
            models, modes, current, report_budget_bits, false_alarm_rate,
            grid=grid, tolerance=tolerance,
        )
        soft_changed = (
            before_used != used or tuple(quality) != before_quality
        )

        switched = False
        current_worst = min(
            peer_pds[q] if modes[q] == "peer" else quality[q]
            for q in range(len(models))
        )
        for q in range(len(models)):
            if modes[q] != "peer":
                continue
            if peer_pds[q] > current_worst + tolerance:
                continue
            other_values = [
                peer_pds[q2] if modes[q2] == "peer" else quality[q2]
                for q2 in range(len(models))
                if q2 != q
            ]
            required_min = max(
                current_worst,
                min(other_values) if other_values else current_worst,
            )
            available = report_budget_bits - used
            if available <= 0:
                continue
            trial_schedule = set(current[q])
            trial_quality = quality[q]
            trial_cost = 0
            while trial_cost < available:
                remaining = available - trial_cost
                best = None
                model = models[q]
                for uav in range(model.num_uavs):
                    if uav == model.owner or uav in trial_schedule:
                        continue
                    bit_cost = int(model.report_bits[uav])
                    if bit_cost <= 0 or bit_cost > remaining:
                        continue
                    candidate_quality = expected_gaussian_detection_probability(
                        model, trial_schedule | {uav}, false_alarm_rate,
                        grid=grid,
                    )
                    gain = candidate_quality - trial_quality
                    if gain <= tolerance:
                        continue
                    score = gain / bit_cost
                    candidate = (score, gain, -bit_cost, uav)
                    if best is None or candidate > best:
                        best = candidate
                if best is None:
                    break
                _, gain, _, uav = best
                trial_schedule.add(uav)
                trial_quality += gain
                trial_cost += int(model.report_bits[uav])
                if trial_quality > peer_pds[q] + tolerance:
                    break
            if trial_quality > required_min + tolerance:
                current[q] = frozenset(trial_schedule)
                quality[q] = trial_quality
                modes[q] = "soft"
                used += trial_cost
                current_worst = min(
                    peer_pds[q2] if modes[q2] == "peer" else quality[q2]
                    for q2 in range(len(models))
                )
                switched = True

        if not soft_changed and not switched:
            break
    return modes, current, quality, used
