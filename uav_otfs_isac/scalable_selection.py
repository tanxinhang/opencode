"""Scalable exact-threshold max-min selection.

For a fixed threshold `t`, the global max-min problem asks whether each
target can select a report subset with expected P_D at least `t` under the
total bit budget.  For target `q`, let `m_q(t)` be the minimum bit cost of a
subset whose expected P_D is at least `t`.  A global schedule with all values
at least `t` exists if and only if

``sum_q m_q(t) <= B``,

because every feasible target subset costs at least `m_q(t)` and the
independent minima are jointly attainable.  Feasibility is monotone in `t`,
so the maximum feasible threshold can be found by binary search.

The per-target minimization uses branch-and-bound.  For any linear score the
shift is bounded above by the closed-form Cauchy bound
`sqrt(a^T Q^-1 a)` (Lemma 4.7G), so no operating-point monotonicity
assumption is required.  A node is pruned when even that upper bound is
below the threshold, when its current cost already reaches the best known
cost, or when the cheapest remaining reports cannot beat the best known
cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations

import numpy as np
from scipy.stats import norm

from .exact_quota_selection import exact_maxmin_select
from .expected_pd import expected_gaussian_detection_probability
from .fusion import pd_shift_upper_bound
from .models import ExpectedPdSelectionResult, TargetEvidenceModel


def _pd_value(
    model: TargetEvidenceModel,
    scheduled: frozenset[int],
    false_alarm_rate: float,
    pd_mode: str,
    grid: int,
) -> float:
    return float(expected_gaussian_detection_probability(
        model, scheduled, false_alarm_rate, pd_mode=pd_mode, grid=grid,
    ))


def _pd_upper_bound(
    model: TargetEvidenceModel,
    scheduled: frozenset[int],
    false_alarm_rate: float,
) -> float:
    """Closed-form upper bound on the P_D-optimal linear-score value."""
    return float(norm.cdf(pd_shift_upper_bound(
        model.mu0, model.mu1, model.sigma0, model.sigma1, scheduled,
        false_alarm_rate,
    )))


def _greedy_min_cost(
    model: TargetEvidenceModel,
    order: Sequence[int],
    threshold: float,
    false_alarm_rate: float,
    pd_mode: str,
    grid: int,
    budget_cap: float,
    value_cache: dict[frozenset[int], float],
) -> tuple[int, frozenset[int]] | None:
    """Greedy warm start: add reports in the given order until feasible."""
    current = {model.owner}
    current_cost = 0
    for report in order:
        if current_cost >= budget_cap:
            return None
        current.add(report)
        current_cost += int(model.report_bits[report])
        scheduled = frozenset(current)
        if scheduled not in value_cache:
            value_cache[scheduled] = _pd_value(
                model, scheduled, false_alarm_rate, pd_mode, grid
            )
        if value_cache[scheduled] >= threshold - 1e-12:
            return current_cost, scheduled
    return None


def _minimum_cost_bounded(
    model: TargetEvidenceModel,
    cap: int,
    threshold: float,
    false_alarm_rate: float,
    pd_mode: str,
    grid: int,
    value_cache: dict[frozenset[int], float],
    stats: dict | None = None,
) -> tuple[int, frozenset[int]] | None:
    """Exact minimum cost among subsets whose total cost is at most ``cap``."""
    reports = sorted(
        (i for i in range(model.num_uavs) if i != model.owner),
        key=lambda i: (int(model.report_bits[i]), i),
    )
    owner = frozenset([model.owner])
    best: tuple[int, frozenset[int]] | None = None
    local_stats = stats if stats is not None else {}

    def value(scheduled: frozenset[int]) -> float:
        if scheduled not in value_cache:
            value_cache[scheduled] = _pd_value(
                model, scheduled, false_alarm_rate, pd_mode, grid
            )
        return value_cache[scheduled]

    def dfs(index: int, scheduled: frozenset[int], cost: int) -> None:
        nonlocal best
        local_stats["nodes"] = local_stats.get("nodes", 0) + 1
        local_stats["max_depth"] = max(
            local_stats.get("max_depth", 0), index
        )
        if value(scheduled) >= threshold - 1e-12:
            if best is None or cost < best[0]:
                best = (cost, scheduled)
            return
        if index == len(reports) or cost >= cap:
            local_stats["prune_leaf"] = local_stats.get("prune_leaf", 0) + 1
            return
        report = reports[index]
        report_cost = int(model.report_bits[report])
        if cost + report_cost <= cap:
            dfs(index + 1, scheduled | {report}, cost + report_cost)
        dfs(index + 1, scheduled, cost)

    dfs(0, owner, 0)
    return best


def _minimum_cost_bruteforce(
    model: TargetEvidenceModel,
    threshold: float,
    false_alarm_rate: float,
    pd_mode: str,
    grid: int,
) -> tuple[int, frozenset[int]] | None:
    """Exact minimum cost by subset enumeration for small report sets."""
    reports = [
        i for i in range(model.num_uavs) if i != model.owner
    ]
    best: tuple[int, frozenset[int]] | None = None
    for size in range(len(reports) + 1):
        for subset in combinations(reports, size):
            scheduled = frozenset([model.owner, *subset])
            value = _pd_value(
                model, scheduled, false_alarm_rate, pd_mode, grid
            )
            if value < threshold - 1e-12:
                continue
            cost = sum(
                int(model.report_bits[i])
                for i in scheduled
                if i != model.owner
            )
            if best is None or cost < best[0]:
                best = (cost, scheduled)
    return best


def minimum_cost_to_threshold(
    model: TargetEvidenceModel,
    threshold: float,
    false_alarm_rate: float,
    *,
    pd_mode: str = "optimal",
    grid: int = 512,
    max_cost: int | None = None,
    max_exhaustive_reports: int = 14,
    stats: dict | None = None,
) -> tuple[int, frozenset[int]] | None:
    """Minimum bit cost of a subset reaching ``threshold`` expected P_D."""
    if pd_mode not in {"optimal", "deflection"}:
        raise ValueError("pd_mode must be 'optimal' or 'deflection'")
    reports = sorted(
        (i for i in range(model.num_uavs) if i != model.owner),
        key=lambda i: (int(model.report_bits[i]), i),
    )
    if threshold < 0.5 - 1e-9 and len(reports) <= max_exhaustive_reports:
        return _minimum_cost_bruteforce(
            model, threshold, false_alarm_rate, pd_mode, grid
        )
    owner_set = frozenset([model.owner])
    if _pd_value(model, owner_set, false_alarm_rate, pd_mode, grid) >= threshold - 1e-12:
        return 0, owner_set
    all_set = frozenset([model.owner, *reports])
    if _pd_upper_bound(model, all_set, false_alarm_rate) < threshold - 1e-12:
        return None

    value_cache: dict[frozenset[int], float] = {}
    local_stats = stats if stats is not None else {}

    def value(scheduled: frozenset[int]) -> float:
        if scheduled not in value_cache:
            value_cache[scheduled] = _pd_value(
                model, scheduled, false_alarm_rate, pd_mode, grid
            )
        return value_cache[scheduled]

    budget_cap = max_cost if max_cost is not None else np.inf
    best_cost = np.inf
    best_set: frozenset[int] | None = None

    # Warm starts: cost-ascending order and single-report value/cost ratio
    # order both produce feasible upper bounds that tighten the DFS.
    cost_order = list(reports)
    ratio_order = sorted(
        reports,
        key=lambda i: (
            -(
                _pd_value(
                    model,
                    owner_set | {i},
                    false_alarm_rate,
                    pd_mode,
                    grid,
                )
                - value(owner_set)
            )
            / max(int(model.report_bits[i]), 1),
            int(model.report_bits[i]),
            i,
        ),
    )
    for order in (cost_order, ratio_order):
        candidate = _greedy_min_cost(
            model, order, threshold, false_alarm_rate, pd_mode, grid,
            budget_cap, value_cache,
        )
        if candidate is not None and candidate[0] < best_cost:
            best_cost = candidate[0]
            best_set = candidate[1]

    # In low-P_D regimes the Cauchy bound is loose and greedy warm starts may
    # miss the feasible subset.  For small budgets, exact cost-bounded
    # enumeration is both exact and faster than a branch-and-bound whose
    # upper bound prunes weakly.
    if (
        best_set is None
        and threshold < 0.5 - 1e-9
        and budget_cap <= 12
    ):
        candidate = _minimum_cost_bounded(
            model, int(budget_cap), threshold, false_alarm_rate, pd_mode,
            grid, value_cache, stats=local_stats,
        )
        if candidate is not None:
            return candidate
        return None

    # When the greedy upper bound is small, prove minimality by enumerating
    # every subset with cost below that bound instead of exploring the full
    # branch-and-bound tree.
    if best_set is not None and best_cost <= 10:
        cap = 0
        while cap < best_cost:
            candidate = _minimum_cost_bounded(
                model, cap, threshold, false_alarm_rate, pd_mode, grid,
                value_cache, stats=local_stats,
            )
            if candidate is not None:
                best_cost = candidate[0]
                best_set = candidate[1]
                break
            cap += 1
        if best_set is not None and best_cost <= budget_cap:
            return int(best_cost), best_set

    # The DFS search order prioritizes high value-per-cost reports, while the
    # cost prune uses the minimum cost among remaining reports.
    search_order = ratio_order
    min_remaining_cost = [0] * (len(search_order) + 1)
    min_remaining_cost[-1] = np.inf
    for index in range(len(search_order) - 1, -1, -1):
        min_remaining_cost[index] = min(
            int(model.report_bits[search_order[index]]),
            min_remaining_cost[index + 1],
        )

    def dfs(index: int, scheduled: frozenset[int], cost: int) -> None:
        nonlocal best_cost, best_set
        local_stats["nodes"] = local_stats.get("nodes", 0) + 1
        local_stats["max_depth"] = max(
            local_stats.get("max_depth", 0), index
        )
        if value(scheduled) >= threshold - 1e-12:
            if cost < best_cost:
                best_cost = cost
                best_set = scheduled
            return
        if index == len(reports):
            local_stats["prune_leaf"] = local_stats.get("prune_leaf", 0) + 1
            return
        if best_set is not None:
            if cost >= best_cost:
                local_stats["prune_cost"] = (
                    local_stats.get("prune_cost", 0) + 1
                )
                return
            if cost + min_remaining_cost[index] >= best_cost:
                local_stats["prune_cost"] = (
                    local_stats.get("prune_cost", 0) + 1
                )
                return
        remaining = frozenset([model.owner, *search_order[index:]])
        if _pd_upper_bound(
            model, scheduled | remaining, false_alarm_rate
        ) < threshold - 1e-12:
            local_stats["prune_upper"] = (
                local_stats.get("prune_upper", 0) + 1
            )
            return
        report = search_order[index]
        report_cost = int(model.report_bits[report])
        dfs(index + 1, scheduled | {report}, cost + report_cost)
        dfs(index + 1, scheduled, cost)

    dfs(0, owner_set, 0)
    if best_set is None:
        return None
    if best_cost > budget_cap:
        return None
    return int(best_cost), best_set


def scaled_maxmin_select(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    false_alarm_rate: float,
    *,
    qos_pd: Sequence[float] | None = None,
    qos_weights: Sequence[float] | None = None,
    pd_mode: str = "optimal",
    grid: int = 512,
    tolerance: float = 1e-6,
    max_iterations: int = 40,
    max_exhaustive_reports: int = 14,
) -> ExpectedPdSelectionResult:
    """Epsilon-certified max-min selection for larger report sets.

    The result's achieved worst-target value is feasible, and the binary-search
    upper bound certifies that the exact max-min value is within
    ``(upper_bound - achieved_min) / 2`` of the returned value.
    """
    if budget_bits < 0:
        raise ValueError("budget_bits must be nonnegative")
    if pd_mode not in {"optimal", "deflection"}:
        raise ValueError("pd_mode must be 'optimal' or 'deflection'")
    count = len(models)
    qos = np.zeros(count, dtype=float) if qos_pd is None else np.asarray(
        qos_pd, dtype=float
    )
    qos_w = np.ones(count, dtype=float) if qos_weights is None else np.asarray(
        qos_weights, dtype=float
    )
    if qos.shape != (count,) or qos_w.shape != (count,):
        raise ValueError("per-target arrays must have one entry per target")

    if all(model.num_uavs - 1 <= max_exhaustive_reports for model in models):
        exact = exact_maxmin_select(
            models, budget_bits, false_alarm_rate, qos_pd=qos_pd,
            qos_weights=qos_weights, pd_mode=pd_mode, grid=grid,
            max_exhaustive_reports=max_exhaustive_reports,
        )
        return replace(
            exact,
            certificate_upper_bound=float(np.min(exact.expected_pd)),
        )

    def gap(values: Sequence[float]) -> float:
        return float(np.sum(
            qos_w * np.maximum(qos - np.asarray(values), 0.0)
            / np.maximum(qos, 1e-12)
        ))

    owner_values = []
    all_upper_values = []
    for model in models:
        owner = frozenset([model.owner])
        all_reports = frozenset(range(model.num_uavs))
        owner_values.append(_pd_value(
            model, owner, false_alarm_rate, pd_mode, grid
        ))
        all_upper_values.append(_pd_upper_bound(
            model, all_reports, false_alarm_rate
        ))

    def feasible(threshold: float) -> bool:
        total_cost = 0
        for model in models:
            result = minimum_cost_to_threshold(
                model, threshold, false_alarm_rate, pd_mode=pd_mode,
                grid=grid, max_cost=budget_bits - total_cost,
                max_exhaustive_reports=max_exhaustive_reports,
            )
            if result is None:
                return False
            total_cost += result[0]
            if total_cost > budget_bits:
                return False
        return True

    lower = float(np.min(owner_values))
    upper = float(np.min(all_upper_values))
    if feasible(upper):
        lower = upper
    else:
        for _ in range(max_iterations):
            mid = 0.5 * (lower + upper)
            if feasible(mid):
                lower = mid
            else:
                upper = mid

    chosen: list[frozenset[int]] = []
    used = 0
    for model in models:
        result = minimum_cost_to_threshold(
            model, lower, false_alarm_rate, pd_mode=pd_mode, grid=grid,
            max_cost=budget_bits - used,
            max_exhaustive_reports=max_exhaustive_reports,
        )
        if result is None:
            raise RuntimeError("binary-search lower bound became infeasible")
        cost, scheduled = result
        chosen.append(scheduled)
        used += cost

    values = np.asarray([
        _pd_value(model, group, false_alarm_rate, pd_mode, grid)
        for model, group in zip(models, chosen)
    ])
    trace = tuple({
        "target": q,
        "reports": sorted(group),
        "cost_bits": sum(
            int(models[q].report_bits[i])
            for i in group
            if i != models[q].owner
        ),
        "expected_pd": float(values[q]),
    } for q, group in enumerate(chosen))
    return ExpectedPdSelectionResult(
        scheduled=tuple(chosen),
        expected_pd=values,
        used_bits=used,
        normalized_qos_gap=gap(values),
        trace=trace,
        certificate_upper_bound=upper,
    )
