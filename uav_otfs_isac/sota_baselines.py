"""SOTA-style baseline schedules and exact hard-decision fusion metrics.

The soft baselines use static deflection Top-K and uniform per-target soft
report allocation with deflection-optimal fusion.  The hard-decision
baseline sends one bit per scheduled report and fuses by counting at the
fusion center.  Its count distribution is computed exactly under the
independent, common-state, or grouped reception law, so the comparison does
not hide correlated erasure effects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import norm

from .baselines import (
    independent_post_report_score,
    ranked_baseline,
    sensing_quality_score,
)
from .expected_pd import expected_gaussian_detection_probability
from .models import TargetEvidenceModel


def _count_distribution(probabilities: Sequence[float]) -> np.ndarray:
    """Exact Poisson-binomial PMF over a sequence of independent votes."""
    probabilities = np.asarray(probabilities, dtype=float)
    dp = np.zeros(probabilities.size + 1, dtype=float)
    dp[0] = 1.0
    for probability in probabilities:
        probability = float(np.clip(probability, 0.0, 1.0))
        new = np.zeros_like(dp)
        new[:-1] += dp[:-1] * (1.0 - probability)
        new[1:] += dp[:-1] * probability
        dp = new
    return dp


def exact_counting_feasible(
    p0: Sequence[float],
    p1: Sequence[float],
    false_alarm_rate: float,
    target_pd: float,
) -> bool:
    """Exact Poisson-binomial majority feasibility for a voter sequence."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if p0.size == 0:
        return False
    pmf0 = _count_distribution(p0)
    pmf1 = _count_distribution(p1)
    best_pd = 0.0
    for candidate in range(1, p0.size + 2):
        pfa = float(np.sum(pmf0[candidate:]))
        if pfa <= false_alarm_rate + 1e-12:
            best_pd = max(best_pd, float(np.sum(pmf1[candidate:])))
    return best_pd >= target_pd - 1e-12


def majority_feasibility_trace(
    p0: Sequence[float],
    p1: Sequence[float],
    false_alarm_rate: float,
    target_pd: float,
) -> list[bool]:
    """Feasibility of every prefix of the voter sequence."""
    trace = []
    for size in range(1, len(p0) + 1):
        trace.append(exact_counting_feasible(
            p0[:size], p1[:size], false_alarm_rate, target_pd
        ))
    return trace


def exact_min_majority_uavs(
    p0: Sequence[float],
    p1: Sequence[float],
    false_alarm_rate: float,
    target_pd: float,
) -> int | None:
    """Smallest prefix size at which majority feasibility holds."""
    trace = majority_feasibility_trace(
        p0, p1, false_alarm_rate, target_pd
    )
    for index, feasible in enumerate(trace):
        if feasible:
            return index + 1
    return None


def hard_decision_local_probabilities(
    model: TargetEvidenceModel,
    uav: int,
    local_false_alarm_rate: float,
) -> tuple[float, float]:
    """Return (H0, H1) one-bit decision probabilities after the BSC."""
    threshold = (
        model.mu0[uav]
        + np.sqrt(max(model.sigma0[uav, uav], 1e-30))
        * norm.ppf(1.0 - local_false_alarm_rate)
    )
    q0 = float(norm.sf(
        (threshold - model.mu0[uav]) / np.sqrt(max(model.sigma0[uav, uav], 1e-30))
    ))
    q1 = float(norm.sf(
        (threshold - model.mu1[uav]) / np.sqrt(max(model.sigma1[uav, uav], 1e-30))
    ))
    flip = float(model.bit_flip_prob[uav])
    p0 = q0 * (1.0 - flip) + (1.0 - q0) * flip
    p1 = q1 * (1.0 - flip) + (1.0 - q1) * flip
    return p0, p1


def hard_decision_fusion(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    false_alarm_rate: float,
    local_false_alarm_rate: float = 0.1,
) -> dict[str, float | int]:
    """Exact counting-fusion P_D/P_FA for one target under the reception law."""
    reports = sorted(set(scheduled) - {model.owner})
    per_uav = {
        uav: hard_decision_local_probabilities(
            model, uav, local_false_alarm_rate
        )
        for uav in reports + [model.owner]
    }
    pfa_pmf = np.zeros(len(reports) + 1, dtype=float)
    pd_pmf = np.zeros(len(reports) + 1, dtype=float)

    def accumulate(weight: float, received: Sequence[int]) -> None:
        nonlocal pfa_pmf, pd_pmf
        p0 = [per_uav[uav][0] for uav in received]
        p1 = [per_uav[uav][1] for uav in received]
        pfa_pmf += weight * _count_distribution(p0)
        pd_pmf += weight * _count_distribution(p1)

    if model.reception_state_probabilities is not None:
        for state_weight, conditional in zip(
            model.reception_state_probabilities,
            model.conditional_success_probabilities,
        ):
            received = [model.owner] + [
                uav for uav in reports if conditional[uav] > 0.0
            ]
            # A report contributes a vote only when both received and decided
            # positive; conditional on the state these events are independent.
            effective_p0 = [
                float(conditional[uav]) * per_uav[uav][0]
                for uav in reports
            ]
            effective_p1 = [
                float(conditional[uav]) * per_uav[uav][1]
                for uav in reports
            ]
            pfa_pmf += state_weight * _count_distribution(
                [per_uav[model.owner][0]] + effective_p0
            )
            pd_pmf += state_weight * _count_distribution(
                [per_uav[model.owner][1]] + effective_p1
            )
        received = [model.owner]
    elif model.reception_patterns is not None:
        for pattern, pattern_weight in zip(
            model.reception_patterns, model.pattern_probabilities
        ):
            received = [model.owner] + [
                uav for uav in reports if pattern[uav] == 1
            ]
            accumulate(float(pattern_weight), received)
    else:
        received = [model.owner] + reports
        p0 = [per_uav[model.owner][0]] + [
            float(model.success_prob[uav]) * per_uav[uav][0]
            for uav in reports
        ]
        p1 = [per_uav[model.owner][1]] + [
            float(model.success_prob[uav]) * per_uav[uav][1]
            for uav in reports
        ]
        pfa_pmf = _count_distribution(p0)
        pd_pmf = _count_distribution(p1)

    threshold = int(len(reports) + 1)
    for candidate in range(1, len(reports) + 2):
        if float(np.sum(pfa_pmf[candidate:])) <= false_alarm_rate + 1e-12:
            threshold = candidate
            break
    pfa = float(np.sum(pfa_pmf[threshold:]))
    pd = float(np.sum(pd_pmf[threshold:]))
    return {
        "pd": pd,
        "pfa": pfa,
        "threshold_votes": threshold,
        "scheduled_reports": len(reports),
        "feasible": pfa <= false_alarm_rate + 1e-9,
    }


def optimized_hard_decision_fusion(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    false_alarm_rate: float,
    local_false_alarm_options: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Distributed hard-decision fusion with optimized local P_FA.

    For each candidate local false-alarm rate, the exact counting P_FA/P_D is
    evaluated and the smallest vote threshold meeting the global P_FA is
    used.  The returned rule maximizes P_D over the candidate local
    thresholds, so the local threshold is a designed distributed parameter
    rather than a fixed constant.
    """
    if local_false_alarm_options is None:
        options = tuple(
            float(value) for value in np.geomspace(0.005, 0.5, 20)
        ) + (0.1,)
    else:
        options = tuple(float(value) for value in local_false_alarm_options)
    best = None
    for local_false_alarm_rate in options:
        result = hard_decision_fusion(
            model, scheduled, false_alarm_rate,
            local_false_alarm_rate=local_false_alarm_rate,
        )
        if float(result["pfa"]) > false_alarm_rate + 1e-9:
            continue
        if best is None or float(result["pd"]) > float(best["pd"]):
            best = dict(result)
            best["local_false_alarm_rate"] = local_false_alarm_rate
    if best is None:
        return {
            "pd": 0.0,
            "pfa": 0.0,
            "threshold_votes": 0,
            "scheduled_reports": 0,
            "local_false_alarm_rate": None,
        }
    return best


def peer_majority_fusion(
    model: TargetEvidenceModel,
    false_alarm_rate: float,
    local_false_alarm_options: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Fully distributed majority fusion without an owner fusion center.

    Every UAV makes a local 1-bit decision about the target.  The target is
    declared present when at least ``K`` of the ``M`` UAVs vote positive.
    The local P_FA and vote threshold are optimized under the global P_FA
    constraint.  This stage removes owner fusion, report links, and global
    scheduling.
    """
    if local_false_alarm_options is None:
        options = tuple(
            float(value) for value in np.geomspace(0.005, 0.5, 20)
        ) + (0.1,)
    else:
        options = tuple(float(value) for value in local_false_alarm_options)
    best = None
    for alpha in options:
        p0 = np.zeros(model.num_uavs, dtype=float)
        p1 = np.zeros(model.num_uavs, dtype=float)
        threshold = (
            model.mu0
            + np.sqrt(np.maximum(np.diag(model.sigma0), 1e-30))
            * norm.ppf(1.0 - alpha)
        )
        q0 = norm.sf(
            (threshold - model.mu0)
            / np.sqrt(np.maximum(np.diag(model.sigma0), 1e-30))
        )
        q1 = norm.sf(
            (threshold - model.mu1)
            / np.sqrt(np.maximum(np.diag(model.sigma1), 1e-30))
        )
        p0_pmf = _count_distribution(q0)
        p1_pmf = _count_distribution(q1)
        vote_threshold = int(model.num_uavs + 1)
        for candidate in range(1, model.num_uavs + 2):
            if float(np.sum(p0_pmf[candidate:])) <= false_alarm_rate + 1e-12:
                vote_threshold = candidate
                break
        pfa = float(np.sum(p0_pmf[vote_threshold:]))
        pd = float(np.sum(p1_pmf[vote_threshold:]))
        if pfa > false_alarm_rate + 1e-9:
            continue
        if best is None or pd > float(best["pd"]):
            best = {
                "pd": pd,
                "pfa": pfa,
                "threshold_votes": vote_threshold,
                "local_false_alarm_rate": alpha,
                "scheduled_reports": model.num_uavs,
                "feasible": True,
            }
    if best is None:
        return {
            "pd": 0.0,
            "pfa": 0.0,
            "threshold_votes": 0,
            "local_false_alarm_rate": None,
            "scheduled_reports": model.num_uavs,
            "feasible": False,
        }
    return best


def degraded_peer_majority_fusion(
    model: TargetEvidenceModel,
    false_alarm_rate: float,
    *,
    observability: float | Sequence[float] | None = None,
    per_hop_reliability: float = 1.0,
    hops: int = 1,
    common_failure_probability: float = 0.0,
    local_false_alarm_options: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Peer majority under partial observability and multi-hop erasure.

    For each UAV ``i``, the probability that its local vote reaches the
    consensus process is

    ``p_i = obs_i * (1 - p_c) * (1 - (1 - r)^hops)``,

    where ``obs_i`` is local observability, ``p_c`` is a network-wide common
    failure probability, and ``r`` is the per-hop link reliability.  The
    effective H0/H1 vote probabilities are
    ``p_i * q0_i`` and ``p_i * q1_i``, and the majority threshold is
    optimized exactly under the global P_FA constraint.
    """
    if not 0.0 <= per_hop_reliability <= 1.0:
        raise ValueError("per_hop_reliability must lie in [0, 1]")
    if hops <= 0:
        raise ValueError("hops must be positive")
    if not 0.0 <= common_failure_probability <= 1.0:
        raise ValueError("common_failure_probability must lie in [0, 1]")
    if observability is None:
        obs = np.ones(model.num_uavs, dtype=float)
    else:
        obs = np.asarray(observability, dtype=float)
        if obs.ndim == 0:
            obs = np.full(model.num_uavs, float(obs))
        if obs.shape != (model.num_uavs,):
            raise ValueError("observability must be scalar or per-UAV")
        if np.any((obs < 0.0) | (obs > 1.0)):
            raise ValueError("observability entries must lie in [0, 1]")
    reachability = 1.0 - (1.0 - per_hop_reliability) ** hops
    participation = obs * (1.0 - common_failure_probability) * reachability
    if local_false_alarm_options is None:
        options = tuple(
            float(value) for value in np.geomspace(0.005, 0.5, 20)
        ) + (0.1,)
    else:
        options = tuple(float(value) for value in local_false_alarm_options)
    best = None
    for alpha in options:
        threshold = (
            model.mu0
            + np.sqrt(np.maximum(np.diag(model.sigma0), 1e-30))
            * norm.ppf(1.0 - alpha)
        )
        q0 = norm.sf(
            (threshold - model.mu0)
            / np.sqrt(np.maximum(np.diag(model.sigma0), 1e-30))
        )
        q1 = norm.sf(
            (threshold - model.mu1)
            / np.sqrt(np.maximum(np.diag(model.sigma1), 1e-30))
        )
        p0 = participation * q0
        p1 = participation * q1
        p0_pmf = _count_distribution(p0)
        p1_pmf = _count_distribution(p1)
        vote_threshold = int(model.num_uavs + 1)
        for candidate in range(1, model.num_uavs + 2):
            if float(np.sum(p0_pmf[candidate:])) <= false_alarm_rate + 1e-12:
                vote_threshold = candidate
                break
        pfa = float(np.sum(p0_pmf[vote_threshold:]))
        pd = float(np.sum(p1_pmf[vote_threshold:]))
        if pfa > false_alarm_rate + 1e-9:
            continue
        if best is None or pd > float(best["pd"]):
            best = {
                "pd": pd,
                "pfa": pfa,
                "threshold_votes": vote_threshold,
                "local_false_alarm_rate": alpha,
                "scheduled_reports": model.num_uavs,
                "feasible": True,
                "mean_participation": float(np.mean(participation)),
            }
    if best is None:
        return {
            "pd": 0.0,
            "pfa": 0.0,
            "threshold_votes": 0,
            "local_false_alarm_rate": None,
            "scheduled_reports": model.num_uavs,
            "feasible": False,
            "mean_participation": float(np.mean(participation)),
        }
    return best


def hard_decision_schedule(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    reports_per_target: int | None = None,
) -> tuple[tuple[frozenset[int], ...], int]:
    """Static 1-bit Top-K schedule with equal per-target report count."""
    count = len(models)
    per_target = (
        max(1, budget_bits // count)
        if reports_per_target is None
        else reports_per_target
    )
    scheduled = [{model.owner} for model in models]
    used = 0
    for q, model in enumerate(models):
        candidates = sorted(
            (
                float(independent_post_report_score(model, i)),
                i,
            )
            for i in range(model.num_uavs)
            if i != model.owner
        )
        candidates.reverse()
        for _, uav in candidates[:per_target]:
            scheduled[q].add(uav)
            used += 1
    return tuple(frozenset(group) for group in scheduled), used


def static_deflection_schedule(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    score: Callable[[TargetEvidenceModel, int], float] = sensing_quality_score,
) -> tuple[tuple[frozenset[int], ...], int]:
    """Global static Top-K schedule with deflection-optimal fusion."""
    qos = np.zeros(len(models), dtype=float)
    weights = np.ones(len(models), dtype=float)
    result = ranked_baseline(
        models, budget_bits, qos, weights, score,
        mode="exact", rng=np.random.default_rng(0),
    )
    return result.scheduled, result.used_bits


def uniform_soft_schedule(
    models: Sequence[TargetEvidenceModel],
    reports_per_target: int,
    score: Callable[[TargetEvidenceModel, int], float] = sensing_quality_score,
) -> tuple[tuple[frozenset[int], ...], int]:
    """Equal per-target soft report allocation, ranked by a static score."""
    scheduled = [{model.owner} for model in models]
    used = 0
    for q, model in enumerate(models):
        candidates = sorted(
            (
                float(score(model, i)),
                i,
            )
            for i in range(model.num_uavs)
            if i != model.owner
        )
        candidates.reverse()
        for _, uav in candidates[:reports_per_target]:
            scheduled[q].add(uav)
            used += int(model.report_bits[uav])
    return tuple(frozenset(group) for group in scheduled), used


def evaluate_schedule_expected_pd(
    models: Sequence[TargetEvidenceModel],
    scheduled: Sequence[Iterable[int]],
    false_alarm_rate: float,
    *,
    pd_mode: str = "optimal",
    grid: int = 512,
) -> np.ndarray:
    """Expected-P_D vector of a schedule under the moment-matched model."""
    return np.asarray([
        expected_gaussian_detection_probability(
            model, scheduled[q], false_alarm_rate,
            pd_mode=pd_mode, grid=grid,
        )
        for q, model in enumerate(models)
    ])
