from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from collections.abc import Sequence

import numpy as np

from .models import TargetEvidenceModel
from .risk import LossDistribution, _received_patterns, _received_quality


@dataclass(frozen=True)
class ReplicationOption:
    target: int
    copy_counts: tuple[int, ...]
    cost_bits: int
    violation_probability: float
    mean_loss: float
    cvar_loss: float
    risk_objective: float


@dataclass(frozen=True)
class ReplicationChanceResult:
    copy_counts: tuple[tuple[int, ...], ...]
    violation_probability_per_target: np.ndarray
    violation_excess_per_target: np.ndarray
    weighted_violation_excess: float
    risk_objective: float
    used_bits: int
    feasible: bool


def replicated_reception_model(
    model: TargetEvidenceModel,
    copy_counts: Sequence[int],
    native_domains: Sequence[int],
    strength: float,
    *,
    num_domains: int = 2,
) -> TargetEvidenceModel:
    """Build the effective report-erasure law after cross-domain replication.

    A report's first copy uses its native failure domain.  Its second copy uses
    the other domain (two-domain case), so replication creates failure
    diversity rather than repeating on the same common-risk resource.
    Per-copy marginal success remains the original reporting-link probability.
    """
    counts = np.asarray(copy_counts, dtype=int)
    domains = np.asarray(native_domains, dtype=int)
    if counts.shape != (model.num_uavs,) or domains.shape != (model.num_uavs,):
        raise ValueError("copy_counts and native_domains must match model.num_uavs")
    if num_domains != 2:
        raise ValueError("the current cross-domain repair model requires two domains")
    if np.any((counts < 0) | (counts > 2)):
        raise ValueError("copy counts must lie in {0, 1, 2}")
    if counts[model.owner] != 0:
        raise ValueError("the owner does not transmit a report")
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    if any(domains[i] not in (0, 1) for i in candidates):
        raise ValueError("each non-owner must have native domain 0 or 1")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")

    states = np.asarray(list(product((0, 1), repeat=num_domains)), dtype=int)
    state_probabilities = np.full(states.shape[0], 1.0 / states.shape[0])
    conditional = np.zeros((states.shape[0], model.num_uavs), dtype=float)
    conditional[:, model.owner] = 1.0
    for state_index, state in enumerate(states):
        for i in candidates:
            if counts[i] == 0:
                continue
            selected_domains = [int(domains[i])]
            if counts[i] == 2:
                selected_domains.append(1 - int(domains[i]))
            p = float(model.success_prob[i])
            displacement = strength * min(p, 1.0 - p)
            copy_success = [
                p + displacement if state[domain] else p - displacement
                for domain in selected_domains
            ]
            conditional[state_index, i] = 1.0 - float(np.prod(
                [1.0 - value for value in copy_success]
            ))
    effective_marginal = state_probabilities @ conditional
    repaired = replace(
        model,
        success_prob=effective_marginal,
        reception_patterns=None,
        pattern_probabilities=None,
        reception_state_probabilities=state_probabilities,
        conditional_success_probabilities=conditional,
    )
    repaired.validate()
    return repaired


def enumerate_replication_options(
    model: TargetEvidenceModel,
    minimum_pd: float,
    target_weight: float,
    native_domains: Sequence[int],
    strength: float,
    maximum_cost_bits: int,
    *,
    false_alarm_rate: float = 0.05,
    beta: float = 0.9,
    tail_weight: float = 1.0,
) -> list[ReplicationOption]:
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    quality_cache: dict[frozenset[int], float] = {}
    options = []
    for candidate_counts in product((0, 1, 2), repeat=len(candidates)):
        counts = np.zeros(model.num_uavs, dtype=int)
        counts[candidates] = candidate_counts
        cost = sum(
            int(counts[i]) * int(model.report_bits[i]) for i in candidates
        )
        if cost > maximum_cost_bits:
            continue
        repaired = replicated_reception_model(
            model, counts, native_domains, strength
        )
        values = []; probabilities = []
        for received, probability in _received_patterns(
            repaired, {model.owner, *[i for i in candidates if counts[i] > 0]}
        ):
            key = frozenset(received)
            if key not in quality_cache:
                quality_cache[key] = _received_quality(
                    model, key, "gaussian_pd", false_alarm_rate
                )
            values.append(max(minimum_pd - quality_cache[key], 0.0))
            probabilities.append(probability)
        distribution = LossDistribution(np.asarray(values), np.asarray(probabilities))
        options.append(ReplicationOption(
            target=model.target_id,
            copy_counts=tuple(int(x) for x in counts),
            cost_bits=cost,
            violation_probability=distribution.violation_probability(),
            mean_loss=distribution.mean,
            cvar_loss=distribution.cvar(beta),
            risk_objective=float(target_weight) * (
                distribution.mean + tail_weight * distribution.cvar(beta)
            ),
        ))
    return options


def optimize_replication_chance_portfolio(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    minimum_pd: Sequence[float],
    target_weights: Sequence[float],
    violation_limits: Sequence[float],
    native_domains: Sequence[Sequence[int]],
    strength: float,
    *,
    false_alarm_rate: float = 0.05,
    beta: float = 0.9,
    tail_weight: float = 1.0,
) -> ReplicationChanceResult:
    weights = np.asarray(target_weights, dtype=float)
    limits = np.asarray(violation_limits, dtype=float)
    if not (len(models) == len(minimum_pd) == len(weights) == len(limits) == len(native_domains)):
        raise ValueError("target parameter lengths must match models")
    groups = [
        enumerate_replication_options(
            model, minimum_pd[q], weights[q], native_domains[q], strength,
            budget_bits, false_alarm_rate=false_alarm_rate,
            beta=beta, tail_weight=tail_weight,
        ) for q, model in enumerate(models)
    ]
    states = {0: ((0.0, 0.0), tuple())}
    for q, options in enumerate(groups):
        next_states = {}
        for prior_cost, (prior_key, chosen) in states.items():
            for option in options:
                cost = prior_cost + option.cost_bits
                if cost > budget_bits:
                    continue
                excess = weights[q] * max(
                    option.violation_probability - limits[q], 0.0
                )
                key = (prior_key[0] + excess, prior_key[1] + option.risk_objective)
                incumbent = next_states.get(cost)
                if incumbent is None or key < incumbent[0]:
                    next_states[cost] = (key, chosen + (option,))
        states = next_states
    used, (key, chosen) = min(
        states.items(), key=lambda item: (item[1][0][0], item[1][0][1], item[0])
    )
    violation = np.asarray([option.violation_probability for option in chosen])
    excess = np.maximum(violation - limits, 0.0)
    return ReplicationChanceResult(
        copy_counts=tuple(option.copy_counts for option in chosen),
        violation_probability_per_target=violation,
        violation_excess_per_target=excess,
        weighted_violation_excess=float(weights @ excess),
        risk_objective=float(key[1]),
        used_bits=used,
        feasible=bool(np.all(excess <= 1e-12)),
    )
