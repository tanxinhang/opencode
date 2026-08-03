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
    domain_cost_bits: tuple[int, ...] = tuple()


@dataclass(frozen=True)
class ReplicationChanceResult:
    copy_counts: tuple[tuple[int, ...], ...]
    violation_probability_per_target: np.ndarray
    violation_excess_per_target: np.ndarray
    weighted_violation_excess: float
    risk_objective: float
    used_bits: int
    feasible: bool


_OBJECTIVE_TOLERANCE = 1e-12


def _lexicographically_less(left, right) -> bool:
    for a, b in zip(left, right):
        if a < b - _OBJECTIVE_TOLERANCE:
            return True
        if a > b + _OBJECTIVE_TOLERANCE:
            return False
    return False


def _objectives_equal(left, right) -> bool:
    return all(
        abs(a - b) <= _OBJECTIVE_TOLERANCE for a, b in zip(left, right)
    )


def _fair_label_dominates(left, right) -> bool:
    """Return whether one partial fair-objective label safely dominates another."""
    return all(
        a <= b + _OBJECTIVE_TOLERANCE for a, b in zip(left, right)
    ) and any(
        a < b - _OBJECTIVE_TOLERANCE for a, b in zip(left, right)
    )


def _solve_dual_domain_groups(
    groups: Sequence[Sequence[ReplicationOption]], budget_bits: int,
    domain_capacities: Sequence[int], weights: np.ndarray, limits: np.ndarray,
    objective_mode: str,
) -> tuple[int, tuple[float, ...], tuple[ReplicationOption, ...]]:
    """Multiple-choice DP with Pareto labels for the non-additive fair maximum."""
    if objective_mode not in ("weighted", "fair"):
        raise ValueError("objective_mode must be weighted or fair")
    if objective_mode == "fair" and np.any(limits <= 0.0):
        raise ValueError("fair objective requires strictly positive violation limits")
    capacity = tuple(int(x) for x in domain_capacities)
    zero_key = (0.0, 0.0, 0.0) if objective_mode == "fair" else (0.0, 0.0)
    states = {(0, tuple(0 for _ in capacity)): [(zero_key, tuple())]}
    for q, options in enumerate(groups):
        next_states: dict[
            tuple[int, tuple[int, ...]],
            list[tuple[tuple[float, ...], tuple[ReplicationOption, ...]]],
        ] = {}
        for (prior_cost, prior_domains), labels in states.items():
            for prior_key, chosen in labels:
                for option in options:
                    cost = prior_cost + option.cost_bits
                    domains = tuple(
                        a + b for a, b in zip(
                            prior_domains, option.domain_cost_bits
                        )
                    )
                    if cost > budget_bits or any(
                        value > capacity[j] for j, value in enumerate(domains)
                    ):
                        continue
                    raw_excess = max(
                        option.violation_probability - limits[q], 0.0
                    )
                    weighted_excess = weights[q] * raw_excess
                    if objective_mode == "fair":
                        key = (
                            max(prior_key[0], raw_excess / limits[q]),
                            prior_key[1] + weighted_excess,
                            prior_key[2] + option.risk_objective,
                        )
                    else:
                        key = (
                            prior_key[0] + weighted_excess,
                            prior_key[1] + option.risk_objective,
                        )
                    state_key = (cost, domains)
                    candidate = (key, chosen + (option,))
                    labels_at_state = next_states.setdefault(state_key, [])
                    if objective_mode == "weighted":
                        if (not labels_at_state or _lexicographically_less(
                                key, labels_at_state[0][0])):
                            labels_at_state[:] = [candidate]
                        continue
                    if any(
                        _objectives_equal(existing[0], key)
                        for existing in labels_at_state
                    ):
                        continue
                    if any(
                        _fair_label_dominates(existing[0], key)
                        for existing in labels_at_state
                    ):
                        continue
                    labels_at_state[:] = [
                        existing for existing in labels_at_state
                        if not _fair_label_dominates(key, existing[0])
                    ]
                    labels_at_state.append(candidate)
        states = next_states
    candidates = [
        (key, used, chosen)
        for (used, _), labels in states.items()
        for key, chosen in labels
    ]
    key, used, chosen = candidates[0]
    for candidate_key, candidate_used, candidate_chosen in candidates[1:]:
        if (_lexicographically_less(candidate_key, key)
                or (_objectives_equal(candidate_key, key)
                    and candidate_used < used)):
            key, used, chosen = (
                candidate_key, candidate_used, candidate_chosen
            )
    return used, key, chosen


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


def dual_layer_reception_model(
    model: TargetEvidenceModel,
    copy_counts: Sequence[int],
    path_groups: Sequence[int],
    native_resources: Sequence[int],
    strength: float,
    path_failure_fraction: float,
    *,
    replication_mode: str = "cross_domain",
    num_resources: int = 2,
) -> TargetEvidenceModel:
    """Effective reception with shared physical-path and schedulable resource risks.

    For report i, the original marginal success p_i is decomposed as
    a_i*c_i=p_i, where a_i=1-lambda*(1-p_i) is physical-path availability.
    Every copy shares the same path event. Resource-domain common states and
    copy-level residual trials are independent of the path layer. Thus a
    second resource can mitigate resource failure, but never path failure.
    """
    counts = np.asarray(copy_counts, dtype=int)
    path_groups = np.asarray(path_groups, dtype=int)
    native = np.asarray(native_resources, dtype=int)
    n = model.num_uavs
    if counts.shape != (n,) or path_groups.shape != (n,) or native.shape != (n,):
        raise ValueError("copy counts, path groups and resources must match num_uavs")
    if not 0.0 <= path_failure_fraction <= 1.0:
        raise ValueError("path_failure_fraction must lie in [0, 1]")
    if replication_mode not in ("cross_domain", "same_domain"):
        raise ValueError("replication_mode must be cross_domain or same_domain")
    candidates = [i for i in range(n) if i != model.owner]
    path_ids = sorted(set(int(path_groups[i]) for i in candidates))
    if any(native[i] < 0 or native[i] >= num_resources for i in candidates):
        raise ValueError("native resource index is outside available resources")
    state_bits = np.asarray(list(product(
        (0, 1), repeat=len(path_ids) + num_resources
    )), dtype=int)
    weights = np.full(state_bits.shape[0], 1.0 / state_bits.shape[0])
    conditional = np.zeros((state_bits.shape[0], n), dtype=float)
    conditional[:, model.owner] = 1.0
    path_column = {group: j for j, group in enumerate(path_ids)}
    for state_index, bits in enumerate(state_bits):
        for i in candidates:
            if counts[i] == 0:
                continue
            p = float(model.success_prob[i])
            path_mean = 1.0 - path_failure_fraction * (1.0 - p)
            resource_mean = p / path_mean
            path_displacement = strength * min(path_mean, 1.0 - path_mean)
            resource_displacement = strength * min(
                resource_mean, 1.0 - resource_mean
            )
            path_good = bits[path_column[int(path_groups[i])]]
            path_probability = path_mean + (
                path_displacement if path_good else -path_displacement
            )
            resources = [int(native[i])]
            if counts[i] == 2:
                resources.append(
                    int(native[i]) if replication_mode == "same_domain"
                    else (int(native[i]) + 1) % num_resources
                )
            copy_success = []
            for resource in resources:
                resource_good = bits[len(path_ids) + resource]
                copy_success.append(resource_mean + (
                    resource_displacement if resource_good
                    else -resource_displacement
                ))
            conditional[state_index, i] = path_probability * (
                1.0 - float(np.prod([1.0 - value for value in copy_success]))
            )
    repaired = replace(
        model,
        success_prob=weights @ conditional,
        reception_patterns=None,
        pattern_probabilities=None,
        reception_state_probabilities=weights,
        conditional_success_probabilities=conditional,
    )
    repaired.validate()
    return repaired


def enumerate_dual_layer_options(
    model: TargetEvidenceModel,
    minimum_pd: float,
    target_weight: float,
    path_groups: Sequence[int],
    native_resources: Sequence[int],
    strength: float,
    path_failure_fraction: float,
    maximum_cost_bits: int,
    domain_capacities: Sequence[int],
    *,
    replication_mode: str,
    maximum_copies: int,
    resource_access: np.ndarray | None = None,
    false_alarm_rate: float = 0.05,
    beta: float = 0.9,
    tail_weight: float = 1.0,
) -> list[ReplicationOption]:
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    native = np.asarray(native_resources, dtype=int)
    capacities = np.asarray(domain_capacities, dtype=int)
    access = (
        np.ones((model.num_uavs, capacities.size), dtype=bool)
        if resource_access is None else np.asarray(resource_access, dtype=bool)
    )
    if access.shape != (model.num_uavs, capacities.size):
        raise ValueError("resource_access must have shape [num_uavs, num_domains]")
    quality_cache: dict[frozenset[int], float] = {}
    options = []
    for candidate_counts in product(range(maximum_copies + 1), repeat=len(candidates)):
        counts = np.zeros(model.num_uavs, dtype=int); counts[candidates] = candidate_counts
        domain_cost = np.zeros(capacities.size, dtype=int)
        for i in candidates:
            if counts[i] >= 1:
                if not access[i, native[i]]:
                    domain_cost[:] = capacities + 1
                    break
                domain_cost[native[i]] += int(model.report_bits[i])
            if counts[i] == 2:
                second = native[i] if replication_mode == "same_domain" else (native[i] + 1) % capacities.size
                if not access[i, second]:
                    domain_cost[:] = capacities + 1
                    break
                domain_cost[second] += int(model.report_bits[i])
        cost = int(domain_cost.sum())
        if cost > maximum_cost_bits or np.any(domain_cost > capacities):
            continue
        repaired = dual_layer_reception_model(
            model, counts, path_groups, native_resources, strength,
            path_failure_fraction, replication_mode=replication_mode,
            num_resources=capacities.size,
        )
        values = []; probabilities = []
        scheduled = {model.owner, *[i for i in candidates if counts[i] > 0]}
        for received, probability in _received_patterns(repaired, scheduled):
            key = frozenset(received)
            if key not in quality_cache:
                quality_cache[key] = _received_quality(model, key, "gaussian_pd", false_alarm_rate)
            values.append(max(minimum_pd - quality_cache[key], 0.0)); probabilities.append(probability)
        distribution = LossDistribution(np.asarray(values), np.asarray(probabilities))
        options.append(ReplicationOption(
            target=model.target_id, copy_counts=tuple(int(x) for x in counts),
            cost_bits=cost, violation_probability=distribution.violation_probability(),
            mean_loss=distribution.mean, cvar_loss=distribution.cvar(beta),
            risk_objective=float(target_weight) * (distribution.mean + tail_weight * distribution.cvar(beta)),
            domain_cost_bits=tuple(int(x) for x in domain_cost),
        ))
    return options


def optimize_dual_layer_chance_portfolio(
    models: Sequence[TargetEvidenceModel], budget_bits: int,
    minimum_pd: Sequence[float], target_weights: Sequence[float],
    violation_limits: Sequence[float], path_groups: Sequence[Sequence[int]],
    native_resources: Sequence[Sequence[int]], strength: float,
    path_failure_fraction: float, domain_capacities: Sequence[int], *,
    replication_mode: str = "cross_domain", maximum_copies: int = 2,
    resource_access: Sequence[np.ndarray] | None = None,
    objective_mode: str = "weighted",
    false_alarm_rate: float = 0.05, beta: float = 0.9,
    tail_weight: float = 1.0,
) -> ReplicationChanceResult:
    weights = np.asarray(target_weights, dtype=float); limits = np.asarray(violation_limits, dtype=float)
    if resource_access is None:
        access_per_target = [None] * len(models)
    else:
        access_per_target = list(resource_access)
        if len(access_per_target) != len(models):
            raise ValueError("resource_access must provide one mask per target")
    groups = [enumerate_dual_layer_options(
        model, minimum_pd[q], weights[q], path_groups[q], native_resources[q],
        strength, path_failure_fraction, budget_bits, domain_capacities,
        replication_mode=replication_mode, maximum_copies=maximum_copies,
        resource_access=access_per_target[q],
        false_alarm_rate=false_alarm_rate, beta=beta, tail_weight=tail_weight,
    ) for q, model in enumerate(models)]
    used, key, chosen = _solve_dual_domain_groups(
        groups, budget_bits, domain_capacities, weights, limits, objective_mode
    )
    violation = np.asarray([option.violation_probability for option in chosen]); excess = np.maximum(violation - limits, 0.0)
    risk_index = 2 if objective_mode == "fair" else 1
    return ReplicationChanceResult(
        copy_counts=tuple(option.copy_counts for option in chosen),
        violation_probability_per_target=violation,
        violation_excess_per_target=excess,
        weighted_violation_excess=float(weights @ excess),
        risk_objective=float(key[risk_index]),
        used_bits=used, feasible=bool(np.all(excess <= 1e-12)),
    )


def _experimental_enumerate_threshold_bundle_options(
    model: TargetEvidenceModel,
    minimum_pd: float,
    target_weight: float,
    path_groups: Sequence[int],
    native_resources: Sequence[int],
    strength: float,
    path_failure_fraction: float,
    maximum_cost_bits: int,
    domain_capacities: Sequence[int],
    *,
    baseline_copy_counts: Sequence[int],
    maximum_bundle_actions: int = 3,
    resource_access: np.ndarray | None = None,
    false_alarm_rate: float = 0.05,
    beta: float = 0.9,
    tail_weight: float = 1.0,
) -> list[ReplicationOption]:
    """Enumerate bounded action bundles and retain their local Pareto frontier.

    The neighborhood starts at a selection-only solution. Each local action
    adds or removes one report copy, so swaps, paired additions, and replication
    repairs are scored only after the complete bundle is formed. For fixed
    depth K the candidate count is O(N**K), rather than the O(3**N)
    configurations used by the exact oracle.
    """
    if maximum_bundle_actions < 0:
        raise ValueError("maximum_bundle_actions must be nonnegative")
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    baseline = tuple(int(x) for x in baseline_copy_counts)
    if len(baseline) != model.num_uavs or baseline[model.owner] != 0:
        raise ValueError("baseline_copy_counts must match the model")
    native = np.asarray(native_resources, dtype=int)
    capacities = np.asarray(domain_capacities, dtype=int)
    access = (
        np.ones((model.num_uavs, capacities.size), dtype=bool)
        if resource_access is None else np.asarray(resource_access, dtype=bool)
    )
    if access.shape != (model.num_uavs, capacities.size):
        raise ValueError("resource_access must have shape [num_uavs, num_domains]")

    quality_cache: dict[frozenset[int], float] = {}
    neighborhoods = {baseline}
    frontier_states = {baseline}
    for _ in range(maximum_bundle_actions):
        next_states = set()
        for state in frontier_states:
            for i in candidates:
                for delta in (-1, 1):
                    value = state[i] + delta
                    if value < 0 or value > 2:
                        continue
                    updated = list(state)
                    updated[i] = value
                    next_states.add(tuple(updated))
        next_states -= neighborhoods
        neighborhoods |= next_states
        frontier_states = next_states

    options = []
    for state in neighborhoods:
        counts = np.asarray(state, dtype=int)
        domain_cost = np.zeros(capacities.size, dtype=int)
        valid = True
        for i in candidates:
            if counts[i] >= 1:
                domain = int(native[i])
                valid &= bool(access[i, domain])
                domain_cost[domain] += int(model.report_bits[i])
            if counts[i] == 2:
                domain = (int(native[i]) + 1) % capacities.size
                valid &= bool(access[i, domain])
                domain_cost[domain] += int(model.report_bits[i])
        cost = int(domain_cost.sum())
        if (not valid or cost > maximum_cost_bits
                or np.any(domain_cost > capacities)):
            continue
        repaired = dual_layer_reception_model(
            model, counts, path_groups, native_resources, strength,
            path_failure_fraction, replication_mode="cross_domain",
            num_resources=capacities.size,
        )
        scheduled = {model.owner, *[i for i in candidates if counts[i] > 0]}
        values = []
        probabilities = []
        for received, probability in _received_patterns(repaired, scheduled):
            key = frozenset(received)
            if key not in quality_cache:
                quality_cache[key] = _received_quality(
                    model, key, "gaussian_pd", false_alarm_rate
                )
            values.append(max(minimum_pd - quality_cache[key], 0.0))
            probabilities.append(probability)
        distribution = LossDistribution(
            np.asarray(values), np.asarray(probabilities)
        )
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
            domain_cost_bits=tuple(int(x) for x in domain_cost),
        ))

    frontier = []
    for option in options:
        dominated = any(
            all(a <= b for a, b in zip(other.domain_cost_bits, option.domain_cost_bits))
            and other.violation_probability <= option.violation_probability + 1e-12
            and other.risk_objective <= option.risk_objective + 1e-12
            and (
                other.domain_cost_bits != option.domain_cost_bits
                or other.violation_probability < option.violation_probability - 1e-12
                or other.risk_objective < option.risk_objective - 1e-12
            )
            for other in options if other is not option
        )
        if not dominated:
            frontier.append(option)
    return frontier


def _experimental_optimize_threshold_bundle_portfolio(
    models: Sequence[TargetEvidenceModel], budget_bits: int,
    minimum_pd: Sequence[float], target_weights: Sequence[float],
    violation_limits: Sequence[float], path_groups: Sequence[Sequence[int]],
    native_resources: Sequence[Sequence[int]], strength: float,
    path_failure_fraction: float, domain_capacities: Sequence[int], *,
    maximum_bundle_actions: int = 3,
    resource_access: Sequence[np.ndarray] | None = None,
    objective_mode: str = "fair",
    false_alarm_rate: float = 0.05, beta: float = 0.9,
    tail_weight: float = 1.0,
) -> ReplicationChanceResult:
    """Solve the two-domain multiple-choice DP over threshold-aware bundles."""
    weights = np.asarray(target_weights, dtype=float)
    limits = np.asarray(violation_limits, dtype=float)
    access_per_target = (
        [None] * len(models) if resource_access is None else list(resource_access)
    )
    if len(access_per_target) != len(models):
        raise ValueError("resource_access must provide one mask per target")
    if objective_mode not in ("weighted", "fair"):
        raise ValueError("objective_mode must be weighted or fair")
    baseline = optimize_dual_layer_chance_portfolio(
        models, budget_bits, minimum_pd, target_weights, violation_limits,
        path_groups, native_resources, strength, path_failure_fraction,
        domain_capacities, replication_mode="cross_domain", maximum_copies=1,
        resource_access=resource_access, objective_mode=objective_mode,
        false_alarm_rate=false_alarm_rate,
        beta=beta, tail_weight=tail_weight,
    )
    groups = [
        _experimental_enumerate_threshold_bundle_options(
            model, minimum_pd[q], weights[q], path_groups[q],
            native_resources[q], strength, path_failure_fraction, budget_bits,
            domain_capacities, baseline_copy_counts=baseline.copy_counts[q],
            maximum_bundle_actions=maximum_bundle_actions,
            resource_access=access_per_target[q],
            false_alarm_rate=false_alarm_rate, beta=beta,
            tail_weight=tail_weight,
        )
        for q, model in enumerate(models)
    ]
    used, key, chosen = _solve_dual_domain_groups(
        groups, budget_bits, domain_capacities, weights, limits, objective_mode
    )
    violation = np.asarray([
        option.violation_probability for option in chosen
    ])
    excess = np.maximum(violation - limits, 0.0)
    risk_index = 2 if objective_mode == "fair" else 1
    return ReplicationChanceResult(
        copy_counts=tuple(option.copy_counts for option in chosen),
        violation_probability_per_target=violation,
        violation_excess_per_target=excess,
        weighted_violation_excess=float(weights @ excess),
        risk_objective=float(key[risk_index]), used_bits=used,
        feasible=bool(np.all(excess <= 1e-12)),
    )


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
