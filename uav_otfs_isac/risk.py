from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from collections.abc import Iterable, Sequence

import numpy as np

from .fusion import gaussian_detection_probability, optimal_deflection
from .models import SelectionResult, TargetEvidenceModel


@dataclass(frozen=True)
class LossDistribution:
    values: np.ndarray
    probabilities: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.probabilities @ self.values)

    def cvar(self, beta: float) -> float:
        """Upper-tail CVaR using the Rockafellar--Uryasev definition."""
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        candidates = np.unique(self.values)
        objectives = [
            eta + float(self.probabilities @ np.maximum(self.values - eta, 0.0)) / (1.0 - beta)
            for eta in candidates
        ]
        return float(min(objectives))

    def violation_probability(self, tolerance: float = 1e-12) -> float:
        return float(self.probabilities[self.values > tolerance].sum())


@dataclass(frozen=True)
class PortfolioOption:
    target: int
    scheduled: frozenset[int]
    cost_bits: int
    expected_deflection: float
    expected_quality: float
    mean_loss: float
    cvar_loss: float
    violation_probability: float
    risk_objective: float


@dataclass(frozen=True)
class PortfolioResult:
    selection: SelectionResult
    mean_loss_per_target: np.ndarray
    cvar_loss_per_target: np.ndarray
    violation_probability_per_target: np.ndarray
    objective: float
    beta: float
    tail_weight: float


@dataclass(frozen=True)
class ChancePortfolioResult:
    portfolio: PortfolioResult
    violation_limits: np.ndarray
    violation_excess_per_target: np.ndarray
    weighted_violation_excess: float
    feasible: bool


@dataclass(frozen=True)
class ReachabilityDiagnosis:
    target: int
    classification: str
    maximum_deterministic_quality: float
    minimum_unlimited_violation_probability: float
    minimum_budgeted_violation_probability: float
    violation_limit: float


@dataclass(frozen=True)
class FairChancePortfolioResult:
    chance: ChancePortfolioResult
    maximum_relative_violation_excess: float


@dataclass(frozen=True)
class FailureDiversityAttribution:
    """Target-level structural and recoverable-headroom diagnostics."""

    minimum_successful_reports: int | None
    supporting_failure_domains: int
    classification: str
    independent_violation: float
    aware_violation: float
    oracle_violation: float
    all_scheduled_violation: float
    oracle_all_scheduled_gap: float
    near_all_scheduled_boundary: bool
    recoverable_headroom: float
    headroom_use_ratio: float | None
    oracle_scheduled: frozenset[int]


def evaluate_portfolio_schedule(
    models: Sequence[TargetEvidenceModel],
    scheduled: Sequence[Iterable[int]],
    minimum_deflection: Sequence[float],
    target_weights: Sequence[float],
    *,
    beta: float = 0.9,
    tail_weight: float = 1.0,
) -> dict[str, np.ndarray | float]:
    distributions = [
        deflection_loss_distribution(model, scheduled[q], minimum_deflection[q])
        for q, model in enumerate(models)
    ]
    mean_loss = np.asarray([distribution.mean for distribution in distributions])
    cvar_loss = np.asarray([distribution.cvar(beta) for distribution in distributions])
    violation = np.asarray([
        distribution.violation_probability() for distribution in distributions
    ])
    weights = np.asarray(target_weights, dtype=float)
    return {
        "mean_loss_per_target": mean_loss,
        "cvar_loss_per_target": cvar_loss,
        "violation_probability_per_target": violation,
        "weighted_mean_loss": float(weights @ mean_loss),
        "weighted_cvar_loss": float(weights @ cvar_loss),
        "risk_objective": float(weights @ (mean_loss + tail_weight * cvar_loss)),
    }


def deflection_loss_distribution(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    minimum_deflection: float,
) -> LossDistribution:
    """Exact distribution induced by independent detectable report erasures."""
    reports = sorted(set(scheduled) - {model.owner})
    values = []
    probabilities = []
    for pattern in product((0, 1), repeat=len(reports)):
        probability = 1.0
        received = {model.owner}
        for uav, success in zip(reports, pattern):
            p = float(model.success_prob[uav])
            probability *= p if success else 1.0 - p
            if success:
                received.add(uav)
        deflection = optimal_deflection(model.delta, model.sigma0, received)
        values.append(max(float(minimum_deflection) - deflection, 0.0))
        probabilities.append(probability)
    distribution = LossDistribution(np.asarray(values), np.asarray(probabilities))
    if not np.isclose(distribution.probabilities.sum(), 1.0, atol=1e-10):
        raise RuntimeError("reception-pattern probabilities do not sum to one")
    return distribution


def gaussian_pd_loss_distribution(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    minimum_pd: float,
    false_alarm_rate: float,
) -> LossDistribution:
    """Gaussian-score P_D deficit distribution under random report erasures."""
    return _quality_loss_distribution(
        model, scheduled, minimum_pd, "gaussian_pd", false_alarm_rate
    )


def _received_patterns(model: TargetEvidenceModel, scheduled: Iterable[int]):
    if model.reception_state_probabilities is not None:
        reports = sorted(set(scheduled) - {model.owner})
        aggregated: dict[frozenset[int], float] = {}
        for state_weight, conditional in zip(
            model.reception_state_probabilities,
            model.conditional_success_probabilities,
        ):
            for pattern in product((0, 1), repeat=len(reports)):
                probability = float(state_weight)
                received = {model.owner}
                for uav, success in zip(reports, pattern):
                    p = float(conditional[uav])
                    probability *= p if success else 1.0 - p
                    if success:
                        received.add(uav)
                key = frozenset(received)
                aggregated[key] = aggregated.get(key, 0.0) + probability
        yield from aggregated.items()
        return
    if model.reception_patterns is not None:
        scheduled_set = set(scheduled)
        for pattern, probability in zip(
            model.reception_patterns, model.pattern_probabilities
        ):
            received = {model.owner}
            received.update(
                i for i in scheduled_set if i != model.owner and pattern[i] == 1
            )
            yield received, float(probability)
        return
    reports = sorted(set(scheduled) - {model.owner})
    for pattern in product((0, 1), repeat=len(reports)):
        probability = 1.0
        received = {model.owner}
        for uav, success in zip(reports, pattern):
            p = float(model.success_prob[uav])
            probability *= p if success else 1.0 - p
            if success:
                received.add(uav)
        yield received, probability


def _received_quality(
    model: TargetEvidenceModel,
    received: Iterable[int],
    quality_mode: str,
    false_alarm_rate: float,
) -> float:
    if quality_mode == "deflection":
        return optimal_deflection(model.delta, model.sigma0, received)
    if quality_mode == "gaussian_pd":
        return gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1,
            received, false_alarm_rate,
        )
    raise ValueError("quality_mode must be 'deflection' or 'gaussian_pd'")


def _quality_loss_distribution(
    model: TargetEvidenceModel,
    scheduled: Iterable[int],
    minimum_quality: float,
    quality_mode: str,
    false_alarm_rate: float,
) -> LossDistribution:
    values = []
    probabilities = []
    for received, probability in _received_patterns(model, scheduled):
        quality = _received_quality(model, received, quality_mode, false_alarm_rate)
        values.append(max(float(minimum_quality) - quality, 0.0))
        probabilities.append(probability)
    distribution = LossDistribution(np.asarray(values), np.asarray(probabilities))
    if not np.isclose(distribution.probabilities.sum(), 1.0, atol=1e-10):
        raise RuntimeError("reception-pattern probabilities do not sum to one")
    return distribution


def enumerate_target_portfolios(
    model: TargetEvidenceModel,
    minimum_deflection: float,
    target_weight: float,
    beta: float,
    tail_weight: float,
    quality_mode: str = "deflection",
    false_alarm_rate: float = 0.05,
) -> list[PortfolioOption]:
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    options = []
    quality_cache: dict[frozenset[int], float] = {}
    deflection_cache: dict[frozenset[int], float] = {}

    def cached_quality(received: Iterable[int]) -> float:
        key = frozenset(received)
        if key not in quality_cache:
            quality_cache[key] = _received_quality(
                model, key, quality_mode, false_alarm_rate
            )
        return quality_cache[key]

    def cached_deflection(received: Iterable[int]) -> float:
        key = frozenset(received)
        if key not in deflection_cache:
            deflection_cache[key] = optimal_deflection(
                model.delta, model.sigma0, key
            )
        return deflection_cache[key]

    for mask in range(1 << len(candidates)):
        scheduled = {model.owner}
        for index, uav in enumerate(candidates):
            if mask & (1 << index):
                scheduled.add(uav)
        values = []; probabilities = []
        expected_deflection = 0.0; expected_quality = 0.0
        for received, probability in _received_patterns(model, scheduled):
            quality = cached_quality(received)
            values.append(max(float(minimum_deflection) - quality, 0.0))
            probabilities.append(probability)
            expected_quality += probability * quality
            expected_deflection += probability * cached_deflection(received)
        distribution = LossDistribution(np.asarray(values), np.asarray(probabilities))
        mean_loss = distribution.mean
        cvar_loss = distribution.cvar(beta)
        cost = sum(int(model.report_bits[i]) for i in scheduled if i != model.owner)
        options.append(PortfolioOption(
            target=model.target_id,
            scheduled=frozenset(scheduled),
            cost_bits=cost,
            expected_deflection=float(expected_deflection),
            expected_quality=float(expected_quality),
            mean_loss=mean_loss,
            cvar_loss=cvar_loss,
            violation_probability=distribution.violation_probability(),
            risk_objective=float(target_weight) * (mean_loss + tail_weight * cvar_loss),
        ))
    return options


def diagnose_target_reachability(
    model: TargetEvidenceModel,
    budget_bits: int,
    minimum_quality: float,
    violation_limit: float,
    *,
    quality_mode: str = "deflection",
    false_alarm_rate: float = 0.05,
) -> ReachabilityDiagnosis:
    """Exact diagnosis that does not assume set monotonicity."""
    options = enumerate_target_portfolios(
        model, minimum_quality, 1.0, beta=0.9, tail_weight=0.0,
        quality_mode=quality_mode, false_alarm_rate=false_alarm_rate,
    )
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    deterministic_qualities = []
    for mask in range(1 << len(candidates)):
        received = {model.owner}
        received.update(
            candidates[j] for j in range(len(candidates)) if mask & (1 << j)
        )
        deterministic_qualities.append(
            _received_quality(model, received, quality_mode, false_alarm_rate)
        )
    maximum_quality = max(deterministic_qualities)
    unlimited_violation = min(option.violation_probability for option in options)
    budgeted = [option for option in options if option.cost_bits <= budget_bits]
    budgeted_violation = min(option.violation_probability for option in budgeted)
    tolerance = 1e-12
    if maximum_quality < minimum_quality - tolerance:
        classification = "sensing_limited"
    elif unlimited_violation > violation_limit + tolerance:
        classification = "reliability_limited"
    elif budgeted_violation > violation_limit + tolerance:
        classification = "budget_limited"
    else:
        classification = "feasible"
    return ReachabilityDiagnosis(
        target=model.target_id,
        classification=classification,
        maximum_deterministic_quality=float(maximum_quality),
        minimum_unlimited_violation_probability=float(unlimited_violation),
        minimum_budgeted_violation_probability=float(budgeted_violation),
        violation_limit=float(violation_limit),
    )


def attribute_failure_diversity_headroom(
    independent_model: TargetEvidenceModel,
    truth_model: TargetEvidenceModel,
    independent_scheduled: Iterable[int],
    aware_scheduled: Iterable[int],
    budget_bits: int,
    minimum_pd: float,
    failure_groups: Sequence[int],
    *,
    false_alarm_rate: float = 0.05,
    tolerance: float = 1e-12,
) -> FailureDiversityAttribution:
    """Explain target-level headroom without assuming set monotonicity.

    The deterministic structural metrics use the evidence moments only.  The
    violation and Oracle metrics use ``truth_model`` and therefore include the
    supplied correlated-erasure law.  ``budget_bits`` is the target-local cost
    ceiling used for the exact Oracle.
    """
    if independent_model.num_uavs != truth_model.num_uavs:
        raise ValueError("independent and truth models must have matching UAVs")
    groups = np.asarray(failure_groups, dtype=int)
    if groups.shape != (truth_model.num_uavs,):
        raise ValueError("failure_groups must have one entry per UAV")
    candidates = [i for i in range(truth_model.num_uavs) if i != truth_model.owner]

    minimum_reports = None
    for count in range(len(candidates) + 1):
        qualities = []
        for subset in combinations(candidates, count):
            received = {truth_model.owner, *subset}
            qualities.append(_received_quality(
                truth_model, received, "gaussian_pd", false_alarm_rate
            ))
        if any(value >= minimum_pd - tolerance for value in qualities):
            minimum_reports = count
            break

    supporting_domains = {
        int(groups[i]) for i in candidates
        if _received_quality(
            truth_model, {truth_model.owner, i}, "gaussian_pd", false_alarm_rate
        ) >= minimum_pd - tolerance
    }
    if minimum_reports is None:
        classification = "sensing_limited"
    elif minimum_reports == 0:
        classification = "owner_sufficient"
    elif minimum_reports == 1 and len(supporting_domains) >= 2:
        classification = "diversifiable_substitute"
    elif minimum_reports == 1:
        classification = "single_domain_substitute"
    else:
        classification = "complementary_evidence"

    options = enumerate_target_portfolios(
        truth_model, minimum_pd, 1.0, beta=0.9, tail_weight=0.0,
        quality_mode="gaussian_pd", false_alarm_rate=false_alarm_rate,
    )
    affordable = [option for option in options if option.cost_bits <= budget_bits]
    if not affordable:
        raise RuntimeError("target-local budget admits no portfolio")
    oracle = min(
        affordable,
        key=lambda option: (
            option.violation_probability,
            option.mean_loss,
            option.cost_bits,
            sorted(option.scheduled),
        ),
    )
    independent_violation = gaussian_pd_loss_distribution(
        truth_model, independent_scheduled, minimum_pd, false_alarm_rate
    ).violation_probability()
    aware_violation = gaussian_pd_loss_distribution(
        truth_model, aware_scheduled, minimum_pd, false_alarm_rate
    ).violation_probability()
    all_scheduled = frozenset(range(truth_model.num_uavs))
    all_violation = gaussian_pd_loss_distribution(
        truth_model, all_scheduled, minimum_pd, false_alarm_rate
    ).violation_probability()
    headroom = max(independent_violation - oracle.violation_probability, 0.0)
    use_ratio = (
        None if headroom <= tolerance
        else (independent_violation - aware_violation) / headroom
    )
    return FailureDiversityAttribution(
        minimum_successful_reports=minimum_reports,
        supporting_failure_domains=len(supporting_domains),
        classification=classification,
        independent_violation=float(independent_violation),
        aware_violation=float(aware_violation),
        oracle_violation=float(oracle.violation_probability),
        all_scheduled_violation=float(all_violation),
        oracle_all_scheduled_gap=float(
            oracle.violation_probability - all_violation
        ),
        near_all_scheduled_boundary=bool(
            abs(oracle.violation_probability - all_violation) <= 1e-6
        ),
        recoverable_headroom=float(headroom),
        headroom_use_ratio=None if use_ratio is None else float(use_ratio),
        oracle_scheduled=oracle.scheduled,
    )


def optimize_risk_portfolio(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    minimum_deflection: Sequence[float],
    target_weights: Sequence[float],
    *,
    beta: float = 0.9,
    tail_weight: float = 1.0,
) -> PortfolioResult:
    """Exact multiple-choice knapsack DP over target-level evidence sets."""
    if budget_bits < 0:
        raise ValueError("budget_bits must be nonnegative")
    if tail_weight < 0:
        raise ValueError("tail_weight must be nonnegative")
    if len(models) != len(minimum_deflection) or len(models) != len(target_weights):
        raise ValueError("target parameter lengths must match models")
    option_groups = [
        enumerate_target_portfolios(model, minimum_deflection[q], target_weights[q], beta, tail_weight)
        for q, model in enumerate(models)
    ]
    # cost -> (objective, selected options); keeping one nondominated state per
    # exact cost is sufficient because all future option groups are shared.
    states: dict[int, tuple[float, tuple[PortfolioOption, ...]]] = {0: (0.0, tuple())}
    for options in option_groups:
        next_states: dict[int, tuple[float, tuple[PortfolioOption, ...]]] = {}
        for prior_cost, (prior_objective, chosen) in states.items():
            for option in options:
                cost = prior_cost + option.cost_bits
                if cost > budget_bits:
                    continue
                candidate = (prior_objective + option.risk_objective, chosen + (option,))
                incumbent = next_states.get(cost)
                if incumbent is None or candidate[0] < incumbent[0] - 1e-12:
                    next_states[cost] = candidate
        states = next_states
    if not states:
        raise RuntimeError("no feasible portfolio")
    used, (objective, chosen) = min(
        states.items(), key=lambda item: (item[1][0], item[0])
    )
    quality = np.asarray([option.expected_deflection for option in chosen])
    mean_loss = np.asarray([option.mean_loss for option in chosen])
    cvar_loss = np.asarray([option.cvar_loss for option in chosen])
    violation = np.asarray([option.violation_probability for option in chosen])
    normalized_gap = float(np.sum(
        np.asarray(target_weights) * mean_loss / np.maximum(np.asarray(minimum_deflection), 1e-12)
    ))
    selection = SelectionResult(
        scheduled=tuple(option.scheduled for option in chosen),
        expected_deflection=quality,
        used_bits=used,
        normalized_qos_gap=normalized_gap,
        trace=tuple(),
        quality_mode="deflection",
        expected_quality=quality.copy(),
    )
    return PortfolioResult(
        selection=selection,
        mean_loss_per_target=mean_loss,
        cvar_loss_per_target=cvar_loss,
        violation_probability_per_target=violation,
        objective=float(objective),
        beta=beta,
        tail_weight=tail_weight,
    )


def optimize_chance_constrained_portfolio(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    minimum_deflection: Sequence[float],
    target_weights: Sequence[float],
    violation_limits: Sequence[float],
    *,
    beta: float = 0.9,
    tail_weight: float = 1.0,
    quality_mode: str = "deflection",
    false_alarm_rate: float = 0.05,
) -> ChancePortfolioResult:
    """Lexicographic exact DP with minimum chance-constraint relaxation.

    The primary objective minimizes weighted excess violation probability. If
    zero excess is feasible, the secondary mean-CVaR objective selects among
    strictly feasible portfolios. Otherwise, the result exposes the minimum
    reliability relaxation required by the available reporting budget.
    """
    limits = np.asarray(violation_limits, dtype=float)
    weights = np.asarray(target_weights, dtype=float)
    if len(models) != len(limits) or len(models) != len(weights):
        raise ValueError("target parameter lengths must match models")
    if np.any((limits < 0.0) | (limits > 1.0)):
        raise ValueError("violation limits must lie in [0, 1]")
    option_groups = [
        enumerate_target_portfolios(
            model, minimum_deflection[q], weights[q], beta, tail_weight,
            quality_mode, false_alarm_rate,
        )
        for q, model in enumerate(models)
    ]
    # cost -> ((weighted chance slack, risk), selected target portfolios)
    states: dict[
        int, tuple[tuple[float, float], tuple[PortfolioOption, ...]]
    ] = {0: ((0.0, 0.0), tuple())}
    for q, options in enumerate(option_groups):
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
    if not states:
        raise RuntimeError("no feasible budget allocation")
    used, (key, chosen) = min(
        states.items(), key=lambda item: (item[1][0][0], item[1][0][1], item[0])
    )
    deflection = np.asarray([option.expected_deflection for option in chosen])
    quality = np.asarray([option.expected_quality for option in chosen])
    mean_loss = np.asarray([option.mean_loss for option in chosen])
    cvar_loss = np.asarray([option.cvar_loss for option in chosen])
    violation = np.asarray([option.violation_probability for option in chosen])
    excess = np.maximum(violation - limits, 0.0)
    normalized_gap = float(np.sum(
        weights * mean_loss / np.maximum(np.asarray(minimum_deflection), 1e-12)
    ))
    selection = SelectionResult(
        scheduled=tuple(option.scheduled for option in chosen),
        expected_deflection=deflection,
        used_bits=used,
        normalized_qos_gap=normalized_gap,
        trace=tuple(),
        quality_mode=quality_mode,
        expected_quality=quality,
    )
    portfolio = PortfolioResult(
        selection=selection,
        mean_loss_per_target=mean_loss,
        cvar_loss_per_target=cvar_loss,
        violation_probability_per_target=violation,
        objective=float(key[1]),
        beta=beta,
        tail_weight=tail_weight,
    )
    return ChancePortfolioResult(
        portfolio=portfolio,
        violation_limits=limits,
        violation_excess_per_target=excess,
        weighted_violation_excess=float(weights @ excess),
        feasible=bool(np.all(excess <= 1e-12)),
    )


def optimize_fair_chance_constrained_portfolio(
    models: Sequence[TargetEvidenceModel],
    budget_bits: int,
    minimum_quality: Sequence[float],
    target_weights: Sequence[float],
    violation_limits: Sequence[float],
    *,
    beta: float = 0.9,
    tail_weight: float = 1.0,
    quality_mode: str = "deflection",
    false_alarm_rate: float = 0.05,
) -> FairChancePortfolioResult:
    """Exact three-level objective: worst relative excess, sum excess, risk."""
    limits = np.asarray(violation_limits, dtype=float)
    weights = np.asarray(target_weights, dtype=float)
    if np.any(limits <= 0.0):
        raise ValueError("fair relative relaxation requires positive violation limits")
    option_groups = [
        enumerate_target_portfolios(
            model, minimum_quality[q], weights[q], beta, tail_weight,
            quality_mode, false_alarm_rate,
        )
        for q, model in enumerate(models)
    ]
    relative_excess = [
        [max(option.violation_probability - limits[q], 0.0) / limits[q]
         for option in options]
        for q, options in enumerate(option_groups)
    ]
    thresholds = sorted({0.0, *(value for group in relative_excess for value in group)})
    chosen_threshold = None
    best = None
    for threshold in thresholds:
        # cost -> ((weighted sum excess, risk), selected options)
        states = {0: ((0.0, 0.0), tuple())}
        for q, options in enumerate(option_groups):
            next_states = {}
            for prior_cost, (prior_key, chosen) in states.items():
                for option, relative in zip(options, relative_excess[q]):
                    if relative > threshold + 1e-12:
                        continue
                    cost = prior_cost + option.cost_bits
                    if cost > budget_bits:
                        continue
                    excess = max(option.violation_probability - limits[q], 0.0)
                    key = (
                        prior_key[0] + weights[q] * excess,
                        prior_key[1] + option.risk_objective,
                    )
                    incumbent = next_states.get(cost)
                    if incumbent is None or key < incumbent[0]:
                        next_states[cost] = (key, chosen + (option,))
            states = next_states
            if not states:
                break
        if states:
            used, (key, selected) = min(
                states.items(), key=lambda item: (item[1][0][0], item[1][0][1], item[0])
            )
            chosen_threshold = threshold
            best = (used, key, selected)
            break
    if best is None or chosen_threshold is None:
        raise RuntimeError("no feasible budget allocation")
    used, key, selected = best
    deflection = np.asarray([option.expected_deflection for option in selected])
    quality = np.asarray([option.expected_quality for option in selected])
    mean_loss = np.asarray([option.mean_loss for option in selected])
    cvar_loss = np.asarray([option.cvar_loss for option in selected])
    violation = np.asarray([option.violation_probability for option in selected])
    excess = np.maximum(violation - limits, 0.0)
    selection = SelectionResult(
        scheduled=tuple(option.scheduled for option in selected),
        expected_deflection=deflection,
        used_bits=used,
        normalized_qos_gap=float(np.sum(
            weights * mean_loss / np.maximum(np.asarray(minimum_quality), 1e-12)
        )),
        trace=tuple(),
        quality_mode=quality_mode,
        expected_quality=quality,
    )
    portfolio = PortfolioResult(
        selection=selection,
        mean_loss_per_target=mean_loss,
        cvar_loss_per_target=cvar_loss,
        violation_probability_per_target=violation,
        objective=float(key[1]),
        beta=beta,
        tail_weight=tail_weight,
    )
    chance = ChancePortfolioResult(
        portfolio=portfolio,
        violation_limits=limits,
        violation_excess_per_target=excess,
        weighted_violation_excess=float(weights @ excess),
        feasible=bool(np.all(excess <= 1e-12)),
    )
    return FairChancePortfolioResult(
        chance=chance,
        maximum_relative_violation_excess=float(np.max(excess / limits)),
    )
