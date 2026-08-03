"""Exact controlled models for ACK-conditioned two-stage repair."""

from __future__ import annotations

from itertools import product
from dataclasses import dataclass

import numpy as np

from .fusion import gaussian_detection_probability


_PROBABILITY_TOLERANCE = 1e-12


def _best_success_candidate(candidates):
    """Maximize probability, then minimize cost, ignoring roundoff noise."""
    best = candidates[0]
    for candidate in candidates[1:]:
        success = -candidate[0]
        best_success = -best[0]
        if (success > best_success + _PROBABILITY_TOLERANCE
                or (abs(success - best_success) <= _PROBABILITY_TOLERANCE
                    and candidate[1:] < best[1:])):
            best = candidate
    return best


@dataclass(frozen=True)
class ControlledRepairResult:
    success_probability: float
    expected_transmissions: float
    policy: tuple[str, ...]


@dataclass(frozen=True)
class ThresholdOracleResult:
    success_probability: float
    expected_bits: float
    first_stage_actions: tuple[tuple[int, int], ...]
    policy_by_ack: tuple[tuple[tuple[int, int], ...], ...]


def _action_subsets(actions, costs, maximum_bits, domain_capacities, *,
                    unique_reports=False):
    actions = tuple(actions)
    subsets = []
    for included in product((0, 1), repeat=len(actions)):
        selected = tuple(
            action for action, use in zip(actions, included) if use
        )
        if unique_reports and len({action[0] for action in selected}) < len(selected):
            continue
        used = sum(int(costs[report]) for report, _ in selected)
        domain_used = tuple(
            sum(int(costs[report]) for report, selected_domain in selected
                if selected_domain == domain)
            for domain in range(len(domain_capacities))
        )
        if used <= maximum_bits and all(
            value <= domain_capacities[domain]
            for domain, value in enumerate(domain_used)
        ):
            subsets.append((selected, used, domain_used))
    return subsets


def _mask_meets_threshold(mask: int, success_by_mask: np.ndarray) -> bool:
    return bool(success_by_mask[mask])


def _conditional_plan_success(
    posterior, success_probabilities, plan, initial_mask, success_by_mask
) -> float:
    if not plan:
        return float(_mask_meets_threshold(initial_mask, success_by_mask))
    total = 0.0
    for state, state_probability in enumerate(posterior):
        for outcome in product((0, 1), repeat=len(plan)):
            probability = float(state_probability)
            mask = initial_mask
            for delivered, (report, domain) in zip(outcome, plan):
                p = success_probabilities[state, report, domain]
                probability *= p if delivered else 1.0 - p
                if delivered:
                    mask |= 1 << report
            total += probability * _mask_meets_threshold(mask, success_by_mask)
    return float(np.clip(total, 0.0, 1.0))


def optimize_single_target_two_stage_oracle(
    state_probabilities, success_probabilities, report_bits,
    resource_access, success_by_mask, total_budget_bits: int,
    first_stage_budget_bits: int, domain_capacities,
) -> dict[str, ThresholdOracleResult]:
    """Exact small-scale threshold oracle under persistent hidden states.

    Stage one sends at most one primary copy per report. Stage two may send one
    copy per accessible report-domain pair, including a same-domain retry. Each
    ACK branch obeys the total and per-domain hard capacities. The adaptive
    action is indexed only by ACK history; the clairvoyant action may also use
    the hidden state and is returned only as an information upper bound.
    """
    prior = np.asarray(state_probabilities, dtype=float)
    success = np.asarray(success_probabilities, dtype=float)
    costs = np.asarray(report_bits, dtype=int)
    access = np.asarray(resource_access, dtype=bool)
    threshold = np.asarray(success_by_mask, dtype=bool)
    capacities = tuple(int(x) for x in domain_capacities)
    if success.ndim != 3 or success.shape[0] != prior.size:
        raise ValueError("success_probabilities must have shape [state, report, domain]")
    num_reports, num_domains = success.shape[1:]
    if costs.shape != (num_reports,) or access.shape != (num_reports, num_domains):
        raise ValueError("report costs and resource access dimensions must agree")
    if threshold.shape != (1 << num_reports,):
        raise ValueError("success_by_mask must contain one value per report subset")
    if not 0 <= first_stage_budget_bits <= total_budget_bits:
        raise ValueError("first-stage budget must lie within total budget")
    if len(capacities) != num_domains:
        raise ValueError("one capacity is required per domain")
    actions = tuple(
        (report, domain) for report in range(num_reports)
        for domain in range(num_domains) if access[report, domain]
    )
    first_plans = _action_subsets(
        actions, costs, first_stage_budget_bits, capacities,
        unique_reports=True,
    )
    second_budget = total_budget_bits - first_stage_budget_bits

    best_adaptive = None
    best_clairvoyant = None
    for first_plan, first_cost, first_domains in first_plans:
        conditional_first = np.asarray([
            success[:, report, domain] for report, domain in first_plan
        ], dtype=float).T if first_plan else np.empty((prior.size, 0))
        histories, history_probability, posterior = ack_joint_distribution(
            prior, conditional_first
        )
        remaining_domains = tuple(
            capacities[d] - first_domains[d] for d in range(num_domains)
        )
        second_plans = _action_subsets(
            actions, costs, second_budget, remaining_domains
        )
        policy = []
        adaptive_success = 0.0
        expected_bits = float(first_cost)
        clairvoyant_success = 0.0
        clairvoyant_expected_bits = float(first_cost)
        for history_index, history in enumerate(histories):
            initial_mask = 0
            for delivered, (report, _) in zip(history, first_plan):
                if delivered:
                    initial_mask |= 1 << report
            branch_candidates = []
            for plan, plan_cost, _ in second_plans:
                conditional_success = _conditional_plan_success(
                    posterior[history_index], success, plan, initial_mask,
                    threshold,
                )
                branch_candidates.append((-conditional_success, plan_cost, plan))
            negative_success, plan_cost, plan = _best_success_candidate(
                branch_candidates
            )
            policy.append(plan)
            adaptive_success -= history_probability[history_index] * negative_success
            expected_bits += history_probability[history_index] * plan_cost

            for state in range(prior.size):
                scenario_probability = (
                    history_probability[history_index]
                    * posterior[history_index, state]
                )
                state_candidates = []
                state_posterior = np.zeros(prior.size)
                state_posterior[state] = 1.0
                for candidate, candidate_cost, _ in second_plans:
                    value = _conditional_plan_success(
                        state_posterior, success, candidate, initial_mask,
                        threshold,
                    )
                    state_candidates.append((-value, candidate_cost, candidate))
                state_negative, state_cost, _ = _best_success_candidate(
                    state_candidates
                )
                clairvoyant_success -= scenario_probability * state_negative
                clairvoyant_expected_bits += scenario_probability * state_cost
        adaptive_success = float(np.clip(adaptive_success, 0.0, 1.0))
        adaptive_key = (-adaptive_success, expected_bits, first_plan)
        if (best_adaptive is None or _best_success_candidate(
                [best_adaptive[0], adaptive_key]) == adaptive_key):
            best_adaptive = (
                adaptive_key,
                ThresholdOracleResult(
                    adaptive_success, expected_bits, first_plan, tuple(policy)
                ),
            )
        clairvoyant_success = float(np.clip(clairvoyant_success, 0.0, 1.0))
        clairvoyant_key = (-clairvoyant_success, clairvoyant_expected_bits, first_plan)
        if (best_clairvoyant is None or _best_success_candidate(
                [best_clairvoyant[0], clairvoyant_key]) == clairvoyant_key):
            best_clairvoyant = (
                clairvoyant_key,
                ThresholdOracleResult(
                    clairvoyant_success, clairvoyant_expected_bits,
                    first_plan, tuple(),
                ),
            )

    def best_static(unique_reports):
        plans = _action_subsets(
            actions, costs, total_budget_bits, capacities,
            unique_reports=unique_reports,
        )
        candidates = []
        for plan, cost, _ in plans:
            value = _conditional_plan_success(
                prior, success, plan, 0, threshold
            )
            candidates.append((-value, cost, plan))
        negative, cost, plan = _best_success_candidate(candidates)
        return ThresholdOracleResult(
            -negative, float(cost), plan, tuple()
        )

    selection = best_static(unique_reports=True)
    static = best_static(unique_reports=False)
    return {
        "selection": selection,
        "static": static,
        "adaptive": best_adaptive[1],
        "clairvoyant": best_clairvoyant[1],
    }


def two_stage_hidden_state_model(
    model, path_groups, strength: float, path_risk_allocation_factor: float,
    *, num_resources: int = 2,
):
    """Build persistent path/resource states and per-domain report success."""
    paths = np.asarray(path_groups, dtype=int)
    if paths.shape != (model.num_uavs,):
        raise ValueError("path_groups must match model.num_uavs")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    if not 0.0 <= path_risk_allocation_factor <= 1.0:
        raise ValueError("path risk allocation factor must lie in [0, 1]")
    reporters = [i for i in range(model.num_uavs) if i != model.owner]
    path_ids = sorted(set(int(paths[i]) for i in reporters))
    path_column = {path: column for column, path in enumerate(path_ids)}
    state_bits = np.asarray(list(product(
        (0, 1), repeat=len(path_ids) + num_resources
    )), dtype=int)
    prior = np.full(state_bits.shape[0], 1.0 / state_bits.shape[0])
    success = np.empty((state_bits.shape[0], len(reporters), num_resources))
    for state_index, bits in enumerate(state_bits):
        for local_report, original_report in enumerate(reporters):
            marginal = float(model.success_prob[original_report])
            path_mean = 1.0 - path_risk_allocation_factor * (1.0 - marginal)
            resource_mean = marginal / path_mean
            path_displacement = strength * min(path_mean, 1.0 - path_mean)
            resource_displacement = strength * min(
                resource_mean, 1.0 - resource_mean
            )
            path_good = bits[path_column[int(paths[original_report])]]
            path_probability = path_mean + (
                path_displacement if path_good else -path_displacement
            )
            for domain in range(num_resources):
                resource_good = bits[len(path_ids) + domain]
                resource_probability = resource_mean + (
                    resource_displacement if resource_good
                    else -resource_displacement
                )
                success[state_index, local_report, domain] = (
                    path_probability * resource_probability
                )
    return reporters, state_bits, prior, success


def gaussian_pd_threshold_by_mask(
    model, reporters, minimum_pd: float, false_alarm_rate: float
) -> np.ndarray:
    """Map every received-report mask to its Gaussian-PD threshold event."""
    result = np.empty(1 << len(reporters), dtype=bool)
    for mask in range(result.size):
        received = {model.owner, *[
            reporter for bit, reporter in enumerate(reporters)
            if mask & (1 << bit)
        ]}
        result[mask] = gaussian_detection_probability(
            model.mu0, model.mu1, model.sigma0, model.sigma1,
            received, false_alarm_rate,
        ) >= minimum_pd
    return result


def _controlled_success_given_action(history, action, repair_success) -> float:
    if history == (1, 1):
        return 1.0
    if history == (0, 1) and action == "repair_a":
        return float(repair_success[0])
    if history == (1, 0) and action == "repair_b":
        return float(repair_success[1])
    return 0.0


def controlled_two_report_oracles(
    state_probabilities, stage_one_success, repair_success
) -> dict[str, ControlledRepairResult]:
    """Exact static, ACK-adaptive, and hidden-state-aware controlled oracles.

    Reports A and B are both required. Stage one sends each report once. At
    most one second-stage repair may be sent, and the hidden state persists
    across both stages while residual trials are conditionally independent.
    """
    prior = np.asarray(state_probabilities, dtype=float)
    stage_one = np.asarray(stage_one_success, dtype=float)
    repair = np.asarray(repair_success, dtype=float)
    histories, history_probability, posterior = ack_joint_distribution(
        prior, stage_one
    )
    if repair.shape != stage_one.shape:
        raise ValueError("repair_success must match stage_one_success")
    if np.any((repair < 0.0) | (repair > 1.0)):
        raise ValueError("repair success probabilities must lie in [0, 1]")
    actions = ("none", "repair_a", "repair_b")
    action_cost = {"none": 0.0, "repair_a": 1.0, "repair_b": 1.0}

    likelihood = np.empty((len(histories), prior.size), dtype=float)
    for history_index, history in enumerate(histories):
        ack = np.asarray(history, dtype=int)
        likelihood[history_index] = np.prod(
            np.where(ack[None, :] == 1, stage_one, 1.0 - stage_one), axis=1
        )
    joint = likelihood * prior[None, :]

    def evaluate_policy(policy):
        success = 0.0
        expected_second_stage = 0.0
        for history_index, history in enumerate(histories):
            action = policy[history_index]
            expected_second_stage += (
                history_probability[history_index] * action_cost[action]
            )
            for state in range(prior.size):
                success += joint[history_index, state] * (
                    _controlled_success_given_action(
                        history, action, repair[state]
                    )
                )
        return success, 2.0 + expected_second_stage

    adaptive_candidates = []
    for policy in product(actions, repeat=len(histories)):
        success, expected = evaluate_policy(policy)
        adaptive_candidates.append((-success, expected, policy))
    _, adaptive_expected, adaptive_policy = min(adaptive_candidates)
    adaptive_success = -min(adaptive_candidates)[0]

    static_candidates = []
    for fixed_action in ("repair_a", "repair_b"):
        policy = tuple(fixed_action for _ in histories)
        success, _ = evaluate_policy(policy)
        static_candidates.append((-success, 3.0, policy))
    _, static_expected, static_policy = min(static_candidates)
    static_success = -min(static_candidates)[0]

    clairvoyant_success = 0.0
    clairvoyant_expected_second = 0.0
    for history_index, history in enumerate(histories):
        for state in range(prior.size):
            scenario_probability = joint[history_index, state]
            candidates = [
                (-_controlled_success_given_action(history, action, repair[state]),
                 action_cost[action], action)
                for action in actions
            ]
            negative_success, cost, _ = min(candidates)
            clairvoyant_success -= scenario_probability * negative_success
            clairvoyant_expected_second += scenario_probability * cost
    return {
        "static": ControlledRepairResult(
            static_success, static_expected, static_policy
        ),
        "adaptive": ControlledRepairResult(
            adaptive_success, adaptive_expected, adaptive_policy
        ),
        "clairvoyant": ControlledRepairResult(
            clairvoyant_success, 2.0 + clairvoyant_expected_second, tuple()
        ),
    }


def ack_joint_distribution(
    state_probabilities, conditional_success_probabilities
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray]:
    """Return ACK histories, their probabilities, and P(state | ACK).

    Conditional on a hidden common state, report-level residual transmission
    trials are independent. The returned posterior is indexed [history, state].
    """
    prior = np.asarray(state_probabilities, dtype=float)
    success = np.asarray(conditional_success_probabilities, dtype=float)
    if prior.ndim != 1 or success.ndim != 2 or success.shape[0] != prior.size:
        raise ValueError("state and conditional-success dimensions must agree")
    if np.any(prior < 0.0) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("state probabilities must be nonnegative and sum to one")
    if np.any((success < 0.0) | (success > 1.0)):
        raise ValueError("conditional success probabilities must lie in [0, 1]")
    histories = tuple(product((0, 1), repeat=success.shape[1]))
    likelihood = np.empty((len(histories), prior.size), dtype=float)
    for history_index, history in enumerate(histories):
        ack = np.asarray(history, dtype=int)
        likelihood[history_index] = np.prod(
            np.where(ack[None, :] == 1, success, 1.0 - success), axis=1
        )
    joint = likelihood * prior[None, :]
    history_probabilities = joint.sum(axis=1)
    positive = history_probabilities > 0.0
    posterior = np.zeros_like(joint)
    posterior[positive] = (
        joint[positive] / history_probabilities[positive, None]
    )
    return histories, history_probabilities, posterior


def complementary_static_success(success_probability: float) -> float:
    """Success when two required reports are sent and one is pre-duplicated."""
    p = float(success_probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("success_probability must lie in [0, 1]")
    return 2.0 * p**2 - p**3


def complementary_adaptive_success(success_probability: float) -> float:
    """Success when ACK identifies which of two required reports to repair."""
    p = float(success_probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("success_probability must lie in [0, 1]")
    return 3.0 * p**2 - 2.0 * p**3
