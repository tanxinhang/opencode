from itertools import product

import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.sota_baselines import (
    degraded_peer_majority_fusion,
    evaluate_schedule_expected_pd,
    exact_counting_feasible,
    exact_min_majority_uavs,
    hard_decision_fusion,
    hard_decision_local_probabilities,
    hard_decision_schedule,
    majority_feasibility_trace,
    optimized_hard_decision_fusion,
    peer_majority_fusion,
    uniform_soft_schedule,
)


def test_hard_decision_fusion_matches_bruteforce():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=1.0
    )
    scheduled = {0, 1, 2}
    result = hard_decision_fusion(model, scheduled, 0.05, local_false_alarm_rate=0.1)
    threshold = int(result["threshold_votes"])
    per_uav = {
        uav: hard_decision_local_probabilities(model, uav, 0.1)
        for uav in scheduled
    }
    pd = 0.0
    pfa = 0.0
    for bits in product((0, 1), repeat=len(scheduled)):
        count = sum(bits)
        if count < threshold:
            continue
        prob0 = 1.0
        prob1 = 1.0
        for bit, uav in zip(bits, sorted(scheduled)):
            p0, p1 = per_uav[uav]
            prob0 *= p0 if bit else 1.0 - p0
            prob1 *= p1 if bit else 1.0 - p1
        pd += prob1
        pfa += prob0
    assert np.isclose(result["pd"], pd, atol=1e-12)
    assert np.isclose(result["pfa"], pfa, atol=1e-12)
    assert result["pfa"] <= 0.05 + 1e-12


def test_hard_decision_schedule_respects_budget():
    models = [
        symmetric_diversity_model(
            np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
        )
        for _ in range(3)
    ]
    scheduled, used = hard_decision_schedule(models, budget_bits=12)
    assert used <= 12
    assert all(len(group) == 1 + 4 for group in scheduled)


def test_uniform_soft_schedule_uses_reported_costs():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=1.0
    )
    scheduled, used = uniform_soft_schedule([model], reports_per_target=2)
    assert used == 2
    assert len(scheduled[0]) == 3


def test_evaluate_schedule_expected_pd_shape():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=1.0
    )
    scheduled, _ = uniform_soft_schedule([model], reports_per_target=1)
    values = evaluate_schedule_expected_pd(
        [model], scheduled, 0.05, pd_mode="optimal", grid=256
    )
    assert values.shape == (1,)
    assert 0.0 <= values[0] <= 1.0


def test_optimized_hard_decision_never_worse_than_default():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
    )
    scheduled = {0, 1, 2, 3}
    default = hard_decision_fusion(model, scheduled, 0.05)
    optimized = optimized_hard_decision_fusion(model, scheduled, 0.05)
    assert optimized["pd"] >= default["pd"] - 1e-12
    assert optimized["pfa"] <= 0.05 + 1e-9


def test_peer_majority_fusion_respects_pfa():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
    )
    result = peer_majority_fusion(model, 0.05)
    assert result["feasible"]
    assert result["pfa"] <= 0.05 + 1e-9
    assert result["pd"] > 0.0
    assert result["threshold_votes"] <= model.num_uavs


def test_degraded_peer_majority_is_feasible_and_not_better():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
    )
    clean = peer_majority_fusion(model, 0.05)
    degraded = degraded_peer_majority_fusion(
        model, 0.05, observability=0.7, per_hop_reliability=0.8, hops=2
    )
    assert degraded["feasible"]
    assert degraded["pfa"] <= 0.05 + 1e-9
    assert degraded["pd"] <= clean["pd"] + 1e-9
    assert 0.0 < degraded["mean_participation"] <= 0.7


def test_common_failure_and_heterogeneous_observability():
    model = symmetric_diversity_model(
        np.array([1.2, 1.0, 0.8, 0.6]), success_probability=0.8
    )
    clean = peer_majority_fusion(model, 0.05)
    degraded = degraded_peer_majority_fusion(
        model, 0.05,
        observability=[1.0, 1.0, 0.5, 0.5, 0.5],
        per_hop_reliability=0.9,
        hops=2,
        common_failure_probability=0.2,
    )
    assert degraded["feasible"]
    assert degraded["pfa"] <= 0.05 + 1e-9
    assert degraded["pd"] <= clean["pd"] + 1e-9
    assert degraded["mean_participation"] < 0.72


def test_exact_counting_feasible_matches_poisson_binomial_prefixes():
    p0 = [0.1] * 5
    p1 = [0.7] * 5
    trace = majority_feasibility_trace(p0, p1, 0.05, 0.7)
    assert trace == [False, False, True, False, True]
    assert exact_min_majority_uavs(p0, p1, 0.05, 0.7) == 3
    assert exact_counting_feasible(p0, p1, 0.05, 0.7)


def test_majority_feasibility_is_not_monotone_in_voter_count():
    p0 = [0.1, 0.1, 0.1, 0.1]
    p1 = [0.7, 0.7, 0.7, 0.7]
    trace = majority_feasibility_trace(p0, p1, 0.05, 0.7)
    assert any(
        trace[index] and not trace[index + 1]
        for index in range(len(trace) - 1)
    )
