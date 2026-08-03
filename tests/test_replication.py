import numpy as np
from itertools import product

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.fusion import gaussian_detection_probability
from uav_otfs_isac.replication import (
    ReplicationOption,
    _solve_dual_domain_groups,
    dual_layer_reception_model,
    optimize_dual_layer_chance_portfolio,
    optimize_replication_chance_portfolio,
    _experimental_optimize_threshold_bundle_portfolio,
    replicated_reception_model,
)
from uav_otfs_isac.reliability import with_grouped_common_state_erasures
from uav_otfs_isac.risk import optimize_chance_constrained_portfolio


DOMAINS = np.array([-1, 0, 0, 1, 1])


def pd(model, received):
    return gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1, received, 0.05
    )


def test_cross_domain_copy_preserves_per_copy_link_model_and_improves_effective_success():
    model = symmetric_diversity_model(success_probability=0.6)
    counts = np.array([0, 2, 0, 0, 0])
    repaired = replicated_reception_model(model, counts, DOMAINS, 1.0)
    assert np.isclose(repaired.success_prob[1], 1.0 - (1.0 - 0.6) ** 2)
    assert np.allclose(repaired.success_prob[[2, 3, 4]], 0.0)


def test_c5_replication_beats_selection_when_only_same_domain_reports_clear_threshold():
    model = symmetric_diversity_model(
        np.array([1.4, 1.4, 0.6, 0.6]), success_probability=0.6
    )
    reference = symmetric_diversity_model(success_probability=0.6)
    threshold = (pd(reference, {0}) + pd(reference, {0, 1})) / 2
    truth = with_grouped_common_state_erasures([model], 1.0, DOMAINS)[0]
    selection = optimize_chance_constrained_portfolio(
        [truth], 2, [threshold], [1.0], [0.0],
        quality_mode="gaussian_pd", false_alarm_rate=0.05,
    )
    result = optimize_replication_chance_portfolio(
        [model], 2, [threshold], [1.0], [0.0], [DOMAINS], 1.0
    )
    counts = result.copy_counts[0]
    assert sum(counts) == 2
    assert max(counts[1], counts[2]) == 2
    assert np.isclose(
        selection.portfolio.violation_probability_per_target[0], 0.32
    )
    assert np.isclose(result.violation_probability_per_target[0], 0.16)


def test_dual_layer_model_reduces_to_no_replication_gain_when_path_risk_is_total():
    model = symmetric_diversity_model(success_probability=0.6)
    one = dual_layer_reception_model(
        model, [0, 1, 0, 0, 0], DOMAINS, DOMAINS, 1.0, 1.0
    )
    two = dual_layer_reception_model(
        model, [0, 2, 0, 0, 0], DOMAINS, DOMAINS, 1.0, 1.0
    )
    assert np.isclose(one.success_prob[1], 0.6)
    assert np.isclose(two.success_prob[1], 0.6)


def test_dual_layer_cross_domain_copy_beats_same_domain_with_shared_path_risk():
    model = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.6
    )
    common = dict(
        models=[model], budget_bits=2, minimum_pd=[0.17],
        target_weights=[1.0], violation_limits=[0.0],
        path_groups=[DOMAINS], native_resources=[DOMAINS], strength=1.0,
        path_failure_fraction=0.5, domain_capacities=[2, 2],
    )
    cross = optimize_dual_layer_chance_portfolio(
        **common, replication_mode="cross_domain", maximum_copies=2
    )
    same = optimize_dual_layer_chance_portfolio(
        **common, replication_mode="same_domain", maximum_copies=2
    )
    assert cross.violation_probability_per_target[0] < same.violation_probability_per_target[0]


def test_resource_access_mask_blocks_unavailable_second_domain():
    model = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.6
    )
    access = np.ones((model.num_uavs, 2), dtype=bool)
    access[1, 1] = False
    result = optimize_dual_layer_chance_portfolio(
        [model], 2, [0.17], [1.0], [0.0], [DOMAINS], [DOMAINS],
        1.0, 0.5, [2, 2], replication_mode="cross_domain",
        maximum_copies=2, resource_access=[access],
    )
    assert result.copy_counts[0][1] < 2


def test_threshold_bundle_matches_oracle_when_depth_covers_budget():
    model = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.6
    )
    common = dict(
        models=[model], budget_bits=3, minimum_pd=[0.17],
        target_weights=[1.0], violation_limits=[0.0],
        path_groups=[DOMAINS], native_resources=[DOMAINS], strength=1.0,
        path_failure_fraction=0.5, domain_capacities=[2, 2],
    )
    oracle = optimize_dual_layer_chance_portfolio(
        **common, replication_mode="cross_domain", maximum_copies=2,
        objective_mode="weighted",
    )
    bundle = _experimental_optimize_threshold_bundle_portfolio(
        **common, maximum_bundle_actions=3, objective_mode="weighted"
    )
    assert np.allclose(
        bundle.violation_probability_per_target,
        oracle.violation_probability_per_target,
    )
    assert bundle.risk_objective <= oracle.risk_objective + 1e-12


def test_fair_objective_minimizes_worst_normalized_excess_first():
    models = [
        symmetric_diversity_model(success_probability=0.6),
        symmetric_diversity_model(success_probability=0.6),
    ]
    common = dict(
        models=models, budget_bits=2, minimum_pd=[0.17, 0.17],
        target_weights=[1.0, 3.0], violation_limits=[0.1, 0.1],
        path_groups=[DOMAINS, DOMAINS],
        native_resources=[DOMAINS, DOMAINS], strength=0.5,
        path_failure_fraction=0.5, domain_capacities=[1, 1],
        replication_mode="cross_domain", maximum_copies=1,
    )
    weighted = optimize_dual_layer_chance_portfolio(
        **common, objective_mode="weighted"
    )
    fair = optimize_dual_layer_chance_portfolio(
        **common, objective_mode="fair"
    )
    assert np.max(fair.violation_excess_per_target / 0.1) <= np.max(
        weighted.violation_excess_per_target / 0.1
    ) + 1e-12


def test_fair_dp_keeps_label_that_becomes_best_after_future_maximum():
    def option(target, excess, cost):
        return ReplicationOption(
            target=target, copy_counts=(cost,), cost_bits=cost,
            violation_probability=1.0 + excess, mean_loss=0.0,
            cvar_loss=0.0, risk_objective=0.0,
            domain_cost_bits=(cost, 0),
        )

    groups = [
        [option(0, 0.2, 1), option(0, 0.3, 0)],
        [option(1, 0.2, 0), option(1, 0.0, 1)],
        [option(2, 0.4, 0)],
    ]
    used, key, chosen = _solve_dual_domain_groups(
        groups, 1, [1, 0], np.ones(3), np.ones(3), "fair"
    )
    assert used == 1
    assert np.allclose(key, [0.4, 0.7, 0.0])
    assert chosen[0].cost_bits == 0
    assert chosen[1].cost_bits == 1


def test_fair_dp_matches_independent_brute_force():
    def option(target, violation, cost, risk):
        return ReplicationOption(
            target=target, copy_counts=(cost,), cost_bits=cost,
            violation_probability=violation, mean_loss=0.0,
            cvar_loss=0.0, risk_objective=risk,
            domain_cost_bits=(cost, 0),
        )

    groups = [
        [option(0, 0.3, 1, 0.2), option(0, 0.4, 0, 0.1)],
        [option(1, 0.5, 0, 0.3), option(1, 0.2, 1, 0.4)],
        [option(2, 0.6, 0, 0.1), option(2, 0.1, 1, 0.8)],
    ]
    weights = np.array([1.0, 2.0, 1.5])
    limits = np.array([0.1, 0.2, 0.3])
    used, key, _ = _solve_dual_domain_groups(
        groups, 2, [2, 0], weights, limits, "fair"
    )
    brute = []
    for chosen in product(*groups):
        cost = sum(value.cost_bits for value in chosen)
        if cost > 2:
            continue
        excess = np.maximum(
            [value.violation_probability for value in chosen] - limits, 0.0
        )
        brute.append((
            (float(np.max(excess / limits)), float(weights @ excess),
             float(sum(value.risk_objective for value in chosen))), cost
        ))
    expected_key, expected_used = min(brute, key=lambda value: (value[0], value[1]))
    assert used == expected_used
    assert np.allclose(key, expected_key)


def test_fair_objective_rejects_nonpositive_violation_limit():
    model = symmetric_diversity_model(success_probability=0.6)
    with np.testing.assert_raises_regex(ValueError, "strictly positive"):
        optimize_dual_layer_chance_portfolio(
            [model], 1, [0.17], [1.0], [0.0], [DOMAINS], [DOMAINS],
            0.5, 0.5, [1, 1], replication_mode="cross_domain",
            maximum_copies=1, objective_mode="fair",
        )


def test_fair_dp_uses_lower_risk_when_higher_priorities_differ_only_by_roundoff():
    first = ReplicationOption(
        target=0, copy_counts=(0,), cost_bits=0,
        violation_probability=0.3, mean_loss=0.0, cvar_loss=0.0,
        risk_objective=1.0, domain_cost_bits=(0, 0),
    )
    second = ReplicationOption(
        target=0, copy_counts=(1,), cost_bits=0,
        violation_probability=0.3 + 5e-14, mean_loss=0.0, cvar_loss=0.0,
        risk_objective=0.5, domain_cost_bits=(0, 0),
    )
    _, key, chosen = _solve_dual_domain_groups(
        [[first, second]], 0, [0, 0], np.ones(1), np.array([0.1]), "fair"
    )
    assert chosen[0] is second
    assert np.isclose(key[2], 0.5)
