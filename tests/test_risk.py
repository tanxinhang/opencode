import itertools

import numpy as np

from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.risk import (
    LossDistribution,
    deflection_loss_distribution,
    evaluate_portfolio_schedule,
    enumerate_target_portfolios,
    optimize_risk_portfolio,
    optimize_chance_constrained_portfolio,
)


def make_model(target, owner, delta, success):
    n = len(delta)
    return TargetEvidenceModel(
        target_id=target, owner=owner, mu0=np.zeros(n), mu1=np.asarray(delta),
        sigma0=np.eye(n), sigma1=np.eye(n), success_prob=np.asarray(success),
        report_bits=np.asarray([0 if i == owner else 1 for i in range(n)]),
        bit_flip_prob=np.zeros(n), quantizer_edges=np.array([-np.inf, 0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )


def test_discrete_cvar_and_loss_distribution():
    distribution = LossDistribution(np.array([0.0, 2.0]), np.array([0.8, 0.2]))
    assert np.isclose(distribution.mean, 0.4)
    assert np.isclose(distribution.cvar(0.8), 2.0)
    assert np.isclose(distribution.violation_probability(), 0.2)
    model = make_model(0, 0, [1.0, 1.0], [1.0, 0.75])
    loss = deflection_loss_distribution(model, {0, 1}, minimum_deflection=1.5)
    assert np.isclose(loss.mean, 0.125)
    assert np.isclose(loss.violation_probability(), 0.25)


def test_portfolio_dp_matches_direct_product_enumeration():
    models = [
        make_model(0, 0, [0.8, 1.0, 0.4], [1.0, 0.6, 0.9]),
        make_model(1, 1, [0.7, 0.9, 0.8], [0.8, 1.0, 0.5]),
    ]
    result = optimize_risk_portfolio(models, 2, [1.5, 1.5], [1.0, 1.2], beta=0.8, tail_weight=1.5)
    groups = [enumerate_target_portfolios(m, 1.5, [1.0, 1.2][q], 0.8, 1.5)
              for q, m in enumerate(models)]
    feasible = [pair for pair in itertools.product(*groups)
                if sum(option.cost_bits for option in pair) <= 2]
    direct = min(sum(option.risk_objective for option in pair) for pair in feasible)
    assert np.isclose(result.objective, direct)
    assert result.selection.used_bits <= 2
    metrics = evaluate_portfolio_schedule(
        models, result.selection.scheduled, [1.5, 1.5], [1.0, 1.2], beta=0.8
    )
    assert np.allclose(metrics["mean_loss_per_target"], result.mean_loss_per_target)


def test_chance_constrained_dp_matches_lexicographic_enumeration():
    models = [
        make_model(0, 0, [0.8, 1.0, 0.4], [1.0, 0.6, 0.9]),
        make_model(1, 1, [0.7, 0.9, 0.8], [0.8, 1.0, 0.5]),
    ]
    limits = np.array([0.25, 0.30])
    weights = np.array([1.0, 1.2])
    result = optimize_chance_constrained_portfolio(
        models, 2, [1.5, 1.5], weights, limits, beta=0.8, tail_weight=1.5
    )
    groups = [
        enumerate_target_portfolios(model, 1.5, weights[q], 0.8, 1.5)
        for q, model in enumerate(models)
    ]
    feasible_budget = [
        pair for pair in itertools.product(*groups)
        if sum(option.cost_bits for option in pair) <= 2
    ]
    direct_key = min(
        (
            sum(weights[q] * max(option.violation_probability - limits[q], 0.0)
                for q, option in enumerate(pair)),
            sum(option.risk_objective for option in pair),
        )
        for pair in feasible_budget
    )
    assert np.isclose(result.weighted_violation_excess, direct_key[0])
    assert np.isclose(result.portfolio.objective, direct_key[1])


def test_infeasible_chance_constraint_returns_minimum_slack():
    model = make_model(0, 0, [0.2, 0.3], [1.0, 0.5])
    result = optimize_chance_constrained_portfolio(
        [model], 0, [2.0], [1.0], [0.1]
    )
    assert not result.feasible
    assert result.weighted_violation_excess > 0.0
