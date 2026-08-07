import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.discrete_descent import discrete_gradient_select
from uav_otfs_isac.expected_pd import expected_pd_greedy_select


def _models():
    return [
        symmetric_diversity_model(
            np.array([1.6, 1.3, 1.1, 0.9]), success_probability=0.7
        )
        for _ in range(3)
    ]


def test_discrete_descent_respects_budget_and_never_worsens():
    models = _models()
    init = expected_pd_greedy_select(
        models, budget_bits=8, false_alarm_rate=0.05, grid=256
    )
    result = discrete_gradient_select(
        models, budget_bits=8, false_alarm_rate=0.05,
        init_schedule=init.scheduled, grid=256,
    )
    assert result.used_bits <= 8
    assert np.all(result.expected_pd >= init.expected_pd - 1e-12)
    assert result.normalized_qos_gap <= init.normalized_qos_gap + 1e-12


def test_discrete_descent_keeps_owner_and_valid_indices():
    models = _models()
    init = expected_pd_greedy_select(
        models, budget_bits=6, false_alarm_rate=0.05, grid=256
    )
    result = discrete_gradient_select(
        models, budget_bits=6, false_alarm_rate=0.05,
        init_schedule=init.scheduled, grid=256,
    )
    for q, model in enumerate(models):
        assert model.owner in result.scheduled[q]
        assert all(0 <= uav < model.num_uavs for uav in result.scheduled[q])
