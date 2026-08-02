import numpy as np

from uav_otfs_isac.config import ExperimentConfig
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select


def test_selection_respects_budget_and_keeps_owner():
    cfg = ExperimentConfig(monte_carlo_trials=20)
    rng = np.random.default_rng(4)
    models = build_models(cfg, rng)
    result = greedy_select(
        models,
        budget_bits=20,
        qos_min=cfg.qos_min_deflection,
        qos_weights=cfg.qos_weights,
        performance_weights=cfg.performance_weights,
        rng=rng,
    )
    assert result.used_bits <= 20
    assert all(model.owner in result.scheduled[q] for q, model in enumerate(models))
    assert np.all(result.expected_deflection >= 0)

