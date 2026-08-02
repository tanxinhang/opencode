import numpy as np

from uav_otfs_isac.ablations import (
    deterministic_link_models,
    diagonal_covariance_models,
    evaluate_schedule_on_truth,
)
from uav_otfs_isac.config import ExperimentConfig
from uav_otfs_isac.scenario import build_models
from uav_otfs_isac.selection import greedy_select


def test_ablation_models_only_change_the_intended_assumption():
    cfg = ExperimentConfig(num_uavs=4, num_targets=1, owners=(0,), target_present=(True,),
                           qos_min_deflection=(3.0,), qos_weights=(1.0,), performance_weights=(1.0,))
    models = build_models(cfg, np.random.default_rng(8))
    diagonal = diagonal_covariance_models(models)[0]
    deterministic = deterministic_link_models(models)[0]
    assert np.allclose(diagonal.sigma0, np.diag(np.diag(models[0].sigma0)))
    assert np.allclose(diagonal.mu1, models[0].mu1)
    assert np.all(deterministic.success_prob == 1.0)
    assert np.allclose(deterministic.sigma0, models[0].sigma0)


def test_ablation_schedule_is_rescored_on_truth():
    cfg = ExperimentConfig(num_uavs=4, num_targets=1, owners=(0,), target_present=(True,),
                           qos_min_deflection=(3.0,), qos_weights=(1.0,), performance_weights=(1.0,))
    truth = build_models(cfg, np.random.default_rng(9))
    assumed = deterministic_link_models(truth)
    selected = greedy_select(assumed, 10, cfg.qos_min_deflection, cfg.qos_weights,
                             cfg.performance_weights)
    rescored = evaluate_schedule_on_truth(truth, selected, cfg.qos_min_deflection, cfg.qos_weights)
    assert rescored.scheduled == selected.scheduled
    assert np.all(rescored.expected_deflection <= selected.expected_deflection + 1e-10)
