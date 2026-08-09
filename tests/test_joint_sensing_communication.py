from dataclasses import replace

import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.expected_pd import (
    expected_pd_greedy_select,
    reliability_weighted_report_cost,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.robustness_stress import StressProfile, build_stress_models


def test_reliability_weighted_cost_grows_with_flip_and_erasure():
    base = symmetric_diversity_model(
        np.array([1.4, 0.1, 0.1, 0.1]), success_probability=0.9
    )
    clean = reliability_weighted_report_cost(base, 1)
    flipped = replace(
        base,
        bit_flip_prob=np.array([0.0, 0.2, 0.2, 0.2, 0.2]),
    )
    assert reliability_weighted_report_cost(flipped, 1) > clean
    erased = replace(
        base,
        success_prob=np.array([1.0, 0.5, 0.5, 0.5, 0.5]),
    )
    assert reliability_weighted_report_cost(erased, 1) > clean


def test_reliability_weighted_greedy_respects_budget():
    cfg = load_config("config/demo.yaml")
    models = build_stress_models(
        cfg,
        cfg.seed,
        StressProfile(
            "combined",
            interference_to_noise=10.0,
            bit_flip_probability=0.10,
            success_probability_scale=0.8,
            mobility_std=4.0,
        ),
    )
    selection = expected_pd_greedy_select(
        models,
        20,
        cfg.false_alarm_rate,
        grid=32,
        cost_mode="reliability_weighted",
    )
    assert selection.used_bits <= 20
    assert np.all(np.isfinite(selection.expected_pd))
    assert np.all((selection.expected_pd >= 0.0) & (selection.expected_pd <= 1.0))


def test_expected_pd_greedy_rejects_unknown_cost_mode():
    cfg = load_config("config/demo.yaml")
    models = build_stress_models(cfg, cfg.seed, StressProfile("clean"))
    try:
        expected_pd_greedy_select(
            models, 20, cfg.false_alarm_rate,
            grid=32, cost_mode="unknown",
        )
    except ValueError as error:
        assert "cost_mode" in str(error)
    else:
        raise AssertionError("unknown cost_mode must be rejected")
