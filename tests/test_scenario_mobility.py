import numpy as np
import pytest

from uav_otfs_isac.config import load_config
from uav_otfs_isac.fusion import optimal_deflection
from uav_otfs_isac.scenario import (
    build_models,
    target_geometry,
    uav_geometry,
)
from uav_otfs_isac.stochastic_mobility import (
    ar1_horizon_prediction,
    ar1_mmse_prediction,
    nominal_target_at,
    stochastic_trajectories,
)


def test_custom_geometry_changes_evidence():
    cfg = load_config("config/demo.yaml")
    rng = np.random.default_rng(cfg.seed)
    positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    baseline = build_models(cfg, rng)
    moved = positions.copy()
    moved[:, 0] += 40.0
    custom = build_models(cfg, rng, transmitter_positions=moved)
    baseline_deflection = optimal_deflection(
        baseline[0].delta, baseline[0].sigma0, {baseline[0].owner}
    )
    custom_deflection = optimal_deflection(
        custom[0].delta, custom[0].sigma0, {custom[0].owner}
    )
    assert abs(custom_deflection - baseline_deflection) > 1e-6


def test_custom_target_positions_shape_validation():
    cfg = load_config("config/demo.yaml")
    rng = np.random.default_rng(cfg.seed)
    with pytest.raises(ValueError):
        build_models(
            cfg, rng, target_positions=[
                np.zeros(2), np.zeros(2), np.zeros(2)
            ]
        )


def test_variable_report_bits_change_model():
    cfg = load_config("config/demo.yaml")
    rng = np.random.default_rng(cfg.seed)
    baseline = build_models(cfg, rng)
    variable = build_models(
        cfg, rng, quantizer_bits_per_uav=[1, 2, 3, 4, 1, 2, 3, 4]
    )
    assert np.any(variable[0].report_bits != baseline[0].report_bits)
    assert not np.allclose(variable[0].mu1, baseline[0].mu1)


def test_interference_reduces_deflection():
    cfg = load_config("config/demo.yaml")
    rng = np.random.default_rng(cfg.seed)
    clean = build_models(cfg, rng)
    rng = np.random.default_rng(cfg.seed)
    noisy = build_models(
        cfg, rng,
        interference_to_noise=np.full(cfg.num_uavs, 10.0),
    )
    clean_deflection = optimal_deflection(
        clean[0].delta, clean[0].sigma0, {clean[0].owner}
    )
    noisy_deflection = optimal_deflection(
        noisy[0].delta, noisy[0].sigma0, {noisy[0].owner}
    )
    assert noisy_deflection < clean_deflection


def test_stochastic_trajectories_shapes_and_bounds():
    cfg = load_config("config/demo.yaml")
    positions, targets, blockages = stochastic_trajectories(
        base_positions=uav_geometry(cfg.num_uavs),
        base_targets=[target_geometry(q) for q in range(cfg.num_targets)],
        seed=7,
        frames=6,
    )
    assert len(positions) == len(targets) == len(blockages) == 6
    assert positions[0].shape == (cfg.num_uavs, 3)
    assert len(targets[0]) == cfg.num_targets
    assert all(np.all(np.isfinite(value)) for value in positions)
    assert all(np.all(np.isfinite(target)) for frame in targets for target in frame)
    assert all(0.002 <= value <= 0.06 for value in blockages)


def test_stochastic_trajectories_seed_reproducible():
    cfg = load_config("config/demo.yaml")
    kwargs = {
        "base_positions": uav_geometry(cfg.num_uavs),
        "base_targets": [target_geometry(q) for q in range(cfg.num_targets)],
        "seed": 11,
        "frames": 5,
    }
    first = stochastic_trajectories(**kwargs)
    second = stochastic_trajectories(**kwargs)
    assert all(np.allclose(a, b) for a, b in zip(first[0], second[0]))
    assert all(
        np.allclose(a, b)
        for a_frame, b_frame in zip(first[1], second[1])
        for a, b in zip(a_frame, b_frame)
    )
    assert np.allclose(first[2], second[2])


def test_ar1_mmse_prediction_outperforms_previous_frame_predictor():
    rng = np.random.default_rng(123)
    rho = 0.8
    sigma = 2.0
    previous = rng.normal(0.0, sigma, size=10000)
    current = rho * previous + np.sqrt(1.0 - rho**2) * sigma * rng.normal(
        size=previous.size
    )
    previous_predictor_errors = np.mean((current - previous) ** 2)
    mmse_errors = np.mean(
        (current - ar1_mmse_prediction(
            previous,
            np.zeros_like(previous),
            np.zeros_like(current),
            rho,
        )) ** 2
    )
    assert mmse_errors < previous_predictor_errors


def test_nominal_target_at_matches_trajectory_trend():
    cfg = load_config("config/demo.yaml")
    base_targets = [target_geometry(q) for q in range(cfg.num_targets)]
    positions, targets, _ = stochastic_trajectories(
        base_positions=uav_geometry(cfg.num_uavs),
        base_targets=base_targets,
        seed=3,
        frames=6,
        target_sigma=0.0,
        position_sigma=0.0,
    )
    for time_index in range(6):
        for q in range(cfg.num_targets):
            expected = nominal_target_at(
                base_targets[q], q, time_index, 6
            )
            assert np.allclose(targets[time_index][q], expected)


def test_ar1_horizon_prediction_error_grows_with_horizon():
    rng = np.random.default_rng(321)
    rho = 0.8
    sigma = 2.0
    x0 = rng.normal(0.0, sigma, size=100000)
    x1 = rho * x0 + np.sqrt(1.0 - rho**2) * sigma * rng.normal(
        size=x0.size
    )
    x2 = rho * x1 + np.sqrt(1.0 - rho**2) * sigma * rng.normal(
        size=x0.size
    )
    error_h1 = np.mean(
        (x1 - ar1_horizon_prediction(
            x0,
            np.zeros_like(x0),
            np.zeros_like(x1),
            rho,
            1,
        )) ** 2
    )
    error_h2 = np.mean(
        (x2 - ar1_horizon_prediction(
            x0,
            np.zeros_like(x0),
            np.zeros_like(x2),
            rho,
            2,
        )) ** 2
    )
    assert error_h1 < error_h2
