import numpy as np

from uav_otfs_isac.mobility_envelope import (
    path_loss_relative_bound,
    range_perturbation_bound,
    range_snr_relative_bound,
    verify_range_snr_envelope,
    verify_displacement_envelope,
)
from uav_otfs_isac.robustness_stress import (
    StressProfile,
    stress_target_positions,
)
from uav_otfs_isac.scenario import target_geometry, uav_geometry
from uav_otfs_isac.config import load_config


def test_range_perturbation_bound_follows_reverse_triangle_inequality():
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    target = np.array([5.0, 0.0, 0.0])
    bound, minimum = range_perturbation_bound(positions, target, 2.0)
    assert np.isclose(bound, 2.0)
    assert np.isclose(minimum, 5.0)


def test_path_loss_relative_bound_is_conservative():
    positions = uav_geometry(8)
    target = target_geometry(0)
    result = verify_displacement_envelope(
        positions, target, max_displacement=8.0, samples=5_000
    )
    assert result["passed"]
    assert result["path_loss_relative_bound"] > 0.0


def test_velocity_limit_bounds_stress_target_displacement():
    cfg = load_config("config/demo.yaml")
    profile = StressProfile(
        "velocity_bounded",
        mobility_std=4.0,
        velocity_limit_mps=10.0,
        frame_duration_s=0.5,
    )
    targets = stress_target_positions(cfg, cfg.seed, profile)
    base = [target_geometry(q) for q in range(cfg.num_targets)]
    for perturbed, original in zip(targets, base):
        assert np.linalg.norm(perturbed - original) <= 5.0 + 1e-12


def test_range_snr_envelope_bounds_actual_build_models_law():
    cfg = load_config("config/demo.yaml")
    positions = uav_geometry(cfg.num_uavs)
    target = target_geometry(0)
    bound = range_snr_relative_bound(
        cfg, positions, target, max_displacement=8.0
    )
    assert np.isfinite(bound)
    assert bound > 0.0
    result = verify_range_snr_envelope(
        cfg, positions, target, max_displacement=8.0, samples=5_000
    )
    assert result["passed"]
    assert result["snr_violations"] == 0
