import numpy as np

from uav_otfs_isac.ris_optimization import target_direction_cosines
from uav_otfs_isac.ris_scenario import RisConfig, ris_array_gain
from uav_otfs_isac.ris_subarray import (
    aperture_allocation_gains,
    bounded_multi_move_certificate,
    coordinate_block_steering_ascent,
    coordinate_aperture_ascent,
    exact_single_move_gradients,
    multi_beam_phase,
)
from uav_otfs_isac.scenario import target_geometry


def _config():
    return RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=96,
        weak_target_id=2,
        phase_bits=None,
    )


def test_multi_beam_phase_preserves_total_aperture():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    allocation = (32, 32, 32)
    phase = multi_beam_phase(config, targets, allocation)
    assert phase.shape == (config.num_elements,)
    assert np.all(np.isfinite(phase))


def test_all_aperture_to_one_target_gives_unit_gain():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    weak_index = 2
    allocation = [0, 0, 0]
    allocation[weak_index] = config.num_elements
    phase = multi_beam_phase(config, targets, allocation)
    gain = ris_array_gain(phase, targets[weak_index], config)
    assert np.isclose(gain, 1.0, atol=1e-9)


def test_coordinate_ascent_preserves_aperture_and_never_worsens():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]

    def objective(allocation):
        return float(np.min(aperture_allocation_gains(
            config, targets, allocation
        )))

    result = coordinate_aperture_ascent(
        config, targets, objective, step_sizes=(16, 8), max_rounds_per_step=3
    )
    assert sum(result["allocation"]) == config.num_elements
    initial = objective(tuple([config.num_elements // 3] * 3))
    assert result["value"] >= initial - 1e-9


def test_allocation_gains_shape():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    gains = aperture_allocation_gains(config, targets, (32, 32, 32))
    assert gains.shape == (3,)
    assert np.all(gains >= 0.0)


def test_multi_beam_phase_accepts_steering_cosines():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    allocation = (32, 32, 32)
    cosines = target_direction_cosines(config, targets)
    default_phase = multi_beam_phase(config, targets, allocation)
    cosine_phase = multi_beam_phase(
        config, targets, allocation, steering_cosines=cosines
    )
    assert np.allclose(default_phase, cosine_phase, atol=1e-9)


def test_block_steering_ascent_never_worsens_objective():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    allocation = (32, 32, 32)

    def objective(cosines):
        phase = multi_beam_phase(
            config, targets, allocation, steering_cosines=cosines
        )
        return float(np.min([
            ris_array_gain(phase, target, config) ** 2
            for target in targets
        ]))

    result = coordinate_block_steering_ascent(
        config, targets, allocation, objective,
        step=0.1, grid_points=5, max_rounds=2,
    )
    initial = objective(target_direction_cosines(config, targets))
    assert result["value"] >= initial - 1e-9
    assert all(-1.0 <= value <= 1.0 for value in result["steering_cosines"])


def test_single_move_gradients_detect_improving_move():
    def objective(allocation):
        return float(sum(
            weight * value for weight, value in zip((1.0, 2.0, 3.0), allocation)
        ))

    result = exact_single_move_gradients(objective, (2, 1, 0))
    assert result["maximum_gradient"] > 1e-9
    assert result["moves"][0]["source"] == 0
    assert result["moves"][0]["target"] == 2
    assert not result["local_optimal"]


def test_single_move_gradients_local_optimal():
    def objective(allocation):
        return float(sum(
            weight * value for weight, value in zip((1.0, 2.0, 3.0), allocation)
        ))

    result = exact_single_move_gradients(objective, (0, 0, 3))
    assert result["maximum_gradient"] <= 1e-9
    assert result["local_optimal"]


def test_coordinate_ascent_accepts_initial_allocation():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]

    def objective(allocation):
        return float(sum(
            weight * value for weight, value in zip((1.0, 2.0, 3.0), allocation)
        ))

    result = coordinate_aperture_ascent(
        config, targets, objective,
        step_sizes=(8, 4), max_rounds_per_step=2,
        initial_allocation=(16, 32, 48),
    )
    assert sum(result["allocation"]) == config.num_elements
    assert result["value"] >= objective((16, 32, 48)) - 1e-9


def test_bounded_multi_move_certificate_detects_improvement():
    def objective(allocation):
        return float(sum(
            weight * value for weight, value in zip((1.0, 2.0, 3.0), allocation)
        ))

    result = bounded_multi_move_certificate(
        objective, (2, 1, 0), max_transfer=1
    )
    assert result["improved"]
    assert not result["local_optimal"]
    assert result["allocation"] == (1, 1, 1)


def test_bounded_multi_move_certificate_local_optimal():
    def objective(allocation):
        return float(sum(
            weight * value for weight, value in zip((1.0, 2.0, 3.0), allocation)
        ))

    result = bounded_multi_move_certificate(
        objective, (0, 0, 3), max_transfer=2
    )
    assert result["local_optimal"]
    assert result["allocation"] == (0, 0, 3)
