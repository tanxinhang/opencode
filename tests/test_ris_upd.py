import numpy as np

from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.ris_upd import (
    upd_array_gain,
    upd_ideal_phase,
    upd_physics_gain_matrix,
)
from uav_otfs_isac.scenario import target_geometry, uav_geometry


def _config():
    return RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=256,
        aperture_shape=(16, 16),
        weak_target_id=2,
        phase_bits=None,
    )


def test_upd_aligned_gain_is_one():
    config = _config()
    target = target_geometry(1)
    phase = upd_ideal_phase(config, target)
    assert np.isclose(upd_array_gain(phase, target, config), 1.0, atol=1e-9)


def test_upd_gain_matrix_shape_and_lower_bound():
    config = _config()
    targets = [target_geometry(q) for q in range(3)]
    transmitters = uav_geometry(4)
    receiver = np.array([0.0, 0.0, 0.0])
    phases = [upd_ideal_phase(config, target) for target in targets]
    gain = upd_physics_gain_matrix(
        config, transmitters, targets, receiver, 1e-2,
        phase_per_target=phases,
    )
    assert gain.shape == (3, 4)
    # P1-1: clean targets keep the >= 1 floor; the blocked weak target is
    # ``direct_blockage + P_ris/P_dir`` and stays below 1.
    strong = np.delete(gain, 2, axis=0)
    assert np.all(strong >= 1.0)
    assert np.all(gain[2] >= 0.01)
