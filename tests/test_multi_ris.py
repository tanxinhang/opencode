import numpy as np

from uav_otfs_isac.multi_ris import (
    multi_ris_control_overhead,
    multi_ris_physics_gain_matrix,
)
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import target_geometry, uav_geometry


def _setup():
    transmitters = uav_geometry(4)
    targets = [target_geometry(q) for q in range(3)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=128,
        weak_target_id=2,
        phase_bits=3,
    )
    return transmitters, targets, receiver, ris


def test_single_ris_matches_existing_model():
    transmitters, targets, receiver, ris = _setup()
    phases = [[ris_beam_phase_for(ris, target) for target in targets]]
    multi = multi_ris_physics_gain_matrix(
        [ris], transmitters, targets, receiver, 1e-2,
        phases_per_ris=phases,
    )
    existing = ris_physics_gain_matrix(
        ris, transmitters, targets, receiver, 1e-2,
        direct_blockage=0.01, phase_per_target=phases[0],
    )
    assert np.allclose(multi, existing, atol=1e-12)


def ris_beam_phase_for(ris, target):
    from uav_otfs_isac.ris_scenario import ris_beam_phase
    return ris_beam_phase(target, ris)


def test_two_ris_gain_is_not_lower():
    transmitters, targets, receiver, ris = _setup()
    second = RisConfig(
        position=np.array([20.0, 10.0, 2.0]),
        num_elements=128,
        weak_target_id=2,
        phase_bits=3,
    )
    phases = [
        [ris_beam_phase_for(ris, target) for target in targets],
        [ris_beam_phase_for(second, target) for target in targets],
    ]
    single = multi_ris_physics_gain_matrix(
        [ris], transmitters, targets, receiver, 1e-2,
        phases_per_ris=[phases[0]],
    )
    double = multi_ris_physics_gain_matrix(
        [ris, second], transmitters, targets, receiver, 1e-2,
        phases_per_ris=phases,
    )
    assert np.all(double >= single - 1e-12)


def test_control_overhead_is_additive():
    _, _, _, ris = _setup()
    second = RisConfig(
        position=np.array([20.0, 10.0, 2.0]),
        num_elements=64,
        weak_target_id=2,
        phase_bits=3,
    )
    total = multi_ris_control_overhead([ris, second], coherence_frames=64)
    assert np.isclose(total, (128 + 64) * 3 / 64)
