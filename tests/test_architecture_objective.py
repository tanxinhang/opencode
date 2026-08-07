import numpy as np

from uav_otfs_isac.architecture_objective import (
    aperture_constants,
    deflection_surrogate,
    derived_surrogate_objective,
    optimal_aperture_formula,
    waterfilling_allocation,
)
from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.scenario import target_geometry, uav_geometry


def _setup():
    config = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=256,
        weak_target_id=2,
    )
    targets = [target_geometry(q) for q in range(3)]
    transmitters = uav_geometry(4)
    receiver = np.array([0.0, 0.0, 0.0])
    return config, transmitters, targets, receiver


def test_aperture_constants_positive_and_weak_target_largest():
    config, transmitters, targets, receiver = _setup()
    constants = aperture_constants(
        config, transmitters, targets, receiver, aperture_scale=1e-2
    )
    assert np.all(constants > 0.0)
    assert constants[-1] > constants[0]


def test_optimal_aperture_formula_satisfies_first_order_condition():
    kappa = 1e-4
    total_budget = 20.0
    phase_bits = 3
    coherence_frames = 256
    rate = phase_bits / coherence_frames
    aperture = optimal_aperture_formula(
        total_budget, phase_bits, coherence_frames, kappa
    )
    assert aperture is not None
    first_order = (
        5.0 * kappa * rate * aperture**2
        - 4.0 * kappa * total_budget * aperture
        + rate
    )
    assert abs(first_order) < 1e-6


def test_derived_surrogate_increases_then_decreases():
    kappa = 1e-6
    values = [
        derived_surrogate_objective(
            n, total_budget=20.0, phase_bits=3,
            coherence_frames=256, kappa=kappa,
        )
        for n in (0.0, 400.0, 1200.0, 1600.0)
    ]
    assert values[1] < values[2]
    assert values[2] > values[3]


def test_deflection_surrogate_shape_and_monotonicity():
    config, transmitters, targets, receiver = _setup()
    constants = aperture_constants(
        config, transmitters, targets, receiver, aperture_scale=1e-2
    )
    base = np.ones(3)
    small = deflection_surrogate((50, 100, 106), 3, constants, base)
    large = deflection_surrogate((50, 100, 212), 3, constants, base)
    assert small.shape == (3,)
    assert np.all(large >= small)


def test_waterfilling_symmetric_targets_is_equal():
    allocation = waterfilling_allocation(
        96, [1e-3, 1e-3, 1e-3], [1.0, 1.0, 1.0]
    )
    assert sum(allocation) == 96
    assert max(allocation) - min(allocation) <= 1


def test_waterfilling_weak_target_gets_more_aperture():
    allocation = waterfilling_allocation(
        96, [1e-3, 1e-3, 1e-3], [1.0, 1.0, 0.1]
    )
    assert sum(allocation) == 96
    assert allocation[2] > allocation[0]
    assert allocation[2] > allocation[1]
