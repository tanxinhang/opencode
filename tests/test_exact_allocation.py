import numpy as np

from uav_otfs_isac.exact_allocation import (
    exact_block_surrogate,
    exact_waterfilling_allocation,
)
from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.scenario import target_geometry


def _config():
    return RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=96,
        weak_target_id=2,
        phase_bits=None,
    )


def _targets():
    return [target_geometry(q) for q in range(3)]


def test_exact_surrogate_all_aperture_to_weak():
    config = _config()
    targets = _targets()
    constants = [1e-5, 1e-5, 2e-5]
    base = [1.0, 1.0, 1.0]
    allocation = [0, 0, config.num_elements]
    values = exact_block_surrogate(
        config, targets, allocation, constants, base
    )
    expected_weak = base[2] * (
        1.0 + constants[2] * config.num_elements**2
    ) ** 2
    assert np.isclose(values[2], expected_weak, atol=1e-6)


def test_exact_waterfilling_preserves_aperture():
    config = _config()
    targets = _targets()
    allocation = exact_waterfilling_allocation(
        config, targets, [1e-5, 1e-5, 2e-5], [1.0, 1.0, 0.1]
    )
    assert sum(allocation) == config.num_elements
    assert all(value >= 0 for value in allocation)


def test_exact_waterfilling_never_worsens_equal_min():
    config = _config()
    targets = _targets()
    constants = [1e-5, 1e-5, 2e-5]
    base = [1.0, 1.0, 0.1]
    equal = [32, 32, 32]
    water = exact_waterfilling_allocation(
        config, targets, constants, base
    )
    equal_min = float(np.min(exact_block_surrogate(
        config, targets, equal, constants, base
    )))
    water_min = float(np.min(exact_block_surrogate(
        config, targets, water, constants, base
    )))
    assert water_min >= equal_min - 1e-9
