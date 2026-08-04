import numpy as np
import pytest

from uav_otfs_isac.multistatic_association import (
    PathCandidate,
    associate_path_candidates,
    position_from_angle_range,
)
from uav_otfs_isac.multistatic_targets import (
    KinematicNode,
    PhysicalTarget,
    generate_bistatic_paths,
)


CARRIER = 5.9e9


def node(position, velocity=(0.0, 0.0)):
    return KinematicNode(position, velocity)


def candidates_from_paths(paths):
    return [PathCandidate(
        path.transmitter_id, path.delay_s, path.doppler_hz,
        path.receive_azimuth_rad,
    ) for path in paths]


def test_candidate_rejects_confidence_above_probability_range():
    with pytest.raises(ValueError):
        PathCandidate(0, 1e-6, 0.0, 0.0, 1.01)


def test_angle_range_inversion_recovers_target_position():
    transmitter = np.array([-100.0, 30.0])
    receiver = np.array([10.0, -20.0])
    target = np.array([45.0, 80.0])
    total_range = np.linalg.norm(target - transmitter) + np.linalg.norm(target - receiver)
    angle = np.arctan2(*(target - receiver)[::-1])
    estimate = position_from_angle_range(transmitter, receiver, angle, total_range)
    assert np.allclose(estimate, target)


def test_unknown_cardinality_association_recovers_two_targets_and_states():
    transmitters = (node((-120.0, 0.0)), node((0.0, -120.0)), node((120.0, 0.0)))
    receiver = node((0.0, 0.0))
    targets = (
        PhysicalTarget(3, (45.0, 100.0), (4.0, -2.0)),
        PhysicalTarget(8, (-60.0, 110.0), (-3.0, 1.5)),
    )
    paths = generate_bistatic_paths(transmitters, targets, receiver, CARRIER)
    groups = associate_path_candidates(
        candidates_from_paths(paths), transmitters, receiver, CARRIER,
        angle_tolerance_rad=np.deg2rad(2.0), position_tolerance_m=5.0,
        doppler_tolerance_hz=2.0,
    )

    assert len(groups) == 2
    estimates = sorted(groups, key=lambda group: group.position[0])
    truth = sorted(targets, key=lambda target: target.position[0])
    for estimate, target in zip(estimates, truth):
        assert len(estimate.paths) == len(transmitters)
        assert np.allclose(estimate.position, target.position, atol=1e-6)
        assert np.allclose(estimate.velocity, target.velocity, atol=1e-6)


def test_spatial_index_preserves_exhaustive_association_result():
    transmitters = tuple(node((200.0 * np.cos(angle), 200.0 * np.sin(angle)))
                         for angle in np.linspace(0.0, 2 * np.pi, 8, endpoint=False))
    receiver = node((0.0, 0.0))
    targets = tuple(PhysicalTarget(
        index, (80.0 - 30.0 * index, 100.0 + 20.0 * index),
        (2.0 - index, 1.0 + index),
    ) for index in range(3))
    candidates = candidates_from_paths(generate_bistatic_paths(
        transmitters, targets, receiver, CARRIER
    ))
    arguments = dict(
        angle_tolerance_rad=np.deg2rad(2.0), position_tolerance_m=5.0,
        doppler_tolerance_hz=2.0,
    )
    indexed = associate_path_candidates(
        candidates, transmitters, receiver, CARRIER,
        use_spatial_index=True, **arguments,
    )
    exhaustive = associate_path_candidates(
        candidates, transmitters, receiver, CARRIER,
        use_spatial_index=False, **arguments,
    )
    assert len(indexed) == len(exhaustive)
    for left, right in zip(indexed, exhaustive):
        assert left.paths == right.paths
        assert np.allclose(left.position, right.position)
        assert np.allclose(left.velocity, right.velocity)


def test_false_singleton_and_missed_path_do_not_set_target_count():
    transmitters = (node((-100.0, 0.0)), node((0.0, -100.0)), node((100.0, 0.0)))
    receiver = node((0.0, 0.0))
    target = PhysicalTarget(0, (20.0, 100.0), (2.0, 1.0))
    paths = generate_bistatic_paths(transmitters, [target], receiver, CARRIER)
    candidates = candidates_from_paths(paths[:2])
    candidates.append(PathCandidate(2, 2e-6, 900.0, -2.0, 0.2))
    groups = associate_path_candidates(
        candidates, transmitters, receiver, CARRIER,
        angle_tolerance_rad=np.deg2rad(3.0), position_tolerance_m=10.0,
        doppler_tolerance_hz=5.0, min_transmitters=2,
    )
    assert len(groups) == 1
    assert len(groups[0].paths) == 2


def test_paths_from_same_transmitter_cannot_form_target_group():
    transmitters = (node((-100.0, 0.0)), node((100.0, 0.0)))
    receiver = node((0.0, 0.0))
    candidate = PathCandidate(0, 1e-6, 0.0, np.pi / 3)
    groups = associate_path_candidates(
        [candidate, candidate], transmitters, receiver, CARRIER,
        angle_tolerance_rad=0.1, position_tolerance_m=20.0,
        doppler_tolerance_hz=5.0,
    )
    assert groups == ()


def test_invalid_shorter_than_direct_range_is_rejected():
    with pytest.raises(ValueError, match="shorter"):
        position_from_angle_range((-100.0, 0.0), (100.0, 0.0), 0.2, 50.0)
