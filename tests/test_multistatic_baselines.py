import numpy as np

from uav_otfs_isac.multistatic_association import PathCandidate
from uav_otfs_isac.multistatic_baselines import (
    conflict_aware_dbscan_association,
    dbscan_path_association,
)
from uav_otfs_isac.multistatic_targets import KinematicNode, PhysicalTarget, generate_bistatic_paths


def node(position):
    return KinematicNode(position, (0.0, 0.0))


def test_dbscan_estimates_two_targets_without_known_count():
    transmitters = tuple(node((200 * np.cos(a), 200 * np.sin(a)))
                         for a in np.linspace(0, 2 * np.pi, 4, endpoint=False))
    receiver = node((0.0, 0.0))
    targets = (
        PhysicalTarget(0, (80.0, 100.0), (1.0, 2.0)),
        PhysicalTarget(1, (-80.0, 120.0), (-1.0, 1.0)),
    )
    paths = generate_bistatic_paths(transmitters, targets, receiver, 5.9e9)
    candidates = [PathCandidate(
        path.transmitter_id, path.delay_s, path.doppler_hz,
        path.receive_azimuth_rad,
    ) for path in paths]
    groups = dbscan_path_association(
        candidates, transmitters, receiver, 5.9e9,
        position_tolerance_m=10.0,
    )
    assert len(groups) == 2
    assert all(len(group.paths) == 4 for group in groups)


def test_identity_constrained_dbscan_keeps_one_path_per_transmitter():
    transmitters = (node((-200.0, 0.0)), node((200.0, 0.0)))
    receiver = node((0.0, 0.0))
    target = PhysicalTarget(0, (0.0, 100.0), (0.0, 0.0))
    paths = generate_bistatic_paths(transmitters, [target], receiver, 5.9e9)
    candidates = [PathCandidate(
        path.transmitter_id, path.delay_s, path.doppler_hz,
        path.receive_azimuth_rad, 0.8,
    ) for path in paths]
    candidates.append(PathCandidate(
        candidates[0].transmitter_id, candidates[0].delay_s,
        candidates[0].doppler_hz, candidates[0].receive_azimuth_rad, 0.2,
    ))
    groups = dbscan_path_association(
        candidates, transmitters, receiver, 5.9e9,
        position_tolerance_m=10.0, enforce_unique_transmitter=True,
    )
    assert len(groups) == 1
    assert len(groups[0].paths) == 2


def test_conflict_aware_dbscan_splits_close_targets_by_identity_multiplicity():
    transmitters = tuple(node((250 * np.cos(a), 250 * np.sin(a)))
                         for a in np.linspace(0, 2 * np.pi, 4, endpoint=False))
    receiver = node((0.0, 0.0))
    targets = (
        PhysicalTarget(0, (100.0, 100.0), (1.0, 0.0)),
        PhysicalTarget(1, (106.0, 104.0), (-1.0, 0.5)),
    )
    paths = generate_bistatic_paths(transmitters, targets, receiver, 5.9e9)
    candidates = [PathCandidate(
        path.transmitter_id, path.delay_s, path.doppler_hz,
        path.receive_azimuth_rad,
    ) for path in paths]
    groups = conflict_aware_dbscan_association(
        candidates, transmitters, receiver, 5.9e9,
        position_tolerance_m=14.0,
    )
    assert len(groups) == 2
    assert all(len(group.paths) == 4 for group in groups)
