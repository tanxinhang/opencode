"""Fair unknown-cardinality baselines for path-to-target association."""

from collections import deque
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .multistatic_association import (
    PathCandidate,
    TargetGroup,
    _fit_group,
    position_from_angle_range,
)
from .multistatic_targets import KinematicNode


def _project(
    candidates: Iterable[PathCandidate],
    transmitters: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    propagation_speed: float,
) -> list[tuple[PathCandidate, np.ndarray]]:
    projected = []
    for candidate in candidates:
        if candidate.transmitter_id >= len(transmitters):
            raise ValueError("candidate transmitter_id is out of range")
        try:
            position = position_from_angle_range(
                transmitters[candidate.transmitter_id].position,
                receiver.position,
                candidate.receive_azimuth_rad,
                candidate.delay_s * propagation_speed,
            )
        except ValueError:
            continue
        projected.append((candidate, position))
    return projected


def _dbscan_labels(features: np.ndarray, radius: float, min_samples: int) -> np.ndarray:
    """Small deterministic DBSCAN implementation using a k-d tree."""
    if len(features) == 0:
        return np.empty(0, dtype=int)
    neighbors = cKDTree(features).query_ball_point(features, radius)
    labels = np.full(len(features), -2, dtype=int)  # -2 unvisited, -1 noise
    cluster = 0
    for start in range(len(features)):
        if labels[start] != -2:
            continue
        if len(neighbors[start]) < min_samples:
            labels[start] = -1
            continue
        labels[start] = cluster
        queue = deque(neighbors[start])
        queued = set(neighbors[start])
        while queue:
            index = queue.popleft()
            if labels[index] == -1:
                labels[index] = cluster
            if labels[index] != -2:
                continue
            labels[index] = cluster
            if len(neighbors[index]) >= min_samples:
                for neighbor in neighbors[index]:
                    if neighbor not in queued:
                        queued.add(neighbor)
                        queue.append(neighbor)
        cluster += 1
    return labels


def dbscan_path_association(
    candidates: Iterable[PathCandidate],
    transmitters: Iterable[KinematicNode],
    receiver: KinematicNode,
    carrier_hz: float,
    *,
    position_tolerance_m: float,
    min_samples: int = 2,
    angle_tolerance_rad: float | None = None,
    enforce_unique_transmitter: bool = False,
    propagation_speed: float = 299_792_458.0,
) -> tuple[TargetGroup, ...]:
    """DBSCAN on reconstructed position, optionally augmented by angle.

    This baseline estimates the target count from density-connected components.
    When identity enforcement is requested, only the highest-confidence path
    per transmitter is retained in each component; discarded conflicts are not
    reassigned.
    """
    nodes = tuple(transmitters)
    projected = _project(candidates, nodes, receiver, propagation_speed)
    if not projected:
        return ()
    positions = np.asarray([entry[1] for entry in projected])
    if angle_tolerance_rad is None:
        features = positions / position_tolerance_m
    else:
        angles = np.asarray([entry[0].receive_azimuth_rad for entry in projected])
        features = np.column_stack((
            positions / position_tolerance_m,
            np.cos(angles) / angle_tolerance_rad,
            np.sin(angles) / angle_tolerance_rad,
        ))
    labels = _dbscan_labels(features, radius=1.0, min_samples=min_samples)
    groups = []
    for label in range(int(labels.max()) + 1):
        members = [projected[index] for index in np.flatnonzero(labels == label)]
        if enforce_unique_transmitter:
            best_by_transmitter = {}
            for member in members:
                transmitter_id = member[0].transmitter_id
                incumbent = best_by_transmitter.get(transmitter_id)
                if incumbent is None or member[0].confidence > incumbent[0].confidence:
                    best_by_transmitter[transmitter_id] = member
            members = list(best_by_transmitter.values())
        if len(members) < min_samples:
            continue
        groups.append(_fit_group(
            tuple(member[0] for member in members),
            tuple(member[1] for member in members),
            nodes, receiver, carrier_hz, propagation_speed,
        ))
    groups.sort(key=lambda group: float(np.arctan2(
        group.position[1] - receiver.position[1],
        group.position[0] - receiver.position[0],
    )))
    return tuple(groups)


def conflict_aware_dbscan_association(
    candidates: Iterable[PathCandidate],
    transmitters: Iterable[KinematicNode],
    receiver: KinematicNode,
    carrier_hz: float,
    *,
    position_tolerance_m: float,
    min_samples: int = 2,
    propagation_speed: float = 299_792_458.0,
    maximum_iterations: int = 20,
) -> tuple[TargetGroup, ...]:
    """DBSCAN with local splitting only for repeated-transmitter conflicts."""
    nodes = tuple(transmitters)
    projected = _project(candidates, nodes, receiver, propagation_speed)
    if not projected:
        return ()
    positions = np.asarray([entry[1] for entry in projected])
    labels = _dbscan_labels(
        positions / position_tolerance_m, radius=1.0, min_samples=min_samples
    )
    output = []
    for label in range(int(labels.max()) + 1):
        members = [projected[index] for index in np.flatnonzero(labels == label)]
        by_transmitter: dict[int, list[tuple[PathCandidate, np.ndarray]]] = {}
        for member in members:
            by_transmitter.setdefault(member[0].transmitter_id, []).append(member)
        local_count = max(map(len, by_transmitter.values()))
        if local_count == 1:
            output.append(_fit_group(
                tuple(member[0] for member in members),
                tuple(member[1] for member in members),
                nodes, receiver, carrier_hz, propagation_speed,
            ))
            continue

        anchor = max(
            by_transmitter.values(),
            key=lambda group: (len(group), sum(item[0].confidence for item in group)),
        )
        centers = np.asarray([
            item[1] for item in sorted(anchor, key=lambda item: tuple(item[1]))
        ])
        assignments: list[list[tuple[PathCandidate, np.ndarray]]] = []
        for _ in range(maximum_iterations):
            assignments = [[] for _ in range(local_count)]
            for transmitter_members in by_transmitter.values():
                member_positions = np.asarray([item[1] for item in transmitter_members])
                costs = np.linalg.norm(
                    member_positions[:, None, :] - centers[None, :, :], axis=2
                )
                rows, columns = linear_sum_assignment(costs)
                for row, column in zip(rows, columns):
                    assignments[int(column)].append(transmitter_members[int(row)])
            updated = centers.copy()
            for index, group in enumerate(assignments):
                if group:
                    updated[index] = np.average(
                        np.asarray([item[1] for item in group]), axis=0,
                        weights=np.asarray([item[0].confidence for item in group]),
                    )
            if np.allclose(updated, centers, atol=1e-6):
                break
            centers = updated
        for group in assignments:
            if len({item[0].transmitter_id for item in group}) < min_samples:
                continue
            output.append(_fit_group(
                tuple(item[0] for item in group),
                tuple(item[1] for item in group),
                nodes, receiver, carrier_hz, propagation_speed,
            ))
    output.sort(key=lambda group: float(np.arctan2(
        group.position[1] - receiver.position[1],
        group.position[0] - receiver.position[0],
    )))
    return tuple(output)
