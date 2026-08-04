"""Unknown-cardinality association of multistatic path candidates.

This is a deliberately small Gate G0-B receiver back end.  It assumes a path
front end has supplied transmitter identity, receive angle, delay, Doppler,
and confidence.  It never receives the number of physical targets.  Each path
is first mapped from its angle/range pair to a 2-D position estimate; compatible
paths from distinct transmitters are then grouped and jointly refined.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from .multistatic_targets import KinematicNode


@dataclass(frozen=True)
class PathCandidate:
    transmitter_id: int
    delay_s: float
    doppler_hz: float
    receive_azimuth_rad: float
    confidence: float = 1.0
    range_sigma_m: float | None = None
    angle_sigma_rad: float | None = None
    doppler_sigma_hz: float | None = None

    def __post_init__(self) -> None:
        values = (self.delay_s, self.doppler_hz, self.receive_azimuth_rad,
                  self.confidence)
        if self.transmitter_id < 0:
            raise ValueError("transmitter_id must be nonnegative")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("candidate values must be finite")
        if self.delay_s <= 0 or not 0 < self.confidence <= 1:
            raise ValueError("delay_s must be positive and confidence must lie in (0, 1]")
        for name, value in (
            ("range_sigma_m", self.range_sigma_m),
            ("angle_sigma_rad", self.angle_sigma_rad),
            ("doppler_sigma_hz", self.doppler_sigma_hz),
        ):
            if value is not None and (
                not np.isfinite(value) or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite when supplied")


@dataclass(frozen=True)
class TargetGroup:
    paths: tuple[PathCandidate, ...]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    residual: float


def _angle_difference(left: float, right: float) -> float:
    return float(abs(np.angle(np.exp(1j * (left - right)))))


def position_from_angle_range(
    transmitter_position: ArrayLike,
    receiver_position: ArrayLike,
    receive_azimuth_rad: float,
    bistatic_range_m: float,
) -> NDArray[np.float64]:
    """Intersect a receive-angle ray with a bistatic range ellipse."""
    transmitter = np.asarray(transmitter_position, dtype=float)
    receiver = np.asarray(receiver_position, dtype=float)
    if transmitter.shape != (2,) or receiver.shape != (2,):
        raise ValueError("association currently supports 2-D geometry")
    if not np.isfinite(bistatic_range_m) or bistatic_range_m <= 0:
        raise ValueError("bistatic_range_m must be positive and finite")
    direct_range = float(np.linalg.norm(transmitter - receiver))
    if bistatic_range_m < direct_range - 1e-9:
        raise ValueError("bistatic range is shorter than the direct path")
    direction = np.array([
        np.cos(receive_azimuth_rad), np.sin(receive_azimuth_rad)
    ])
    baseline = transmitter - receiver
    numerator = bistatic_range_m ** 2 - direct_range ** 2
    denominator = 2.0 * (
        bistatic_range_m - float(np.dot(direction, baseline))
    )
    if numerator <= 1e-10 or denominator <= 1e-12:
        raise ValueError("direct-path range does not identify a target distance")
    distance = numerator / denominator
    return receiver + distance * direction


def bistatic_position_covariance(
    transmitter_position,
    receiver_position,
    bistatic_range_m: float,
    receive_azimuth_rad: float,
    range_sigma_m: float = 1.5,
    angle_sigma_rad: float = np.deg2rad(0.4),
    eigenvalue_floor_m2: float = 0.04,
) -> NDArray[np.float64]:
    """Delta-method covariance of bistatic range/bearing position inversion."""
    transmitter = np.asarray(transmitter_position, dtype=float)
    receiver = np.asarray(receiver_position, dtype=float)
    baseline = transmitter - receiver
    rho = float(bistatic_range_m)
    unit = np.asarray((np.cos(receive_azimuth_rad),
                       np.sin(receive_azimuth_rad)))
    tangent = np.asarray((-unit[1], unit[0]))
    direct_squared = float(np.dot(baseline, baseline))
    projection = float(np.dot(unit, baseline))
    gap = rho - projection
    numerator = rho * rho - direct_squared
    if numerator <= 0 or gap <= 1e-9:
        raise ValueError("bistatic geometry is not locally identifiable")
    distance = numerator / (2.0 * gap)
    distance_range = (
        2.0 * rho * gap - numerator
    ) / (2.0 * gap * gap)
    distance_angle = (
        numerator * float(np.dot(tangent, baseline))
    ) / (2.0 * gap * gap)
    jacobian = np.column_stack((
        distance_range * unit,
        distance_angle * unit + distance * tangent,
    ))
    covariance = jacobian @ np.diag((
        range_sigma_m ** 2, angle_sigma_rad ** 2,
    )) @ jacobian.T
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, eigenvalue_floor_m2)
    return (vectors * values) @ vectors.T


def _position_cell(position: NDArray[np.float64], cell_size: float) -> tuple[int, int]:
    return tuple(np.floor(position / cell_size).astype(int))


def _fit_group(
    paths: tuple[PathCandidate, ...],
    positions: tuple[NDArray[np.float64], ...],
    transmitters: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    propagation_speed: float,
) -> TargetGroup:
    weights = np.asarray([path.confidence for path in paths], dtype=float)
    position = np.average(np.asarray(positions), axis=0, weights=weights)
    rows = []
    observations = []
    offsets = []
    for path in paths:
        transmitter = transmitters[path.transmitter_id]
        tx_leg = position - transmitter.position
        rx_leg = position - receiver.position
        tx_unit = tx_leg / np.linalg.norm(tx_leg)
        rx_unit = rx_leg / np.linalg.norm(rx_leg)
        rows.append(tx_unit + rx_unit)
        offsets.append(np.dot(tx_unit, transmitter.velocity)
                       + np.dot(rx_unit, receiver.velocity))
        observations.append(path.doppler_hz * propagation_speed / carrier_hz)
    matrix = np.asarray(rows)
    right = np.asarray(observations) + np.asarray(offsets)
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    weighted_right = right * np.sqrt(weights)
    velocity = np.linalg.lstsq(weighted_matrix, weighted_right, rcond=None)[0]
    doppler_residual = matrix @ velocity - right
    position_residual = np.linalg.norm(np.asarray(positions) - position, axis=1)
    residual = float(np.sqrt(np.average(
        position_residual ** 2 + doppler_residual ** 2, weights=weights
    )))
    return TargetGroup(paths, position, velocity, residual)


def _maximum_doppler_error(
    group: TargetGroup,
    transmitters: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    propagation_speed: float,
) -> float:
    errors = []
    for candidate in group.paths:
        transmitter = transmitters[candidate.transmitter_id]
        tx_leg = group.position - transmitter.position
        rx_leg = group.position - receiver.position
        tx_unit = tx_leg / np.linalg.norm(tx_leg)
        rx_unit = rx_leg / np.linalg.norm(rx_leg)
        rate = (
            np.dot(tx_unit, group.velocity - transmitter.velocity)
            + np.dot(rx_unit, group.velocity - receiver.velocity)
        )
        predicted = carrier_hz * rate / propagation_speed
        errors.append(abs(predicted - candidate.doppler_hz))
    return float(max(errors, default=0.0))


def associate_path_candidates(
    candidates: Iterable[PathCandidate],
    transmitters: Iterable[KinematicNode],
    receiver: KinematicNode,
    carrier_hz: float,
    *,
    angle_tolerance_rad: float,
    position_tolerance_m: float,
    doppler_tolerance_hz: float,
    min_transmitters: int = 2,
    propagation_speed: float = 299_792_458.0,
    use_spatial_index: bool = True,
) -> tuple[TargetGroup, ...]:
    """Associate paths without being given the number of targets.

    A candidate can join a group only if its transmitter is not already used,
    its angle and reconstructed position agree with every group member, and a
    joint velocity fit does not produce excessive Doppler residual.  Complete-
    link compatibility avoids the chaining failure of single-link clustering.
    """
    paths = tuple(candidates)
    nodes = tuple(transmitters)
    if not nodes:
        raise ValueError("at least one transmitter is required")
    if receiver.position.size != 2 or any(node.position.size != 2 for node in nodes):
        raise ValueError("association currently supports 2-D geometry")
    if carrier_hz <= 0 or propagation_speed <= 0:
        raise ValueError("carrier_hz and propagation_speed must be positive")
    if angle_tolerance_rad <= 0 or position_tolerance_m <= 0:
        raise ValueError("angle and position tolerances must be positive")
    if doppler_tolerance_hz <= 0 or min_transmitters < 1:
        raise ValueError("doppler tolerance and min_transmitters must be positive")
    if any(path.transmitter_id >= len(nodes) for path in paths):
        raise ValueError("candidate transmitter_id is out of range")

    valid: list[tuple[PathCandidate, NDArray[np.float64]]] = []
    for path in paths:
        try:
            position = position_from_angle_range(
                nodes[path.transmitter_id].position,
                receiver.position,
                path.receive_azimuth_rad,
                path.delay_s * propagation_speed,
            )
        except ValueError:
            continue
        valid.append((path, position))
    valid.sort(key=lambda item: item[0].confidence, reverse=True)

    clusters: list[list[tuple[PathCandidate, NDArray[np.float64]]]] = []
    cluster_cells: list[tuple[int, int]] = []
    spatial_index: dict[tuple[int, int], set[int]] = {}
    for item in valid:
        path, position = item
        options = []
        if use_spatial_index:
            cell = _position_cell(position, position_tolerance_m)
            nearby_indices: set[int] = set()
            for first_offset in (-1, 0, 1):
                for second_offset in (-1, 0, 1):
                    nearby_indices.update(spatial_index.get((
                        cell[0] + first_offset, cell[1] + second_offset
                    ), ()))
            candidate_indices = sorted(nearby_indices)
        else:
            candidate_indices = range(len(clusters))
        for cluster_index in candidate_indices:
            cluster = clusters[cluster_index]
            if any(existing.transmitter_id == path.transmitter_id
                   for existing, _ in cluster):
                continue
            if any(_angle_difference(existing.receive_azimuth_rad,
                                     path.receive_azimuth_rad) > angle_tolerance_rad
                   or np.linalg.norm(existing_position - position)
                   > position_tolerance_m
                   for existing, existing_position in cluster):
                continue
            trial = cluster + [item]
            trial_paths = tuple(entry[0] for entry in trial)
            trial_positions = tuple(entry[1] for entry in trial)
            fitted = _fit_group(
                trial_paths, trial_positions, nodes, receiver, carrier_hz,
                propagation_speed,
            )
            maximum_error = _maximum_doppler_error(
                fitted, nodes, receiver, carrier_hz, propagation_speed
            )
            if maximum_error <= doppler_tolerance_hz:
                options.append((fitted.residual, cluster_index))
        if options:
            _, best_index = min(options)
            clusters[best_index].append(item)
            if use_spatial_index:
                old_cell = cluster_cells[best_index]
                cluster_positions = np.asarray([
                    entry[1] for entry in clusters[best_index]
                ])
                cluster_weights = np.asarray([
                    entry[0].confidence for entry in clusters[best_index]
                ])
                new_cell = _position_cell(
                    np.average(cluster_positions, axis=0, weights=cluster_weights),
                    position_tolerance_m,
                )
                if new_cell != old_cell:
                    spatial_index[old_cell].remove(best_index)
                    spatial_index.setdefault(new_cell, set()).add(best_index)
                    cluster_cells[best_index] = new_cell
        else:
            clusters.append([item])
            new_cell = _position_cell(position, position_tolerance_m)
            cluster_cells.append(new_cell)
            spatial_index.setdefault(new_cell, set()).add(len(clusters) - 1)

    # Greedy insertion can split one target when an early noisy path starts a
    # second fragment. Merge only locally compatible fragments with disjoint
    # transmitter identities; the number of fragments is normally O(N).
    changed = True
    while changed:
        changed = False
        for left_index in range(len(clusters)):
            if changed:
                break
            for right_index in range(left_index + 1, len(clusters)):
                left = clusters[left_index]
                right = clusters[right_index]
                left_ids = {entry[0].transmitter_id for entry in left}
                right_ids = {entry[0].transmitter_id for entry in right}
                if left_ids & right_ids:
                    continue
                if any(
                    _angle_difference(first[0].receive_azimuth_rad,
                                      second[0].receive_azimuth_rad)
                    > angle_tolerance_rad
                    or np.linalg.norm(first[1] - second[1]) > position_tolerance_m
                    for first in left for second in right
                ):
                    continue
                combined = left + right
                fitted = _fit_group(
                    tuple(entry[0] for entry in combined),
                    tuple(entry[1] for entry in combined),
                    nodes, receiver, carrier_hz, propagation_speed,
                )
                if _maximum_doppler_error(
                    fitted, nodes, receiver, carrier_hz, propagation_speed
                ) > doppler_tolerance_hz:
                    continue
                clusters[left_index] = combined
                del clusters[right_index]
                changed = True
                break

    groups = []
    for cluster in clusters:
        if len({entry[0].transmitter_id for entry in cluster}) < min_transmitters:
            continue
        groups.append(_fit_group(
            tuple(entry[0] for entry in cluster),
            tuple(entry[1] for entry in cluster),
            nodes, receiver, carrier_hz, propagation_speed,
        ))
    groups.sort(key=lambda group: float(np.arctan2(
        group.position[1] - receiver.position[1],
        group.position[0] - receiver.position[0],
    )))
    return tuple(groups)
