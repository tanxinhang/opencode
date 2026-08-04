"""Geometry-consistent multistatic paths for concurrent-UAV OTFS-ISAC.

The physical scene contains ``N`` targets, while ``M`` concurrent illuminators
can produce up to ``M * N`` bistatic paths at one array receiver.  Delay,
Doppler, and receive angle are derived from target state rather than treated as
independent path parameters.  This module deliberately stops at the physical
scene layer; an unknown-cardinality receiver must still recover paths and
associate them into targets.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, name: str, dimension: int | None = None) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.size not in (2, 3):
        raise ValueError(f"{name} must be a finite 2-D or 3-D vector")
    if dimension is not None and vector.size != dimension:
        raise ValueError(f"{name} must have dimension {dimension}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector.copy()


@dataclass(frozen=True)
class KinematicNode:
    """Position and velocity of a transmitter or receiver."""

    position: ArrayLike
    velocity: ArrayLike

    def __post_init__(self) -> None:
        position = _vector(self.position, "position")
        velocity = _vector(self.velocity, "velocity", position.size)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


@dataclass(frozen=True)
class PhysicalTarget:
    """One physical target shared by all illuminating UAV paths."""

    target_id: int
    position: ArrayLike
    velocity: ArrayLike

    def __post_init__(self) -> None:
        if self.target_id < 0:
            raise ValueError("target_id must be nonnegative")
        position = _vector(self.position, "target position")
        velocity = _vector(self.velocity, "target velocity", position.size)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


@dataclass(frozen=True)
class BistaticPath:
    """A geometry-consistent transmitter--target--receiver path.

    ``doppler_hz`` uses the positive-range-rate convention: a path whose total
    bistatic length is increasing has positive Doppler.  A receiver using the
    opposite baseband convention should negate this value.
    """

    transmitter_id: int
    target_id: int
    delay_s: float
    doppler_hz: float
    receive_azimuth_rad: float
    bistatic_range_m: float


def bistatic_delay(
    transmitter_position: ArrayLike,
    target_position: ArrayLike,
    receiver_position: ArrayLike,
    propagation_speed: float = 299_792_458.0,
) -> float:
    """Return transmitter--target--receiver propagation delay."""
    tx = _vector(transmitter_position, "transmitter_position")
    target = _vector(target_position, "target_position", tx.size)
    receiver = _vector(receiver_position, "receiver_position", tx.size)
    if not np.isfinite(propagation_speed) or propagation_speed <= 0:
        raise ValueError("propagation_speed must be positive and finite")
    total_range = np.linalg.norm(target - tx) + np.linalg.norm(target - receiver)
    if total_range == 0:
        raise ValueError("degenerate zero-length bistatic path")
    return float(total_range / propagation_speed)


def bistatic_doppler(
    transmitter: KinematicNode,
    target: PhysicalTarget,
    receiver: KinematicNode,
    carrier_hz: float,
    propagation_speed: float = 299_792_458.0,
) -> float:
    """Return Doppler from the time derivative of total bistatic range."""
    if not np.isfinite(carrier_hz) or carrier_hz <= 0:
        raise ValueError("carrier_hz must be positive and finite")
    if not np.isfinite(propagation_speed) or propagation_speed <= 0:
        raise ValueError("propagation_speed must be positive and finite")
    dimension = transmitter.position.size
    if target.position.size != dimension or receiver.position.size != dimension:
        raise ValueError("transmitter, target, and receiver dimensions must match")
    tx_leg = target.position - transmitter.position
    rx_leg = target.position - receiver.position
    tx_distance = np.linalg.norm(tx_leg)
    rx_distance = np.linalg.norm(rx_leg)
    if tx_distance == 0 or rx_distance == 0:
        raise ValueError("target cannot coincide with transmitter or receiver")
    range_rate = (
        np.dot(tx_leg / tx_distance, target.velocity - transmitter.velocity)
        + np.dot(rx_leg / rx_distance, target.velocity - receiver.velocity)
    )
    return float(carrier_hz * range_rate / propagation_speed)


def receive_azimuth(target_position: ArrayLike, receiver_position: ArrayLike) -> float:
    """Return target azimuth at the receiver in radians."""
    target = _vector(target_position, "target_position")
    receiver = _vector(receiver_position, "receiver_position", target.size)
    displacement = target - receiver
    if np.linalg.norm(displacement) == 0:
        raise ValueError("target cannot coincide with receiver")
    return float(np.arctan2(displacement[1], displacement[0]))


def generate_bistatic_paths(
    transmitters: Iterable[KinematicNode],
    targets: Iterable[PhysicalTarget],
    receiver: KinematicNode,
    carrier_hz: float,
    visibility: ArrayLike | None = None,
    propagation_speed: float = 299_792_458.0,
) -> tuple[BistaticPath, ...]:
    """Generate up to ``M * N`` paths in transmitter-major order.

    ``visibility[m, n]`` can disable paths that are blocked or not illuminated.
    Target identifiers must be unique; they need not be contiguous.
    """
    transmitter_list = tuple(transmitters)
    target_list = tuple(targets)
    if not transmitter_list:
        raise ValueError("at least one transmitter is required")
    if not target_list:
        raise ValueError("at least one target is required")
    dimension = receiver.position.size
    if any(node.position.size != dimension for node in transmitter_list):
        raise ValueError("all nodes and targets must use the same dimensions")
    if any(target.position.size != dimension for target in target_list):
        raise ValueError("all nodes and targets must use the same dimensions")
    target_ids = [target.target_id for target in target_list]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_id values must be unique")

    if visibility is None:
        visible = np.ones((len(transmitter_list), len(target_list)), dtype=bool)
    else:
        visible_array = np.asarray(visibility)
        if visible_array.shape != (len(transmitter_list), len(target_list)):
            raise ValueError("visibility must have shape (num_transmitters, num_targets)")
        if visible_array.dtype.kind != "b":
            raise ValueError("visibility must be boolean")
        visible = visible_array

    paths: list[BistaticPath] = []
    for transmitter_id, transmitter in enumerate(transmitter_list):
        for target_index, target in enumerate(target_list):
            if not visible[transmitter_id, target_index]:
                continue
            delay = bistatic_delay(
                transmitter.position, target.position, receiver.position,
                propagation_speed,
            )
            paths.append(BistaticPath(
                transmitter_id=transmitter_id,
                target_id=target.target_id,
                delay_s=delay,
                doppler_hz=bistatic_doppler(
                    transmitter, target, receiver, carrier_hz, propagation_speed
                ),
                receive_azimuth_rad=receive_azimuth(
                    target.position, receiver.position
                ),
                bistatic_range_m=delay * propagation_speed,
            ))
    return tuple(paths)


def group_paths_by_target(
    paths: Iterable[BistaticPath],
) -> dict[int, tuple[BistaticPath, ...]]:
    """Group path-level components into their physical-target truth groups."""
    groups: dict[int, list[BistaticPath]] = {}
    for path in paths:
        groups.setdefault(path.target_id, []).append(path)
    return {target_id: tuple(group) for target_id, group in groups.items()}
