"""Velocity-bounded target-mobility envelope for the sensing layer.

For a target displacement with norm at most ``R``, the reverse triangle
inequality bounds every UAV-target range change by ``R``.  Under a free-space
power law ``P(d) = P_ref / d^2``, the largest relative power increase is
attained at the shortest possible post-move range, which gives the
conservative envelope bound used by the gate.
"""

from __future__ import annotations

import numpy as np

from .config import ExperimentConfig


def range_perturbation_bound(
    transmitter_positions: np.ndarray,
    target: np.ndarray,
    max_displacement: float,
) -> tuple[float, float]:
    """Return (max absolute range change, minimum UAV-target range)."""
    max_displacement = float(max_displacement)
    if max_displacement < 0.0:
        raise ValueError("max_displacement must be nonnegative")
    distances = np.linalg.norm(
        np.asarray(transmitter_positions, dtype=float)
        - np.asarray(target, dtype=float),
        axis=1,
    )
    return max_displacement, float(distances.min())


def path_loss_relative_bound(
    transmitter_positions: np.ndarray,
    target: np.ndarray,
    max_displacement: float,
) -> float:
    """Conservative relative power increase under free-space path loss."""
    _, minimum_distance = range_perturbation_bound(
        transmitter_positions, target, max_displacement
    )
    closest_after = max(minimum_distance - max_displacement, 1e-9)
    return float(
        (minimum_distance / closest_after) ** 2 - 1.0
    )


def range_snr_relative_bound(
    cfg: ExperimentConfig,
    transmitter_positions: np.ndarray,
    target: np.ndarray,
    max_displacement: float,
) -> float:
    """Bound on the range-derived SNR used by ``build_models``.

    ``build_models`` maps the distance vector ``d`` to
    ``snr_db = snr_hi - span * (d - min(d)) / ptp(d)``.  Under bounded
    displacement ``R`` every entry changes by at most ``R``, the min changes
    by at most ``R``, and ``ptp`` changes by at most ``2R``, so the
    normalized range term changes by at most ``4R / (ptp - 2R)``.  The bound
    is converted to a linear-SNR relative increase.
    """
    max_displacement = float(max_displacement)
    if max_displacement < 0.0:
        raise ValueError("max_displacement must be nonnegative")
    distances = np.linalg.norm(
        np.asarray(transmitter_positions, dtype=float)
        - np.asarray(target, dtype=float),
        axis=1,
    )
    ptp = float(np.ptp(distances))
    if ptp <= 2.0 * max_displacement:
        return float("inf")
    normalized_change = 4.0 * max_displacement / (ptp - 2.0 * max_displacement)
    snr_lo, snr_hi = cfg.otfs.snr_db_range
    max_db_change = (snr_hi - snr_lo) * normalized_change
    return float(10.0 ** (max_db_change / 10.0) - 1.0)


def verify_range_snr_envelope(
    cfg: ExperimentConfig,
    transmitter_positions: np.ndarray,
    target: np.ndarray,
    max_displacement: float,
    *,
    samples: int = 5_000,
    seed: int = 0,
) -> dict:
    """Sample bounded displacements and check the actual range-SNR bound."""
    max_displacement = float(max_displacement)
    rng = np.random.default_rng(seed)
    positions = np.asarray(transmitter_positions, dtype=float)
    base_target = np.asarray(target, dtype=float)
    base_distances = np.linalg.norm(positions - base_target, axis=1)
    snr_lo, snr_hi = cfg.otfs.snr_db_range
    span = snr_hi - snr_lo

    def db_from_distances(distances: np.ndarray) -> np.ndarray:
        ptp = max(float(np.ptp(distances)), 1e-9)
        minimum = float(distances.min())
        return snr_hi - span * (distances - minimum) / ptp

    base_db = db_from_distances(base_distances)
    base_snr = 10.0 ** (base_db / 10.0)
    bound = range_snr_relative_bound(
        cfg, positions, base_target, max_displacement
    )
    if not np.isfinite(bound):
        return {
            "max_displacement": max_displacement,
            "range_snr_relative_bound": None,
            "samples": samples,
            "snr_violations": 0,
            "passed": False,
            "reason": "ptp <= 2R; no finite range-SNR bound",
        }
    violations = 0
    for _ in range(samples):
        direction = rng.standard_normal(3)
        norm = float(np.linalg.norm(direction))
        radius = max_displacement if norm <= 1e-12 else (
            max_displacement * min(1.0, rng.random())
        )
        displacement = (direction / max(norm, 1e-12)) * radius
        moved_distances = np.linalg.norm(
            positions - (base_target + displacement), axis=1
        )
        moved_snr = 10.0 ** (db_from_distances(moved_distances) / 10.0)
        if np.any(
            np.abs(moved_snr - base_snr) / base_snr
            > bound + 1e-9
        ):
            violations += 1
    return {
        "max_displacement": max_displacement,
        "range_snr_relative_bound": float(bound),
        "samples": samples,
        "snr_violations": violations,
        "passed": violations == 0,
    }


def verify_displacement_envelope(
    transmitter_positions: np.ndarray,
    target: np.ndarray,
    max_displacement: float,
    *,
    samples: int = 5_000,
    seed: int = 0,
) -> dict:
    """Sample bounded displacements and check the range/power bounds."""
    max_displacement = float(max_displacement)
    rng = np.random.default_rng(seed)
    positions = np.asarray(transmitter_positions, dtype=float)
    base_target = np.asarray(target, dtype=float)
    base_distances = np.linalg.norm(positions - base_target, axis=1)
    base_power = 1.0 / np.maximum(base_distances, 1e-9) ** 2
    bound = path_loss_relative_bound(
        positions, base_target, max_displacement
    )
    range_violations = 0
    power_violations = 0
    for _ in range(samples):
        direction = rng.standard_normal(3)
        norm = float(np.linalg.norm(direction))
        radius = max_displacement if norm <= 1e-12 else (
            max_displacement * min(1.0, rng.random())
        )
        displacement = (direction / max(norm, 1e-12)) * radius
        moved = base_target + displacement
        moved_distances = np.linalg.norm(positions - moved, axis=1)
        if np.any(np.abs(moved_distances - base_distances) > max_displacement + 1e-9):
            range_violations += 1
        moved_power = 1.0 / np.maximum(moved_distances, 1e-9) ** 2
        if np.any(
            np.abs(moved_power - base_power) / base_power
            > bound + 1e-9
        ):
            power_violations += 1
    return {
        "max_displacement": max_displacement,
        "path_loss_relative_bound": float(bound),
        "samples": samples,
        "range_violations": range_violations,
        "power_violations": power_violations,
        "passed": range_violations == 0 and power_violations == 0,
    }
