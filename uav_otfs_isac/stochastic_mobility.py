"""Stochastic AR(1) mobility model for UAV-ISAC frames.

The trajectory is a deterministic rotation/sinusoidal trend plus an
AR(1)-correlated random perturbation.  This gives temporal correlation
without a continuous-time stochastic differential model; the frame-level
declaration is explicit in the paper.
"""

from __future__ import annotations

import numpy as np


def rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    result = points.copy()
    result[:, 0] = cosine * points[:, 0] - sine * points[:, 1]
    result[:, 1] = sine * points[:, 0] + cosine * points[:, 1]
    return result


def nominal_target_at(
    base_target: np.ndarray,
    target_id: int,
    time_index: int,
    frames: int,
) -> np.ndarray:
    """Deterministic nominal target position at a frame."""
    phase = 2.0 * np.pi * time_index / frames
    offset = np.array([
        2.0 * np.sin(phase + target_id),
        1.5 * np.cos(phase + target_id),
        0.0,
    ])
    return np.asarray(base_target, dtype=float) + offset


def ar1_mmse_prediction(
    previous_true: np.ndarray,
    previous_nominal: np.ndarray,
    current_nominal: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """Conditional-mean AR(1) prediction of a position vector."""
    return np.asarray(current_nominal, dtype=float) + correlation * (
        np.asarray(previous_true, dtype=float)
        - np.asarray(previous_nominal, dtype=float)
    )


def ar1_horizon_prediction(
    previous_true: np.ndarray,
    previous_nominal: np.ndarray,
    current_nominal: np.ndarray,
    correlation: float,
    horizon: int,
) -> np.ndarray:
    """H-step AR(1) conditional-mean prediction.

    For a stationary AR(1) process with coefficient ``rho``,
    ``E[x_{t+h} | x_t] = rho^h x_t`` and the prediction-error covariance is
    ``(1 - rho^{2h}) sigma^2 I``.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return ar1_mmse_prediction(
        previous_true,
        previous_nominal,
        current_nominal,
        float(correlation) ** int(horizon),
    )


def stochastic_trajectories(
    *,
    base_positions: np.ndarray,
    base_targets: list[np.ndarray],
    seed: int,
    frames: int,
    position_sigma: float = 4.0,
    target_sigma: float = 2.0,
    correlation: float = 0.8,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[float]]:
    """AR(1) trajectory with a bounded sinusoidal trend and random blockage."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    rng = np.random.default_rng(seed)
    num_uavs = base_positions.shape[0]
    num_targets = len(base_targets)
    position_perturb = np.zeros((frames, num_uavs, 3))
    target_perturb = np.zeros((frames, num_targets, 3))
    scale = np.sqrt(max(0.0, 1.0 - correlation**2))
    for time_index in range(1, frames):
        position_perturb[time_index] = (
            correlation * position_perturb[time_index - 1]
            + scale * position_sigma * rng.standard_normal((num_uavs, 3))
        )
        target_perturb[time_index] = (
            correlation * target_perturb[time_index - 1]
            + scale * target_sigma * rng.standard_normal((num_targets, 3))
        )
    positions = []
    targets = []
    blockages = []
    for time_index in range(frames):
        angle = 2.0 * np.pi * time_index / frames * 0.15
        positions.append(
            rotate_z(base_positions, angle) + position_perturb[time_index]
        )
        frame_targets = []
        for q, target in enumerate(base_targets):
            frame_targets.append(
                nominal_target_at(target, q, time_index, frames)
                + target_perturb[time_index][q]
            )
        targets.append(frame_targets)
        blockage = float(np.clip(
            0.005 + 0.02 * (0.5 + 0.5 * np.sin(
                2.0 * np.pi * time_index / frames
            )) + rng.normal(0.0, 0.003),
            0.002,
            0.06,
        ))
        blockages.append(blockage)
    return positions, targets, blockages
