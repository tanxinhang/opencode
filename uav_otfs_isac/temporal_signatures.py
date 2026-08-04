"""Known-state multi-frame signatures for short-window OTFS separation."""

from __future__ import annotations

import numpy as np

from .identifiability import normalized_gram


def validate_transition_matrix(transition):
    """Validate and return a finite row-stochastic transition matrix."""
    matrix = np.asarray(transition, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("transition must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("transition probabilities must be finite and nonnegative")
    if not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-12):
        raise ValueError("transition rows must sum to one")
    return matrix


def sample_markov_path(transition, initial_state, frames, rng):
    """Sample a first-order finite-state path, including its initial state."""
    matrix = validate_transition_matrix(transition)
    if frames <= 0 or int(frames) != frames:
        raise ValueError("frames must be a positive integer")
    if not 0 <= int(initial_state) < matrix.shape[0]:
        raise ValueError("initial_state is outside the state space")
    path = np.empty(int(frames), dtype=int)
    path[0] = int(initial_state)
    for frame in range(1, int(frames)):
        path[frame] = rng.choice(
            matrix.shape[0], p=matrix[path[frame - 1]]
        )
    return path


def deterministic_cycle_path(initial_state, states, frames, step=1):
    """Return a deterministic cyclic state schedule."""
    if states <= 0 or frames <= 0 or int(states) != states or int(frames) != frames:
        raise ValueError("states and frames must be positive integers")
    if not 0 <= int(initial_state) < states:
        raise ValueError("initial_state is outside the state space")
    if int(step) != step:
        raise ValueError("step must be an integer")
    return (int(initial_state) + int(step) * np.arange(int(frames))) % int(states)


def gauss_markov_gains(frames, correlation, rng, initial_gain=1.0 + 0.0j):
    """Generate unit-variance complex Gauss--Markov frame gains."""
    if frames <= 0 or int(frames) != frames:
        raise ValueError("frames must be a positive integer")
    if not np.isfinite(correlation) or not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [0, 1]")
    gains = np.empty(int(frames), dtype=complex)
    gains[0] = complex(initial_gain)
    innovation_scale = np.sqrt(1.0 - correlation ** 2)
    for frame in range(1, int(frames)):
        innovation = (
            rng.standard_normal() + 1j * rng.standard_normal()
        ) / np.sqrt(2.0)
        gains[frame] = correlation * gains[frame - 1] + innovation_scale * innovation
    return gains


def multiframe_joint_signature(pilot_codebook, state_path, steering, waveform,
                               doppler_phase_step=0.0, frame_gains=None):
    """Stack known pilot-angle-DD signatures over a short normal-frame window."""
    codebook = np.asarray(pilot_codebook, dtype=complex)
    path = np.asarray(state_path, dtype=int)
    steering = np.asarray(steering, dtype=complex)
    waveform = np.asarray(waveform, dtype=complex)
    if codebook.ndim != 2 or 0 in codebook.shape:
        raise ValueError("pilot_codebook must have shape [state, pilot feature]")
    if path.ndim != 1 or path.size == 0 or np.any(path < 0) or np.any(path >= codebook.shape[0]):
        raise ValueError("state_path must contain valid state indices")
    if steering.ndim != 1 or waveform.ndim != 1 or steering.size == 0 or waveform.size == 0:
        raise ValueError("steering and waveform must be nonempty vectors")
    if not np.isfinite(doppler_phase_step):
        raise ValueError("doppler_phase_step must be finite")
    gains = (
        np.ones(path.size, dtype=complex)
        if frame_gains is None else np.asarray(frame_gains, dtype=complex)
    )
    if gains.shape != path.shape or not np.all(np.isfinite(gains)):
        raise ValueError("frame_gains must be finite and match state_path")
    physical = np.kron(steering, waveform)
    blocks = [
        gains[frame]
        * np.exp(1j * doppler_phase_step * frame)
        * np.kron(codebook[state], physical)
        for frame, state in enumerate(path)
    ]
    signature = np.concatenate(blocks)
    norm = np.linalg.norm(signature)
    if norm <= 0.0:
        raise ValueError("multi-frame signature must have nonzero energy")
    return signature / norm


def multiframe_joint_gram(codebooks, state_paths, steerings, waveforms,
                          doppler_phase_steps=None, frame_gains=None):
    """Return the normalized Gram matrix of known-state multi-frame sources."""
    source_count = len(codebooks)
    if not all(len(values) == source_count for values in (
        state_paths, steerings, waveforms
    )):
        raise ValueError("all source lists must have equal length")
    phases = (
        [0.0] * source_count
        if doppler_phase_steps is None else doppler_phase_steps
    )
    gains = (
        [None] * source_count if frame_gains is None else frame_gains
    )
    if len(phases) != source_count or len(gains) != source_count:
        raise ValueError("phase and gain lists must match source count")
    columns = np.column_stack([
        multiframe_joint_signature(
            codebooks[source], state_paths[source], steerings[source],
            waveforms[source], phases[source], gains[source],
        )
        for source in range(source_count)
    ])
    return normalized_gram(columns)


def stationary_switch_probability(transition, stationary=None):
    """Return the one-step state-switch probability under a state distribution."""
    matrix = validate_transition_matrix(transition)
    if stationary is None:
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        distribution = np.real(eigenvectors[:, index])
        distribution = np.abs(distribution) / np.sum(np.abs(distribution))
    else:
        distribution = np.asarray(stationary, dtype=float)
        if distribution.shape != (matrix.shape[0],) or np.any(distribution < 0.0):
            raise ValueError("stationary distribution has invalid shape or mass")
        distribution = distribution / np.sum(distribution)
    return float(1.0 - np.sum(distribution * np.diag(matrix)))
