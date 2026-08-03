import numpy as np

from uav_otfs_isac.dd_patterns import (
    assignment_cost,
    exhaustive_pattern_assignment,
    greedy_pattern_assignment,
    geometry_collision_tensor,
    full_grid_ambiguity_metrics,
    pattern_collision_matrix,
    select_balanced_ambiguity_codebook,
)
from uav_otfs_isac.otfs_physical import (
    cyclic_impulse_pattern,
    delay_doppler_path,
    otfs_modulate,
    qpsk_phase_pattern,
)


def test_collision_matrix_detects_fractional_loss_of_orthogonality():
    patterns = [
        cyclic_impulse_pattern(8, 16, 0, 0),
        cyclic_impulse_pattern(8, 16, 2, 4),
        cyclic_impulse_pattern(8, 16, 4, 8),
    ]
    shifts = np.linspace(-0.45, 0.45, 5)
    matrix = pattern_collision_matrix(patterns, shifts, shifts)
    assert np.allclose(matrix, matrix.T, atol=1e-12)
    assert np.all(np.diag(matrix) > 0.05)
    off_diagonal = matrix[~np.eye(3, dtype=bool)]
    assert np.all(off_diagonal > 0.0)
    assert np.max(off_diagonal) < np.min(np.diag(matrix))


def test_oracle_and_greedy_reduce_collision_over_same_pattern():
    collisions = np.array([
        [1.0, 0.10, 0.25],
        [0.10, 1.0, 0.15],
        [0.25, 0.15, 1.0],
    ])
    weights = np.array([
        [0.0, 3.0, 1.0, 2.0],
        [3.0, 0.0, 2.0, 1.0],
        [1.0, 2.0, 0.0, 3.0],
        [2.0, 1.0, 3.0, 0.0],
    ])
    same = (0, 0, 0, 0)
    oracle = exhaustive_pattern_assignment(weights, collisions)
    greedy = greedy_pattern_assignment(weights, collisions)
    assert assignment_cost(oracle, weights, collisions) <= (
        assignment_cost(greedy, weights, collisions) + 1e-12
    )
    assert assignment_cost(greedy, weights, collisions) < (
        assignment_cost(same, weights, collisions) / 2.0
    )


def test_collision_weights_require_positive_mass_and_are_not_mutated():
    patterns = [
        cyclic_impulse_pattern(4, 8, 0, 0),
        cyclic_impulse_pattern(4, 8, 1, 2),
    ]
    weights = np.ones((3, 3))
    original = weights.copy()
    pattern_collision_matrix(patterns, [-0.2, 0.0, 0.2],
                             [-0.2, 0.0, 0.2], weights)
    assert np.array_equal(weights, original)
    with np.testing.assert_raises_regex(ValueError, "positive mass"):
        pattern_collision_matrix(patterns, [0.0], [0.0], np.zeros((1, 1)))


def test_geometry_collision_tensor_is_pair_and_pattern_specific():
    patterns = [
        cyclic_impulse_pattern(4, 8, 0, 0),
        cyclic_impulse_pattern(4, 8, 1, 2),
    ]
    tensor = geometry_collision_tensor(
        patterns, [0.2, 1.4, 2.1], [0.1, 0.8, 1.7], [1.0, 0.8, 0.6]
    )
    assert tensor.shape == (3, 3, 2, 2)
    assert np.allclose(tensor, tensor.transpose(1, 0, 3, 2))
    assert not np.allclose(tensor[0, 1], tensor[1, 2])


def test_geometry_collision_matches_direct_waveform_templates():
    patterns = [
        cyclic_impulse_pattern(8, 16, 0, 0),
        cyclic_impulse_pattern(8, 16, 2, 5),
    ]
    delays = [1.2, 4.35]
    dopplers = [0.35, 2.18]
    gains = [1.0, 0.85]
    tensor = geometry_collision_tensor(patterns, delays, dopplers, gains)
    first = delay_doppler_path(
        otfs_modulate(patterns[0]), delays[0], dopplers[0], 8
    )
    second = delay_doppler_path(
        otfs_modulate(patterns[1]), delays[1], dopplers[1], 8
    )
    cross_power = abs(np.vdot(first, second)) ** 2
    expected = 0.5 * (
        abs(gains[1]) ** 2 / abs(gains[0]) ** 2
        * cross_power / np.vdot(first, first).real ** 2
        + abs(gains[0]) ** 2 / abs(gains[1]) ** 2
        * cross_power / np.vdot(second, second).real ** 2
    )
    assert np.isclose(tensor[0, 1, 0, 1], expected, atol=1e-15)


def test_cyclic_impulses_are_not_identity_codes_over_full_search_grid():
    first = cyclic_impulse_pattern(8, 16, 0, 0)
    shifted = cyclic_impulse_pattern(8, 16, 2, 5)
    ambiguity = np.abs(pattern_collision_matrix(
        [first, shifted], np.arange(16), np.arange(8)
    ))
    # Averaging can hide it, but some unknown path shift aligns the two
    # impulses exactly; they are shifted copies rather than distinct codes.
    from uav_otfs_isac.otfs_physical import dd_cross_ambiguity
    full_map = np.abs(dd_cross_ambiguity(
        first, shifted, np.arange(16), np.arange(8)
    )) ** 2
    assert np.isclose(np.max(full_map), 1.0)
    assert ambiguity.shape == (2, 2)


def test_phase_codes_limit_cross_ambiguity_over_full_search_grid():
    first = qpsk_phase_pattern(8, 16, 11)
    second = qpsk_phase_pattern(8, 16, 29)
    from uav_otfs_isac.otfs_physical import dd_cross_ambiguity
    full_map = np.abs(dd_cross_ambiguity(
        first, second, np.arange(16), np.arange(8)
    )) ** 2
    assert np.max(full_map) < 0.1


def test_greedy_initialization_uses_only_already_assigned_uavs():
    collisions = np.array([[1.0, 0.1], [0.1, 1.0]])
    weights = np.ones((3, 3)) - np.eye(3)
    assignment = greedy_pattern_assignment(weights, collisions)
    assert assignment[0] == 0
    assert assignment[1] == 1


def test_assignment_cost_rejects_invalid_cost_models():
    collisions = np.eye(2)
    asymmetric = np.array([[0.0, 1.0], [0.0, 0.0]])
    with np.testing.assert_raises_regex(ValueError, "symmetric"):
        assignment_cost((0, 1), asymmetric, collisions)
    with np.testing.assert_raises_regex(ValueError, "nonnegative"):
        assignment_cost((0, 1), np.ones((2, 2)), -collisions)


def test_full_grid_metrics_and_codebook_selection_are_deterministic():
    candidates = [qpsk_phase_pattern(4, 8, seed) for seed in range(6)]
    indices, metrics = select_balanced_ambiguity_codebook(candidates, 3)
    repeated, repeated_metrics = select_balanced_ambiguity_codebook(candidates, 3)
    assert indices == repeated
    assert metrics == repeated_metrics
    direct = full_grid_ambiguity_metrics([candidates[i] for i in indices])
    assert metrics == direct
    assert len(metrics["auto_peak_sidelobes"]) == 3
    assert len(metrics["cross_peaks"]) == 3
