import numpy as np

from scripts.run_joint_gram_gate import (
    best_trigger_threshold,
    conditional_resolution_probability,
    progressive_probe_pair,
    wilson_interval,
)


def test_progressive_probe_pair_reaches_orthogonality_at_design_length():
    correlations = []
    for length in (1, 2, 3, 4):
        first, second = progressive_probe_pair(length)
        correlations.append(abs(np.vdot(first, second)))
    assert correlations[0] == 1.0
    assert correlations[-1] < 1e-12
    assert all(first >= second for first, second in zip(
        correlations, correlations[1:]
    ))


def test_trigger_threshold_separates_ordered_hard_and_easy_cases():
    result = best_trigger_threshold(
        [0.0, 0.1, 0.8, 1.0], [True, True, False, False]
    )
    assert result["hard_trigger_rate"] == 1.0
    assert result["easy_false_trigger_rate"] == 0.0


def test_better_conditioned_gram_improves_joint_ls_resolution():
    poor = np.array([[1.0, 0.95], [0.95, 1.0]], dtype=complex)
    good = np.eye(2, dtype=complex)
    poor_probability = conditional_resolution_probability(
        poor, noise_variance=0.05, trials=100_000, seed=20260823
    )
    good_probability = conditional_resolution_probability(
        good, noise_variance=0.05, trials=100_000, seed=20260823
    )
    assert good_probability > poor_probability + 0.5


def test_wilson_interval_is_not_degenerate_for_perfect_small_sample():
    lower, upper = wilson_interval(9, 9)
    assert 0.7 < lower < 1.0
    assert upper <= 1.0
