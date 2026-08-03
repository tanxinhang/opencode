import numpy as np

from uav_otfs_isac.identifiability import (
    factorized_joint_gram,
    gram_identifiability_metrics,
    joint_signature,
    minimum_probe_length,
    normalized_gram,
    worst_case_gram_metrics,
)


def test_factorized_gram_matches_explicit_joint_signatures():
    rng = np.random.default_rng(20260821)
    probes = [rng.standard_normal(3) + 1j * rng.standard_normal(3) for _ in range(2)]
    angles = [rng.standard_normal(4) + 1j * rng.standard_normal(4) for _ in range(2)]
    waveforms = [rng.standard_normal(5) + 1j * rng.standard_normal(5) for _ in range(2)]
    explicit = normalized_gram(np.column_stack([
        joint_signature(probes[i], angles[i], waveforms[i]) for i in range(2)
    ]))
    factorized = factorized_joint_gram(probes, angles, waveforms)
    assert np.allclose(factorized, explicit, atol=1e-12)


def test_gram_metrics_detect_rank_loss_and_orthogonality():
    rank_one = np.ones((2, 2), dtype=complex)
    collapsed = gram_identifiability_metrics(rank_one)
    assert collapsed["lambda_min"] == 0.0
    assert np.isinf(collapsed["condition_number"])
    assert collapsed["max_effective_coherence"] == 1.0
    separated = gram_identifiability_metrics(np.eye(2))
    assert separated["lambda_min"] == 1.0
    assert separated["condition_number"] == 1.0
    assert separated["max_effective_coherence"] == 0.0


def test_worst_case_metrics_and_minimum_probe_length():
    poor = np.array([[1.0, 0.9], [0.9, 1.0]])
    medium = np.array([[1.0, 0.5], [0.5, 1.0]])
    good = np.eye(2)
    worst = worst_case_gram_metrics([medium, good])
    assert np.isclose(worst["worst_lambda_min"], 0.5)
    length, decisions = minimum_probe_length(
        {1: [poor], 2: [medium, good], 3: [good]}, 0.4
    )
    assert length == 2
    assert set(decisions) == {1, 2}
    unresolved, _ = minimum_probe_length({1: [poor]}, 0.4)
    assert unresolved is None
