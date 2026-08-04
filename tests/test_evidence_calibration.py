import numpy as np

from uav_otfs_isac.evidence_calibration import (
    collect_evidence,
    delta_deflection_vs_delta_pd,
    estimate_moments,
    evidence_matrices,
    moment_health,
    shrink_covariance,
)
from uav_otfs_isac.front_end import (
    FrontEndConfig,
    identity_patterns,
    precompute_templates,
)


def test_evidence_collection_returns_both_hypotheses():
    config = FrontEndConfig(noise_variance=0.02)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    rng = np.random.default_rng(3)
    records = collect_evidence(
        config, patterns, templates, rng, trials=2,
        integration_frames=1, amplitude=2.0,
    )
    hypotheses = {record.hypothesis for record in records}
    assert hypotheses == {0, 1}
    assert len(records) == 2 * 4 + 2 * 4 * 1


def test_shrink_covariance_is_positive_definite():
    rng = np.random.default_rng(5)
    samples = rng.normal(size=(50, 4)) + np.asarray([1.0, 2.0, 3.0, 4.0])
    covariance = shrink_covariance(samples, alpha=0.5)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0


def test_delta_vs_pd_orders_synthetic_moments():
    mu0 = np.zeros(4)
    mu1 = np.asarray([0.5, 1.0, 2.0, 3.0])
    cov0 = np.eye(4) * 0.25
    cov1 = np.eye(4) * 0.25
    moments = {
        "means": {"h0": mu0, "h1": mu1},
        "covariances": {"h0": cov0, "h1": cov1},
    }
    result = delta_deflection_vs_delta_pd(moments)
    # Smoke only: the ordering direction must be positive.  The formal G1-A
    # gate requires a bootstrap Spearman of at least 0.6.
    assert result["spearman"] >= 0.2


def test_moment_health_accepts_estimated_covariances():
    config = FrontEndConfig(noise_variance=0.02)
    patterns = identity_patterns(config, 4)
    templates = precompute_templates(config, patterns)
    rng = np.random.default_rng(7)
    records = collect_evidence(
        config, patterns, templates, rng, trials=4,
        integration_frames=1, amplitude=2.0,
    )
    moments = estimate_moments(evidence_matrices(records, len(patterns)))
    health = moment_health(moments)
    assert health["h0_positive_definite"]
    assert health["h1_positive_definite"]
