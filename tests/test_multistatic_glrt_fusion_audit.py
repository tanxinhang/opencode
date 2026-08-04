import numpy as np

from scripts.run_multistatic_glrt_fusion_audit import empirical_percentile


def test_empirical_percentile_is_monotone_and_finite_sample_bounded():
    values = empirical_percentile([1.0, 2.0, 3.0], [0.0, 2.0, 4.0])
    assert np.all(np.diff(values) >= 0)
    assert np.allclose(values, (0.25, 0.75, 1.0))
