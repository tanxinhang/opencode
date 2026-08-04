import numpy as np

from scripts.run_multistatic_glrt_information_audit import empirical_auc


def test_empirical_auc_handles_order_and_ties():
    assert empirical_auc([0.0, 1.0], [2.0, 3.0]) == 1.0
    assert empirical_auc([2.0, 3.0], [0.0, 1.0]) == 0.0
    assert empirical_auc([1.0, 1.0], [1.0]) == 0.5


def test_empirical_auc_matches_pairwise_definition():
    null = np.asarray([0.0, 2.0, 4.0])
    alternative = np.asarray([1.0, 2.0, 5.0])
    pairwise = np.mean(
        (alternative[:, None] > null[None, :])
        + 0.5 * (alternative[:, None] == null[None, :])
    )
    assert np.isclose(empirical_auc(null, alternative), pairwise)
