import numpy as np

from scripts.run_g1c_conditional_ranking_gate import (
    _greedy,
    _static_id_topk,
)


def test_conditional_greedy_degenerates_to_static_id_topk():
    mu0 = np.zeros(4)
    mu1 = np.asarray([1.0, 2.0, 3.0, 4.0])
    cov0 = np.eye(4)
    assert _greedy(mu0, mu1, cov0, 2) == _static_id_topk(mu0, mu1, cov0, 2)


def test_conditional_greedy_prefers_low_correlation_after_first_report():
    mu0 = np.zeros(3)
    mu1 = np.asarray([3.0, 2.9, 2.0])
    cov0 = np.asarray([
        [1.0, 0.9, 0.05],
        [0.9, 1.0, 0.05],
        [0.05, 0.05, 1.0],
    ])
    greedy = _greedy(mu0, mu1, cov0, 2)
    static = _static_id_topk(mu0, mu1, cov0, 2)
    assert 0 in greedy
    assert 2 in greedy
    assert 1 in static
