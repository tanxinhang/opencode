import numpy as np

from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.oracle import exhaustive_oracle
from uav_otfs_isac.selection import greedy_select


def make_model(target, owner, delta):
    n = len(delta)
    return TargetEvidenceModel(
        target_id=target,
        owner=owner,
        mu0=np.zeros(n),
        mu1=np.asarray(delta, dtype=float),
        sigma0=np.eye(n),
        sigma1=np.eye(n),
        success_prob=np.ones(n),
        report_bits=np.asarray([0 if i == owner else 1 for i in range(n)]),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )


def test_oracle_dominates_greedy_on_small_instance():
    models = [make_model(0, 0, [1.0, 0.8, 0.5]), make_model(1, 1, [0.6, 1.0, 0.7])]
    greedy = greedy_select(models, 2, [1.2, 1.2], [1, 1], [1, 1])
    oracle = exhaustive_oracle(models, 2, [1.2, 1.2], [1, 1], [1, 1])
    assert oracle.normalized_qos_gap <= greedy.normalized_qos_gap + 1e-12
    if np.isclose(oracle.normalized_qos_gap, greedy.normalized_qos_gap):
        assert oracle.expected_deflection.sum() >= greedy.expected_deflection.sum() - 1e-12

