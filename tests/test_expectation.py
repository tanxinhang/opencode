import numpy as np

from uav_otfs_isac.expectation import expected_deflection_exact
from uav_otfs_isac.models import TargetEvidenceModel


def model(success=(1.0, 0.5)):
    return TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.zeros(2),
        mu1=np.ones(2),
        sigma0=np.eye(2),
        sigma1=np.eye(2),
        success_prob=np.array(success),
        report_bits=np.array([0, 3]),
        bit_flip_prob=np.zeros(2),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )


def test_exact_expectation_respects_random_effective_set():
    m = model()
    # D({owner})=1 and D({owner, report})=2, hence E[D]=1.5.
    assert np.isclose(expected_deflection_exact(m, {0, 1}), 1.5)


def test_erasure_is_not_interpreted_as_zero_evidence():
    m = model(success=(1.0, 0.0))
    assert np.isclose(expected_deflection_exact(m, {0, 1}), 1.0)

