from __future__ import annotations

import numpy as np

from .models import TargetEvidenceModel


def symmetric_diversity_model(
    report_delta: np.ndarray | None = None,
    success_probability: float = 0.6,
) -> TargetEvidenceModel:
    """One owner and four controlled reports for failure-diversity audits."""
    delta_reports = (
        np.ones(4, dtype=float)
        if report_delta is None
        else np.asarray(report_delta, dtype=float)
    )
    if delta_reports.shape != (4,):
        raise ValueError("report_delta must contain four values")
    delta = np.concatenate(([0.2], delta_reports))
    n = delta.size
    model = TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.zeros(n),
        mu1=delta,
        sigma0=np.eye(n),
        sigma1=np.eye(n),
        success_prob=np.array([1.0] + [success_probability] * 4),
        report_bits=np.array([0, 1, 1, 1, 1]),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )
    model.validate()
    return model
