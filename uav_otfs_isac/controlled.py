from __future__ import annotations

import numpy as np

from .models import TargetEvidenceModel


def symmetric_diversity_model(
    report_delta: np.ndarray | None = None,
    success_probability: float = 0.6,
    report_bits: np.ndarray | None = None,
) -> TargetEvidenceModel:
    """One owner and four controlled reports for failure-diversity audits.

    ``report_bits`` supplies the per-report bit cost (owner entry ignored and
    forced to zero); the default is one bit per report.
    """
    delta_reports = (
        np.ones(4, dtype=float)
        if report_delta is None
        else np.asarray(report_delta, dtype=float)
    )
    if delta_reports.shape != (4,):
        raise ValueError("report_delta must contain four values")
    delta = np.concatenate(([0.2], delta_reports))
    n = delta.size
    if report_bits is None:
        report_bits = np.ones(4, dtype=int)
    else:
        report_bits = np.asarray(report_bits, dtype=int)
        if report_bits.shape != (4,):
            raise ValueError("report_bits must contain four values")
        if np.any(report_bits <= 0):
            raise ValueError("report costs must be positive")
    model = TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.zeros(n),
        mu1=delta,
        sigma0=np.eye(n),
        sigma1=np.eye(n),
        success_prob=np.array([1.0] + [success_probability] * 4),
        report_bits=np.concatenate(([0], report_bits)),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )
    model.validate()
    return model
