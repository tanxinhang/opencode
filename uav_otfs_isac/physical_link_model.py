"""Physical sensing and communication channels with explicit decoupling.

The sensing channel is UAV-to-target and determines the H0/H1 evidence
moments.  The communication channel is UAV-to-owner and determines report
bit-flip and erasure statistics.  The two channels are not assumed to share
path-loss exponents, reference SNRs, shadowing, or outage thresholds.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .config import ExperimentConfig
from .models import TargetEvidenceModel
from .scenario import build_models, uav_geometry


def bpsk_bit_flip_probability(snr_db: float) -> float:
    """Uncoded BPSK bit-error probability from link SNR in dB."""
    snr = 10.0 ** (float(snr_db) / 10.0)
    return float(norm.sf(np.sqrt(2.0 * snr)))


def lognormal_outage_success(
    snr_db: float,
    threshold_db: float,
    shadowing_db: float,
) -> float:
    """Survival probability of a log-normal link above the SNR threshold."""
    snr_db = float(snr_db)
    threshold_db = float(threshold_db)
    shadowing_db = float(shadowing_db)
    if shadowing_db <= 0.0:
        return 1.0 if snr_db >= threshold_db else 0.0
    return float(norm.sf((threshold_db - snr_db) / shadowing_db))


def report_link_snr_db(
    transmitter_positions: np.ndarray,
    owner: int,
    *,
    reference_distance: float = 100.0,
    reference_snr_db: float = 25.0,
    path_loss_exponent: float = 2.0,
) -> np.ndarray:
    """Free-space path-loss SNR of every UAV relative to the owner."""
    positions = np.asarray(transmitter_positions, dtype=float)
    distance = np.linalg.norm(positions - positions[owner], axis=1)
    return (
        float(reference_snr_db)
        - 10.0
        * float(path_loss_exponent)
        * np.log10(np.maximum(distance / float(reference_distance), 1e-9))
    )


def physical_report_link_parameters(
    cfg: ExperimentConfig,
    *,
    reference_distance: float = 100.0,
    reference_snr_db: float = 25.0,
    threshold_db: float = 5.0,
    shadowing_db: float = 3.0,
    path_loss_exponent: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bit-flip, success) matrices derived from report geometry."""
    positions = uav_geometry(cfg.num_uavs)
    flip = np.zeros((cfg.num_targets, cfg.num_uavs), dtype=float)
    success = np.ones((cfg.num_targets, cfg.num_uavs), dtype=float)
    for q, owner in enumerate(cfg.owners):
        snr_db = report_link_snr_db(
            positions,
            owner,
            reference_distance=reference_distance,
            reference_snr_db=reference_snr_db,
            path_loss_exponent=path_loss_exponent,
        )
        flip[q] = [bpsk_bit_flip_probability(value) for value in snr_db]
        success[q] = [
            lognormal_outage_success(value, threshold_db, shadowing_db)
            for value in snr_db
        ]
        flip[q, owner] = 0.0
        success[q, owner] = 1.0
    return flip, success


def build_physical_link_models(
    cfg: ExperimentConfig,
    seed: int,
    *,
    reference_distance: float = 100.0,
    reference_snr_db: float = 25.0,
    threshold_db: float = 5.0,
    shadowing_db: float = 3.0,
    path_loss_exponent: float = 2.0,
) -> list[TargetEvidenceModel]:
    """Build moment-matched models with geometry-derived report links."""
    flip, success = physical_report_link_parameters(
        cfg,
        reference_distance=reference_distance,
        reference_snr_db=reference_snr_db,
        threshold_db=threshold_db,
        shadowing_db=shadowing_db,
        path_loss_exponent=path_loss_exponent,
    )
    return build_models(
        cfg,
        np.random.default_rng(seed),
        report_bit_flip_probabilities=flip,
        report_success_probabilities=success,
    )
