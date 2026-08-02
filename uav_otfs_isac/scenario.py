from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .config import ExperimentConfig
from .linalg import regularize_covariance
from .models import TargetEvidenceModel
from .reporting import post_bsc_moments, quantizer_from_gaussian_range


def uav_geometry(n: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = 180.0 + 35.0 * np.sin(3.0 * angles)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles), 100 + 15 * np.cos(angles)))


def target_geometry(q: int) -> np.ndarray:
    return np.array([45.0 * np.cos(1.7 * q), 55.0 * np.sin(1.3 * q), 0.0])


def build_models(cfg: ExperimentConfig, rng: np.random.Generator) -> list[TargetEvidenceModel]:
    positions = uav_geometry(cfg.num_uavs)
    models: list[TargetEvidenceModel] = []
    for q in range(cfg.num_targets):
        target = target_geometry(q)
        distance = np.linalg.norm(positions - target, axis=1)
        snr_lo, snr_hi = cfg.otfs.snr_db_range
        base_snr_db = snr_hi - (snr_hi - snr_lo) * (distance - distance.min()) / max(np.ptp(distance), 1e-9)
        fractional = rng.uniform(*cfg.otfs.fractional_doppler_range, cfg.num_uavs)
        leakage = np.sinc(fractional) ** 2
        effective_snr = 10 ** (base_snr_db / 10.0) * np.clip(leakage, 0.12, 1.0)
        effective_snr /= 1.0 + cfg.otfs.residual_interference

        # Gamma/chi-square moment matching after L non-coherent accumulations.
        l_acc = cfg.otfs.accumulation
        mu0_local = np.full(cfg.num_uavs, float(l_acc))
        var0_local = np.full(cfg.num_uavs, float(l_acc))
        noncentrality = l_acc * effective_snr
        mu1_local = mu0_local + noncentrality
        var1_local = var0_local + 2.0 * noncentrality

        view = positions - target
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        geometry_similarity = np.clip(view @ view.T, 0.0, 1.0)
        dd_similarity = np.exp(-np.abs(fractional[:, None] - fractional[None, :]) / 0.18)
        common_strength = cfg.otfs.common_factor_strength
        correlation = common_strength * geometry_similarity * dd_similarity
        np.fill_diagonal(correlation, 1.0)
        sigma0_local = correlation * np.sqrt(var0_local[:, None] * var0_local[None, :])
        sigma1_local = correlation * np.sqrt(var1_local[:, None] * var1_local[None, :])
        sigma0_local = regularize_covariance(sigma0_local, cfg.covariance_shrinkage, cfg.covariance_epsilon)
        sigma1_local = regularize_covariance(sigma1_local, cfg.covariance_shrinkage, cfg.covariance_epsilon)

        edges, values = quantizer_from_gaussian_range(
            mu0_local, sigma0_local, mu1_local, sigma1_local, cfg.quantizer_bits
        )
        owner = cfg.owners[q]
        # Air-to-air reporting quality is tied to reporter-to-owner distance.
        # A UAV close to the target need not have the best reporting link.
        report_distance = np.linalg.norm(positions - positions[owner], axis=1)
        normalized_report_distance = report_distance / max(report_distance.max(), 1e-9)
        success_lo, success_hi = cfg.reporting.success_probability_range
        flip_lo, flip_hi = cfg.reporting.bit_flip_probability_range
        link_jitter = rng.normal(0.0, 0.025, cfg.num_uavs)
        success = np.clip(
            success_hi - (success_hi - success_lo) * normalized_report_distance + link_jitter,
            success_lo,
            success_hi,
        )
        p_flip = np.clip(
            flip_lo + (flip_hi - flip_lo) * normalized_report_distance - 0.15 * link_jitter,
            flip_lo,
            flip_hi,
        )
        success[owner] = 1.0; p_flip[owner] = 0.0

        post_mu0 = np.empty(cfg.num_uavs); post_mu1 = np.empty(cfg.num_uavs)
        post_var0 = np.empty(cfg.num_uavs); post_var1 = np.empty(cfg.num_uavs)
        for i in range(cfg.num_uavs):
            post_mu0[i], post_var0[i] = post_bsc_moments(
                mu0_local[i], sigma0_local[i, i], edges, values, cfg.quantizer_bits, p_flip[i]
            )
            post_mu1[i], post_var1[i] = post_bsc_moments(
                mu1_local[i], sigma1_local[i, i], edges, values, cfg.quantizer_bits, p_flip[i]
            )
        # Propagate local dependence through moment-matched attenuation factors.
        g0 = np.sqrt(np.maximum(post_var0 - cfg.reporting.calibration_std**2, 1e-12) / np.diag(sigma0_local))
        g1 = np.sqrt(np.maximum(post_var1 - cfg.reporting.calibration_std**2, 1e-12) / np.diag(sigma1_local))
        sigma0 = np.outer(g0, g0) * sigma0_local
        sigma1 = np.outer(g1, g1) * sigma1_local
        np.fill_diagonal(sigma0, post_var0 + cfg.reporting.calibration_std**2)
        np.fill_diagonal(sigma1, post_var1 + cfg.reporting.calibration_std**2)
        sigma0[owner] = sigma0_local[owner]; sigma0[:, owner] = sigma0_local[:, owner]
        sigma1[owner] = sigma1_local[owner]; sigma1[:, owner] = sigma1_local[:, owner]
        post_mu0[owner] = mu0_local[owner]; post_mu1[owner] = mu1_local[owner]
        sigma0 = regularize_covariance(sigma0, cfg.covariance_shrinkage, cfg.covariance_epsilon)
        sigma1 = regularize_covariance(sigma1, cfg.covariance_shrinkage, cfg.covariance_epsilon)
        bits = np.full(cfg.num_uavs, cfg.quantizer_bits + 2, dtype=int)
        bits[owner] = 0
        model = TargetEvidenceModel(
            target_id=q,
            owner=owner,
            mu0=post_mu0,
            mu1=post_mu1,
            sigma0=sigma0,
            sigma1=sigma1,
            success_prob=success,
            report_bits=bits,
            bit_flip_prob=p_flip,
            quantizer_edges=edges,
            quantizer_values=values,
        )
        model.validate(); models.append(model)
    return models
