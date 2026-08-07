from __future__ import annotations

from collections.abc import Sequence

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


def build_models(
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    snr_gain: np.ndarray | None = None,
    transmitter_positions: np.ndarray | None = None,
    target_positions: Sequence | None = None,
    quantizer_bits_per_uav: Sequence[int] | None = None,
    interference_to_noise: np.ndarray | None = None,
) -> list[TargetEvidenceModel]:
    """Build the moment-matched system models.

    ``snr_gain`` optionally supplies a per-target, per-UAV evidence SNR gain
    matrix of shape ``(num_targets, num_uavs)``, e.g. the gain induced by a
    RIS-assisted cascaded channel before quantization and reporting.

    ``transmitter_positions`` and ``target_positions`` override the static
    geometry for time-varying scenarios; the default is
    :func:`uav_geometry` and :func:`target_geometry`.

    ``quantizer_bits_per_uav`` overrides the uniform quantizer bit count and
    gives each report its own rate, enabling variable-rate soft reporting.

    ``interference_to_noise`` applies an interference-to-noise ratio per UAV,
    reducing the effective SINR as ``SNR / (1 + INR)``.
    """
    positions = (
        uav_geometry(cfg.num_uavs)
        if transmitter_positions is None
        else np.asarray(transmitter_positions, dtype=float)
    )
    if positions.shape != (cfg.num_uavs, 3):
        raise ValueError("transmitter_positions must have shape (num_uavs, 3)")
    if target_positions is None:
        target_positions = [target_geometry(q) for q in range(cfg.num_targets)]
    else:
        target_positions = [
            np.asarray(position, dtype=float) for position in target_positions
        ]
        if len(target_positions) != cfg.num_targets:
            raise ValueError("one target position is required per target")
    if any(position.shape != (3,) for position in target_positions):
        raise ValueError("target positions must be 3-D")
    if quantizer_bits_per_uav is None:
        per_uav_bits = np.full(cfg.num_uavs, cfg.quantizer_bits, dtype=int)
    else:
        per_uav_bits = np.asarray(quantizer_bits_per_uav, dtype=int)
        if per_uav_bits.shape != (cfg.num_uavs,):
            raise ValueError("quantizer_bits_per_uav must have one entry per UAV")
    if np.any(per_uav_bits <= 0):
        raise ValueError("quantizer bits must be positive")
    if interference_to_noise is not None:
        interference_to_noise = np.asarray(interference_to_noise, dtype=float)
        if interference_to_noise.shape != (cfg.num_uavs,):
            raise ValueError("interference_to_noise must have one entry per UAV")
        if np.any(interference_to_noise < 0.0):
            raise ValueError("interference_to_noise entries must be nonnegative")
    if snr_gain is not None:
        gain = np.asarray(snr_gain, dtype=float)
        if gain.shape != (cfg.num_targets, cfg.num_uavs):
            raise ValueError("snr_gain must have shape (num_targets, num_uavs)")
        if np.any(gain < 0.0):
            raise ValueError("snr_gain entries must be nonnegative")
    models: list[TargetEvidenceModel] = []
    for q in range(cfg.num_targets):
        target = target_positions[q]
        distance = np.linalg.norm(positions - target, axis=1)
        snr_lo, snr_hi = cfg.otfs.snr_db_range
        base_snr_db = snr_hi - (snr_hi - snr_lo) * (distance - distance.min()) / max(np.ptp(distance), 1e-9)
        fractional = rng.uniform(*cfg.otfs.fractional_doppler_range, cfg.num_uavs)
        leakage = np.sinc(fractional) ** 2
        effective_snr = 10 ** (base_snr_db / 10.0) * np.clip(leakage, 0.12, 1.0)
        interference = (
            cfg.otfs.residual_interference
            if interference_to_noise is None
            else interference_to_noise
        )
        effective_snr /= 1.0 + interference
        if snr_gain is not None:
            effective_snr *= gain[q]

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
            mu0_local, sigma0_local, mu1_local, sigma1_local,
            int(per_uav_bits[0]),
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
            bits_i = int(per_uav_bits[i])
            if quantizer_bits_per_uav is None:
                local_edges, local_values = edges, values
            else:
                local_edges, local_values = quantizer_from_gaussian_range(
                    mu0_local[i:i + 1], sigma0_local[i:i + 1, i:i + 1],
                    mu1_local[i:i + 1], sigma1_local[i:i + 1, i:i + 1],
                    bits_i,
                )
            post_mu0[i], post_var0[i] = post_bsc_moments(
                mu0_local[i], sigma0_local[i, i], local_edges, local_values,
                bits_i, p_flip[i],
            )
            post_mu1[i], post_var1[i] = post_bsc_moments(
                mu1_local[i], sigma1_local[i, i], local_edges, local_values,
                bits_i, p_flip[i],
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
        bits = np.asarray(per_uav_bits + 2, dtype=int)
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
