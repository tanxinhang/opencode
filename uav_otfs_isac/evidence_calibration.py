"""G1-A evidence-moment calibration from the toy G0-C front end.

The paper mainline requires per-target, per-UAV detection evidence
``z_iq`` with conditional moments ``(mu_h, Sigma_h)``.  This module samples
those moments from the waveform-level front end rather than filling a
covariance matrix by hand: H1 evidence is read at the true angle-delay-Doppler
cell of every geometry-generated path, and H0 evidence is the noise-floor max
of the same matched-filter cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from .front_end import (
    FrontEndConfig,
    integrated_detection_cubes,
    simulate_received,
)
from .fusion import (
    conditional_marginal_deflection,
    gaussian_detection_probability,
)
from .multistatic_targets import (
    KinematicNode,
    PhysicalTarget,
    generate_bistatic_paths,
)


PROPAGATION_SPEED = 299_792_458.0


@dataclass(frozen=True)
class EvidenceRecord:
    trial_id: int
    hypothesis: int
    target_id: int
    uav_id: int
    raw_mf_energy: float
    calibrated_score: float
    fractional_doppler: float
    dd_leakage_ratio: float
    local_sinr: float
    peak_sidelobe_ratio: float
    common_rcs_factor: float
    common_clutter_id: int
    fisher_variance: float


def transmitter_geometry(count: int = 4) -> tuple[KinematicNode, ...]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return tuple(KinematicNode(
        (100.0 * np.cos(angle), 100.0 * np.sin(angle)), (0.0, 0.0)
    ) for angle in angles)


def receiver_node() -> KinematicNode:
    return KinematicNode((0.0, 0.0), (0.0, 0.0))


def draw_target(rng: np.random.Generator) -> PhysicalTarget:
    angle = rng.uniform(-40.0, 20.0)
    range_m = 55.0 + rng.uniform(-8.0, 8.0)
    return PhysicalTarget(
        target_id=0,
        position=range_m * np.asarray((
            np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        )),
        velocity=rng.uniform(-1.5, 1.5, 2),
    )


def _burst_received(
    config: FrontEndConfig,
    patterns: list[np.ndarray],
    paths: tuple,
    amplitude: float,
    rng: np.random.Generator,
    integration_frames: int,
    common_gain: complex = 1.0,
) -> list[np.ndarray]:
    received = []
    for _ in range(integration_frames):
        gains = [
            amplitude * complex(common_gain)
            * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
            for _ in paths
        ]
        received.append(simulate_received(
            config, patterns, paths, gains, rng
        ))
    return received


def _max_outside_neighborhood(
    cube: np.ndarray,
    angle_index: int,
    doppler_index: int,
    delay_index: int,
    config: FrontEndConfig,
    guard: int = 2,
) -> float:
    shape = (config.doppler_bins, config.delay_bins)
    working = cube.reshape((config.angle_grid_degrees.size,) + shape).copy()
    for da in range(-guard, guard + 1):
        a = angle_index + da
        if not 0 <= a < working.shape[0]:
            continue
        for dk in range(-guard, guard + 1):
            for dl in range(-guard, guard + 1):
                working[
                    a,
                    (doppler_index + dk) % config.doppler_bins,
                    (delay_index + dl) % config.delay_bins,
                ] = -np.inf
    return float(np.max(working))


def collect_evidence(
    config: FrontEndConfig,
    patterns: list[np.ndarray],
    templates,
    rng: np.random.Generator,
    *,
    trials: int,
    integration_frames: int = 1,
    amplitude: float = 2.0,
) -> list[EvidenceRecord]:
    """Sample H0/H1 evidence moments from true-cell matched-filter energy."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    nodes = transmitter_geometry(len(patterns))
    receiver = receiver_node()
    records = []
    trial_id = 0
    for hypothesis in (0, 1):
        for _ in range(trials):
            if hypothesis == 1:
                target = draw_target(rng)
                paths = generate_bistatic_paths(
                    nodes, [target], receiver, 5.9e9
                )
                common_factor = float(np.exp(
                    0.25 * rng.standard_normal()
                ))
                received = _burst_received(
                    config, patterns, paths, amplitude, rng,
                    integration_frames,
                    common_gain=common_factor,
                )
                cubes = integrated_detection_cubes(
                    config, received, patterns, templates=templates
                )
                for path in paths:
                    angle_index = int(np.argmin(np.abs(
                        config.angle_grid_degrees
                        - np.rad2deg(path.receive_azimuth_rad)
                    )))
                    doppler_index = int(np.round(
                        config.wrapped_doppler_bin(path.doppler_hz)
                    )) % config.doppler_bins
                    delay_index = int(np.round(
                        config.delay_bin(path.delay_s)
                    )) % config.delay_bins
                    flat = doppler_index * config.delay_bins + delay_index
                    cube = cubes[path.transmitter_id]
                    energy = float(cube[angle_index, flat])
                    sidelobe = _max_outside_neighborhood(
                        cube, angle_index, doppler_index, delay_index, config
                    )
                    fractional = float(
                        config.wrapped_doppler_bin(path.doppler_hz)
                        - doppler_index
                    )
                    records.append(EvidenceRecord(
                        trial_id=trial_id,
                        hypothesis=1,
                        target_id=target.target_id,
                        uav_id=path.transmitter_id,
                        raw_mf_energy=energy,
                        calibrated_score=float(np.clip(
                            energy / 3.0, 1e-4, 1.0 - 1e-4
                        )),
                        fractional_doppler=fractional,
                        dd_leakage_ratio=float(
                            sidelobe / max(energy, 1e-12)
                        ),
                        local_sinr=float(
                            energy / max(config.noise_variance, 1e-12)
                        ),
                        peak_sidelobe_ratio=float(
                            energy / max(sidelobe, 1e-12)
                        ),
                        common_rcs_factor=common_factor,
                        common_clutter_id=0,
                        fisher_variance=config.doppler_resolution_hz ** 2,
                    ))
            else:
                received = _burst_received(
                    config, patterns, (), amplitude, rng,
                    integration_frames,
                )
                cubes = integrated_detection_cubes(
                    config, received, patterns, templates=templates
                )
                for uav_id, cube in enumerate(cubes):
                    energy = float(np.max(cube))
                    records.append(EvidenceRecord(
                        trial_id=trial_id,
                        hypothesis=0,
                        target_id=0,
                        uav_id=uav_id,
                        raw_mf_energy=energy,
                        calibrated_score=float(np.clip(
                            energy / 3.0, 1e-4, 1.0 - 1e-4
                        )),
                        fractional_doppler=0.0,
                        dd_leakage_ratio=1.0,
                        local_sinr=float(
                            energy / max(config.noise_variance, 1e-12)
                        ),
                        peak_sidelobe_ratio=1.0,
                        common_rcs_factor=1.0,
                        common_clutter_id=0,
                        fisher_variance=config.doppler_resolution_hz ** 2,
                    ))
            trial_id += 1
    return records


def shrink_covariance(samples: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Convex shrinkage toward the diagonal variance estimate."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError("samples must have shape [trials, features]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("shrinkage alpha must lie in [0, 1]")
    covariance = np.cov(samples, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(covariance))
    return (1.0 - alpha) * covariance + alpha * diagonal


def evidence_matrices(
    records: list[EvidenceRecord], num_uavs: int, target_id: int = 0
) -> dict[str, np.ndarray]:
    h0 = []
    h1 = []
    by_trial: dict[tuple[int, int], dict[int, float]] = {}
    for record in records:
        by_trial.setdefault(
            (record.trial_id, record.hypothesis), {}
        )[record.uav_id] = record.raw_mf_energy
    for (trial_id, hypothesis), values in by_trial.items():
        row = np.asarray([
            values.get(uav_id, 0.0) for uav_id in range(num_uavs)
        ])
        (h0 if hypothesis == 0 else h1).append(row)
    return {
        "h0": np.asarray(h0),
        "h1": np.asarray(h1),
    }


def estimate_moments(
    matrices: dict[str, np.ndarray], alpha: float = 0.5
) -> dict[str, np.ndarray]:
    means = {}
    covariances = {}
    for key in ("h0", "h1"):
        samples = matrices[key]
        if len(samples) == 0:
            raise ValueError(f"no {key} evidence samples")
        means[key] = np.mean(samples, axis=0)
        covariances[key] = shrink_covariance(samples, alpha)
    return {"means": means, "covariances": covariances}


def moment_health(moments: dict[str, np.ndarray]) -> dict:
    eig0 = np.linalg.eigvalsh(moments["covariances"]["h0"])
    eig1 = np.linalg.eigvalsh(moments["covariances"]["h1"])
    return {
        "h0_positive_definite": bool(np.min(eig0) > 0.0),
        "h1_positive_definite": bool(np.min(eig1) > 0.0),
        "h0_min_eigenvalue": float(np.min(eig0)),
        "h1_min_eigenvalue": float(np.min(eig1)),
    }


def delta_deflection_vs_delta_pd(
    moments: dict[str, np.ndarray],
    false_alarm_rate: float = 0.05,
    bootstrap_replicates: int = 1000,
    seed: int = 20260804,
    actual_gain_mode: str = "relative_deficit_reduction",
    actual_moments: dict[str, np.ndarray] | None = None,
) -> dict:
    mu0 = np.asarray(moments["means"]["h0"], dtype=float)
    mu1 = np.asarray(moments["means"]["h1"], dtype=float)
    cov0 = np.asarray(moments["covariances"]["h0"], dtype=float)
    cov1 = np.asarray(moments["covariances"]["h1"], dtype=float)
    if actual_moments is None:
        actual_moments = moments
    actual_mu0 = np.asarray(actual_moments["means"]["h0"], dtype=float)
    actual_mu1 = np.asarray(actual_moments["means"]["h1"], dtype=float)
    actual_cov0 = np.asarray(
        actual_moments["covariances"]["h0"], dtype=float
    )
    actual_cov1 = np.asarray(
        actual_moments["covariances"]["h1"], dtype=float
    )
    delta = np.asarray(mu1) - np.asarray(mu0)
    deflection_pairs = []
    uav_ids = range(len(mu0))
    for selected_size in range(len(mu0)):
        for selected in combinations(uav_ids, selected_size):
            for candidate in uav_ids:
                if candidate in selected:
                    continue
                if selected:
                    predicted = conditional_marginal_deflection(
                        delta, cov0, selected, candidate
                    )
                else:
                    predicted = (
                        float(delta[candidate]) ** 2
                        / max(float(cov0[candidate, candidate]), 1e-12)
                    )
                base_pd = (
                    false_alarm_rate if not selected
                    else gaussian_detection_probability(
                        actual_mu0, actual_mu1, actual_cov0, actual_cov1,
                        selected, false_alarm_rate,
                    )
                )
                new_pd = gaussian_detection_probability(
                    actual_mu0, actual_mu1, actual_cov0, actual_cov1,
                    tuple(selected) + (candidate,), false_alarm_rate,
                )
                deflection_pairs.append((predicted, base_pd, new_pd))
    if not deflection_pairs:
        return {"spearman": None, "pairs": 0}
    predicted = np.asarray([value[0] for value in deflection_pairs])
    base_pd = np.asarray([value[1] for value in deflection_pairs])
    new_pd = np.asarray([value[2] for value in deflection_pairs])
    rng = np.random.default_rng(seed)
    modes = {}
    clip = 1e-6
    indices = np.arange(len(predicted))
    for mode in ("pd_gain", "relative_deficit_reduction", "logit_gain"):
        if mode == "pd_gain":
            actual = np.maximum(new_pd - base_pd, 0.0)
        elif mode == "relative_deficit_reduction":
            actual = np.maximum(
                (np.maximum(1.0 - base_pd, 1e-6)
                 - np.maximum(1.0 - new_pd, 0.0))
                / np.maximum(1.0 - base_pd, 1e-6),
                0.0,
            )
        else:
            base_logit = np.log(
                np.clip(base_pd, clip, 1.0 - clip)
                / np.clip(1.0 - base_pd, clip, 1.0 - clip)
            )
            new_logit = np.log(
                np.clip(new_pd, clip, 1.0 - clip)
                / np.clip(1.0 - new_pd, clip, 1.0 - clip)
            )
            actual = np.maximum(new_logit - base_logit, 0.0)
        if predicted.size < 2 or np.std(predicted) == 0.0:
            modes[mode] = {"spearman": None, "pairs": len(predicted)}
            continue
        correlation, p_value = spearmanr(predicted, actual)
        replicates = []
        for _ in range(bootstrap_replicates):
            sample = rng.choice(indices, size=len(indices), replace=True)
            if np.std(predicted[sample]) == 0.0 or np.std(actual[sample]) == 0.0:
                continue
            replicates.append(spearmanr(predicted[sample], actual[sample])[0])
        ci = (
            [float(np.quantile(replicates, 0.025)),
             float(np.quantile(replicates, 0.975))]
            if replicates else None
        )
        modes[mode] = {
            "spearman": float(correlation),
            "spearman_p_value": float(p_value),
            "spearman_bootstrap_ci95": ci,
            "pairs": len(predicted),
            "actual_gains": actual.tolist(),
        }
    default = modes[actual_gain_mode]
    return {
        "spearman": default["spearman"],
        "spearman_p_value": default.get("spearman_p_value"),
        "spearman_bootstrap_ci95": default.get("spearman_bootstrap_ci95"),
        "pairs": len(predicted),
        "predicted_deflection": predicted.tolist(),
        "modes": modes,
    }
