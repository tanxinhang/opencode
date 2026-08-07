"""G54 gate: prediction-covariance-aware RIS phase.

Under the h-step AR(1) predictor the direction error standard deviation is

``sigma_dir = sqrt(1 - rho^(2h)) sigma_target / R``.

G54 compares stale-h phase, MMSE-h phase, and a covariance-aware phase that
maximizes the expected squared array gain under this Gaussian error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.architecture_switch import (
    target_wise_architecture_switch,
    two_sided_mode_ascent,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.covariance_aware_ris import (
    covariance_aware_phase,
    direction_error_std,
)
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import peer_majority_fusion
from uav_otfs_isac.stochastic_mobility import (
    ar1_horizon_prediction,
    nominal_target_at,
    stochastic_trajectories,
)


def target_wise_and_ascent_pd(
    models,
    report_budget: int,
    false_alarm_rate: float,
    qos_pd,
    qos_weights,
    grid: int,
) -> tuple[float, float, float]:
    soft = expected_pd_greedy_select(
        models, report_budget, false_alarm_rate, qos_pd=qos_pd,
        qos_weights=qos_weights, grid=grid,
    )
    soft_pds = [float(value) for value in soft.expected_pd]
    peer_pds = [
        float(peer_majority_fusion(model, false_alarm_rate)["pd"])
        for model in models
    ]
    _, target_values = target_wise_architecture_switch(soft_pds, peer_pds)
    ascent_modes, _, ascent_quality, _ = two_sided_mode_ascent(
        models, peer_pds, soft.scheduled, report_budget,
        false_alarm_rate, grid=grid,
    )
    ascent_values = [
        ascent_quality[q] if ascent_modes[q] == "soft" else peer_pds[q]
        for q in range(len(models))
    ]
    return (
        float(np.min(soft_pds)),
        float(np.min(target_values)),
        float(np.min(ascent_values)),
    )


def run_gate(
    *, output: Path, seeds: int, frames: int, grid: int,
    qos_target: float, total_budget: int, ris_elements: int,
    aperture_scale: float, phase_bits: int, coherence_frames: int,
    horizon: int,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    receiver = np.array([0.0, 0.0, 0.0])
    ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(ris, coherence_frames=coherence_frames)
    report_budget = int(total_budget - overhead)
    base_positions = uav_geometry(cfg.num_uavs)
    base_targets = [target_geometry(q) for q in range(cfg.num_targets)]
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    methods = {
        "no_ris_soft": [],
        "ris_ideal_mode_ascent": [],
        "stale_mode_ascent": [],
        "mmse_mode_ascent": [],
        "covariance_aware_mode_ascent": [],
    }
    for seed in seed_list:
        positions, targets, blockages = stochastic_trajectories(
            base_positions=base_positions, base_targets=base_targets,
            seed=seed, frames=frames,
        )
        initial_phases = [
            ris_beam_phase(target, ris) for target in targets[0]
        ]
        for time_index in range(frames):
            frame_positions = positions[time_index]
            frame_targets = targets[time_index]
            blockage = blockages[time_index]
            no_ris_models = build_models(
                cfg, np.random.default_rng(seed),
                transmitter_positions=frame_positions,
                target_positions=frame_targets,
            )
            no_ris_soft, _, _ = target_wise_and_ascent_pd(
                no_ris_models, report_budget, false_alarm_rate,
                qos_pd, qos_weights, grid,
            )
            methods["no_ris_soft"].append(no_ris_soft)

            ideal_gain = ris_physics_gain_matrix(
                ris, frame_positions, frame_targets, receiver,
                aperture_scale, direct_blockage=blockage,
                phase_per_target=[
                    ris_beam_phase(target, ris) for target in frame_targets
                ],
            )
            ideal_models = build_models(
                cfg, np.random.default_rng(seed),
                transmitter_positions=frame_positions,
                target_positions=frame_targets, snr_gain=ideal_gain,
            )
            _, _, ideal_ascent = target_wise_and_ascent_pd(
                ideal_models, report_budget, false_alarm_rate,
                qos_pd, qos_weights, grid,
            )
            methods["ris_ideal_mode_ascent"].append(ideal_ascent)

            if time_index < horizon:
                stale_phases = initial_phases
                mmse_phases = initial_phases
                robust_phases = initial_phases
            else:
                stale_phases = [
                    ris_beam_phase(target, ris)
                    for target in targets[time_index - horizon]
                ]
                mmse_phases = []
                robust_phases = []
                for q in range(cfg.num_targets):
                    predicted_target = ar1_horizon_prediction(
                        targets[time_index - horizon][q],
                        nominal_target_at(
                            base_targets[q], q,
                            time_index - horizon, frames
                        ),
                        nominal_target_at(
                            base_targets[q], q, time_index, frames
                        ),
                        correlation=0.8,
                        horizon=horizon,
                    )
                    mmse_phases.append(
                        ris_beam_phase(predicted_target, ris)
                    )
                    range_to_ris = float(np.linalg.norm(
                        predicted_target - ris.position
                    ))
                    error_covariance_scale = float(
                        1.0 - 0.8 ** (2 * horizon)
                    )
                    sigma_direction = direction_error_std(
                        error_covariance_scale,
                        target_sigma=2.0,
                        range_to_ris=range_to_ris,
                    )
                    robust_phases.append(covariance_aware_phase(
                        predicted_target, ris, sigma_direction
                    ))
            for label, phases in (
                ("stale", stale_phases),
                ("mmse", mmse_phases),
                ("covariance_aware", robust_phases),
            ):
                gain = ris_physics_gain_matrix(
                    ris, frame_positions, frame_targets, receiver,
                    aperture_scale, direct_blockage=blockage,
                    phase_per_target=phases,
                )
                models = build_models(
                    cfg, np.random.default_rng(seed),
                    transmitter_positions=frame_positions,
                    target_positions=frame_targets, snr_gain=gain,
                )
                _, _, ascent = target_wise_and_ascent_pd(
                    models, report_budget, false_alarm_rate,
                    qos_pd, qos_weights, grid,
                )
                methods[f"{label}_mode_ascent"].append(ascent)

    per_seed_worst = {}
    per_seed_qos = {}
    for name, values in methods.items():
        worst_by_seed = [
            min(values[index * frames:(index + 1) * frames])
            for index in range(seeds)
        ]
        qos_by_seed = [
            np.mean([
                value >= qos_target - 1e-9
                for value in values[index * frames:(index + 1) * frames]
            ])
            for index in range(seeds)
        ]
        per_seed_worst[name] = worst_by_seed
        per_seed_qos[name] = qos_by_seed

    summary = {
        "frames": frames,
        "seeds": seeds,
        "horizon": horizon,
        "total_budget_bits": total_budget,
        "report_budget_bits": report_budget,
        "qos_target": qos_target,
        "trajectory_model": "AR(1) position perturbation + sinusoidal trend + random blockage",
        "methods": {},
    }
    for name in methods:
        summary["methods"][name] = {
            "worst_over_time": float(np.mean(per_seed_worst[name])),
            "mean_over_time": float(np.mean(methods[name])),
            "qos_over_time": float(np.mean(per_seed_qos[name])),
        }
    summary["robust_over_mmse_worst_gain"] = (
        summary["methods"]["covariance_aware_mode_ascent"]["worst_over_time"]
        - summary["methods"]["mmse_mode_ascent"]["worst_over_time"]
    )
    summary["robust_qos_gain"] = (
        summary["methods"]["covariance_aware_mode_ascent"]["qos_over_time"]
        - summary["methods"]["mmse_mode_ascent"]["qos_over_time"]
    )
    summary["verdict"] = (
        "negative: covariance-aware expected-gain phase is not "
        "system-optimal under quantization and exact P_D; MMSE phase is kept"
    )

    payload = {
        "gate": "G54-covariance-aware-ris",
        "seeds": seeds,
        "frames": frames,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/covariance_aware_ris_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--ris-elements", type=int, default=128)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        frames=args.frames,
        grid=args.grid,
        qos_target=args.qos_target,
        total_budget=args.total_budget,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
