"""G53 gate: multi-step MMSE prediction under RIS reconfiguration latency.

G52 covers horizon one.  This gate generalizes the AR(1) conditional-mean
predictor to an arbitrary reconfiguration latency ``h``:

``hat p_{t|t-h} = n_t + rho^h (p_{t-h} - n_{t-h})``,

with prediction-error covariance ``(1 - rho^{2h}) sigma^2 I``.  The gate
compares stale ``p_{t-h}`` phase with MMSE-h phase for h=1,2,3.
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
    horizons = (1, 2, 3)
    methods = {
        "no_ris_soft": [],
        "ris_ideal_mode_ascent": [],
        "oracle_horizon_mode_ascent": [],
    }
    for horizon in horizons:
        methods[f"stale_h{horizon}_mode_ascent"] = []
        methods[f"mmse_h{horizon}_mode_ascent"] = []
    candidate_trajectories = []
    for seed in seed_list:
        seed_candidates = []
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

            horizon_ascent_values = []
            for horizon in horizons:
                if time_index < horizon:
                    stale_phases = initial_phases
                    mmse_phases = initial_phases
                else:
                    stale_phases = [
                        ris_beam_phase(target, ris)
                        for target in targets[time_index - horizon]
                    ]
                    mmse_phases = []
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
                for label, phases in (
                    (f"stale_h{horizon}", stale_phases),
                    (f"mmse_h{horizon}", mmse_phases),
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
                    horizon_ascent_values.append(ascent)
            methods["oracle_horizon_mode_ascent"].append(
                float(np.max(horizon_ascent_values))
            )
            seed_candidates.append(list(horizon_ascent_values))
        candidate_trajectories.append(seed_candidates)

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
        "total_budget_bits": total_budget,
        "report_budget_bits": report_budget,
        "qos_target": qos_target,
        "trajectory_model": "AR(1) position perturbation + sinusoidal trend + random blockage",
        "predictor": "h-step conditional-mean AR(1), rho=0.8",
        "methods": {},
        "horizons": [],
    }
    for name in methods:
        summary["methods"][name] = {
            "worst_over_time": float(np.mean(per_seed_worst[name])),
            "mean_over_time": float(np.mean(methods[name])),
            "qos_over_time": float(np.mean(per_seed_qos[name])),
        }
    for horizon in horizons:
        stale_name = f"stale_h{horizon}_mode_ascent"
        mmse_name = f"mmse_h{horizon}_mode_ascent"
        summary["horizons"].append({
            "horizon": horizon,
            "prediction_coefficient": float(0.8 ** horizon),
            "error_covariance_scale": float(1.0 - 0.8 ** (2 * horizon)),
            "stale_worst_over_time": summary["methods"][stale_name]["worst_over_time"],
            "mmse_worst_over_time": summary["methods"][mmse_name]["worst_over_time"],
            "mmse_over_stale_worst_gain": (
                summary["methods"][mmse_name]["worst_over_time"]
                - summary["methods"][stale_name]["worst_over_time"]
            ),
            "mmse_qos_over_time": summary["methods"][mmse_name]["qos_over_time"],
            "stale_qos_over_time": summary["methods"][stale_name]["qos_over_time"],
        })
    best_fixed_mmse_worst = max(
        summary["methods"][f"mmse_h{horizon}_mode_ascent"]["worst_over_time"]
        for horizon in horizons
    )
    summary["oracle_horizon_worst_over_time"] = (
        summary["methods"]["oracle_horizon_mode_ascent"]["worst_over_time"]
    )
    summary["oracle_over_best_fixed_mmse_worst_gain"] = (
        summary["oracle_horizon_worst_over_time"]
        - best_fixed_mmse_worst
    )

    hysteresis_rows = []
    deltas = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
    for delta in deltas:
        per_seed_worst = []
        per_seed_switches = []
        per_seed_qos = []
        for seed_candidates in candidate_trajectories:
            incumbent_index = None
            incumbent_value = None
            switches = 0
            frame_values = []
            for frame_candidates in seed_candidates:
                best_index = int(np.argmax(frame_candidates))
                best_value = float(frame_candidates[best_index])
                if incumbent_value is None:
                    incumbent_index = best_index
                    incumbent_value = best_value
                    frame_values.append(best_value)
                    continue
                current_incumbent_value = float(
                    frame_candidates[incumbent_index]
                )
                if best_value > current_incumbent_value + delta:
                    incumbent_index = best_index
                    incumbent_value = best_value
                    switches += 1
                frame_values.append(float(frame_candidates[incumbent_index]))
            per_seed_worst.append(min(frame_values))
            per_seed_switches.append(switches)
            per_seed_qos.append(float(np.mean([
                value >= qos_target - 1e-9 for value in frame_values
            ])))
        hysteresis_rows.append({
            "delta": delta,
            "worst_over_time": float(np.mean(per_seed_worst)),
            "mean_switches_per_seed": float(np.mean(per_seed_switches)),
            "qos_over_time": float(np.mean(per_seed_qos)),
        })
    summary["architecture_reconfiguration"] = hysteresis_rows
    switch_cost_analysis = []
    control_budget_bits = max(1, int(overhead))
    for switch_cost in (0, 1, 3, 6):
        best_delta = None
        best_worst = -1.0
        best_switches = None
        for row in hysteresis_rows:
            mean_switch_cost = (
                row["mean_switches_per_seed"] * switch_cost
            )
            if (
                mean_switch_cost <= control_budget_bits + 1e-9
                and row["worst_over_time"] > best_worst
            ):
                best_delta = row["delta"]
                best_worst = row["worst_over_time"]
                best_switches = row["mean_switches_per_seed"]
        switch_cost_analysis.append({
            "switch_cost_bits": switch_cost,
            "control_budget_bits": control_budget_bits,
            "best_delta": best_delta,
            "best_worst_over_time": best_worst,
            "best_switches": best_switches,
        })
    summary["switch_cost_analysis"] = switch_cost_analysis
    summary["oracle_horizon_qos_over_time"] = (
        summary["methods"]["oracle_horizon_mode_ascent"]["qos_over_time"]
    )

    payload = {
        "gate": "G53-multi-step-prediction",
        "seeds": seeds,
        "frames": frames,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary["horizons"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/multi_step_prediction_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--ris-elements", type=int, default=128)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
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
    )


if __name__ == "__main__":
    main()
