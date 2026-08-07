"""G26 gate: mobility and time-varying blockage.

The scenario is upgraded from static geometry to a deterministic
time-varying scene:

- UAVs rotate around the receiver with a smooth trajectory;
- targets move on bounded circular paths;
- the weak-target direct-path blockage varies sinusoidally in time;
- RIS phase and subarray allocation are either static (designed at t=0) or
  adaptively recomputed each frame.

The resource identity and report budget are the same at every frame, and the
primary metric is worst-over-time QoS.
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

from uav_otfs_isac.architecture_objective import (
    aperture_constants,
    waterfilling_allocation,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.fusion import optimal_deflection
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
    ris_quantized_gain_loss,
)
from uav_otfs_isac.ris_subarray import multi_beam_phase
from uav_otfs_isac.scenario import (
    build_models,
    target_geometry,
    uav_geometry,
)


def rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    result = points.copy()
    result[:, 0] = cosine * points[:, 0] - sine * points[:, 1]
    result[:, 1] = sine * points[:, 0] + cosine * points[:, 1]
    return result


def target_path(position: np.ndarray, time_index: int, frames: int) -> np.ndarray:
    phase = 2.0 * np.pi * time_index / frames
    offset = np.array([
        3.0 * np.sin(phase + position[1] / 10.0),
        2.0 * np.cos(phase + position[0] / 10.0),
        0.0,
    ])
    return position + offset


def time_blockage(time_index: int, frames: int) -> float:
    return float(
        0.005 + 0.02 * (0.5 + 0.5 * np.sin(
            2.0 * np.pi * time_index / frames
        ))
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
    trajectories = []
    for time_index in range(frames):
        angle = 2.0 * np.pi * time_index / frames * 0.15
        positions = rotate_z(base_positions, angle)
        targets = [
            target_path(base_targets[q], time_index, frames)
            for q in range(cfg.num_targets)
        ]
        trajectories.append((positions, targets, time_blockage(time_index, frames)))

    initial_positions, initial_targets, _ = trajectories[0]
    initial_models = [
        build_models(
            cfg, np.random.default_rng(seed),
            transmitter_positions=initial_positions,
            target_positions=initial_targets,
        )
        for seed in seed_list
    ]
    initial_constants = aperture_constants(
        ris, initial_positions, initial_targets, receiver,
        aperture_scale, direct_blockage=time_blockage(0, frames),
    )
    initial_base = np.mean([
        [
            optimal_deflection(
                model.delta, model.sigma0, {model.owner}
            )
            for model in models
        ]
        for models in initial_models
    ], axis=0)
    static_allocation = waterfilling_allocation(
        ris_elements,
        initial_constants * ris_quantized_gain_loss(phase_bits),
        initial_base,
    )

    methods = {
        "no_ris": [],
        "ris_ideal": [],
        "ris_static_subarray": [],
        "ris_adaptive_subarray": [],
    }
    for seed in seed_list:
        for time_index, (positions, targets, blockage) in enumerate(trajectories):
            no_ris_models = build_models(
                cfg, np.random.default_rng(seed),
                transmitter_positions=positions, target_positions=targets,
            )
            no_ris_selection = expected_pd_greedy_select(
                no_ris_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["no_ris"].append(float(np.min(no_ris_selection.expected_pd)))

            phases = [ris_beam_phase(target, ris) for target in targets]
            ideal_gain = ris_physics_gain_matrix(
                ris, positions, targets, receiver, aperture_scale,
                direct_blockage=blockage, phase_per_target=phases,
            )
            ideal_models = build_models(
                cfg, np.random.default_rng(seed),
                transmitter_positions=positions, target_positions=targets,
                snr_gain=ideal_gain,
            )
            ideal_selection = expected_pd_greedy_select(
                ideal_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["ris_ideal"].append(float(np.min(ideal_selection.expected_pd)))

            static_phase = multi_beam_phase(
                ris, targets, static_allocation
            )
            static_gain = shared_phase_gain_matrix(
                ris, positions, targets, receiver, aperture_scale,
                direct_blockage=blockage, phase=static_phase,
            )
            static_models = build_models(
                cfg, np.random.default_rng(seed),
                transmitter_positions=positions, target_positions=targets,
                snr_gain=static_gain,
            )
            static_selection = expected_pd_greedy_select(
                static_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["ris_static_subarray"].append(
                float(np.min(static_selection.expected_pd))
            )

            frame_constants = aperture_constants(
                ris, positions, targets, receiver, aperture_scale,
                direct_blockage=blockage,
            )
            frame_models = build_models(
                cfg, np.random.default_rng(seed),
                transmitter_positions=positions, target_positions=targets,
            )
            frame_base = np.asarray([
                optimal_deflection(
                    model.delta, model.sigma0, {model.owner}
                )
                for model in frame_models
            ])
            adaptive_allocation = waterfilling_allocation(
                ris_elements,
                frame_constants * ris_quantized_gain_loss(phase_bits),
                frame_base,
            )
            adaptive_phase = multi_beam_phase(
                ris, targets, adaptive_allocation
            )
            adaptive_gain = shared_phase_gain_matrix(
                ris, positions, targets, receiver, aperture_scale,
                direct_blockage=blockage, phase=adaptive_phase,
            )
            adaptive_models = build_models(
                cfg, np.random.default_rng(seed),
                transmitter_positions=positions, target_positions=targets,
                snr_gain=adaptive_gain,
            )
            adaptive_selection = expected_pd_greedy_select(
                adaptive_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["ris_adaptive_subarray"].append(
                float(np.min(adaptive_selection.expected_pd))
            )

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
        "static_allocation": list(static_allocation),
        "methods": {},
    }
    for name in methods:
        summary["methods"][name] = {
            "worst_over_time": float(np.mean(per_seed_worst[name])),
            "mean_over_time": float(np.mean(methods[name])),
            "qos_over_time": float(np.mean(per_seed_qos[name])),
        }

    payload = {
        "gate": "G26-mobility-blockage",
        "seeds": seeds,
        "frames": frames,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/mobility_blockage_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--ris-elements", type=int, default=256)
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
