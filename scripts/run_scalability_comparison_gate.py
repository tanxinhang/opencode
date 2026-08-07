"""G24 gate: performance comparison across target/UAV counts.

The gate varies the number of targets Q and UAVs M under a fair per-target
report budget and the same RIS control-overhead identity.  Three
architectures are compared:

1. no-RIS centralized soft fusion;
2. RIS per-target ideal phase + centralized soft fusion;
3. fully distributed peer majority with optimized local thresholds.

Report bits scale as ``20 * Q``; the RIS adds 12 control bits, so all
architectures use the same total budget.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def spaced_owners(num_uavs: int, num_targets: int) -> tuple[int, ...]:
    if num_targets == 1:
        return (0,)
    return tuple(
        int(round(index * (num_uavs - 1) / (num_targets - 1)))
        for index in range(num_targets)
    )


def run_gate(
    *, output: Path, seeds: int, grid: int, qos_target: float,
    ris_elements: int, aperture_scale: float, phase_bits: int,
    coherence_frames: int, direct_blockage: float,
    target_counts, uav_ratios,
) -> None:
    base = load_config("config/demo.yaml")
    false_alarm_rate = base.false_alarm_rate
    summary = []
    for num_targets in target_counts:
        for ratio in uav_ratios:
            num_uavs = num_targets * ratio
            owners = spaced_owners(num_uavs, num_targets)
            report_budget = 20 * num_targets
            cfg = replace(
                base,
                num_uavs=num_uavs,
                num_targets=num_targets,
                owners=owners,
                target_present=tuple([True] * num_targets),
                qos_min_deflection=tuple([3.0] * num_targets),
                qos_weights=tuple([1.0] * num_targets),
                performance_weights=tuple([1.0] * num_targets),
                report_budget_bits=report_budget,
            )
            cfg.validate()
            qos_pd = np.full(num_targets, qos_target)
            qos_weights = np.ones(num_targets)
            transmitter_positions = uav_geometry(num_uavs)
            targets = [target_geometry(q) for q in range(num_targets)]
            receiver = np.array([0.0, 0.0, 0.0])
            ris = RisConfig(
                position=np.array([0.0, 30.0, 6.0]),
                num_elements=ris_elements,
                weak_target_id=num_targets - 1,
                phase_bits=phase_bits,
            )
            overhead = ris_control_overhead_bits(
                ris, coherence_frames=coherence_frames
            )
            total_budget = report_budget + int(overhead)
            phases = [ris_beam_phase(target, ris) for target in targets]
            gain = ris_physics_gain_matrix(
                ris, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=direct_blockage,
                phase_per_target=phases,
            )
            no_ris_worst = []
            ris_worst = []
            peer_worst = []
            for offset in range(seeds):
                seed = cfg.seed + offset
                no_ris_models = build_models(
                    cfg, np.random.default_rng(seed)
                )
                no_ris_selection = expected_pd_greedy_select(
                    no_ris_models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                no_ris_worst.append(float(np.min(no_ris_selection.expected_pd)))
                ris_models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain
                )
                ris_selection = expected_pd_greedy_select(
                    ris_models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                ris_worst.append(float(np.min(ris_selection.expected_pd)))
                peer_values = [
                    float(peer_majority_fusion(model, false_alarm_rate)["pd"])
                    for model in ris_models
                ]
                peer_worst.append(float(np.min(peer_values)))
            summary.append({
                "num_targets": num_targets,
                "num_uavs": num_uavs,
                "uav_to_target_ratio": ratio,
                "report_budget_bits": report_budget,
                "total_budget_bits": total_budget,
                "no_ris_worst": float(np.mean(no_ris_worst)),
                "no_ris_qos": float(np.mean([
                    value >= qos_target - 1e-9 for value in no_ris_worst
                ])),
                "ris_ideal_worst": float(np.mean(ris_worst)),
                "ris_ideal_qos": float(np.mean([
                    value >= qos_target - 1e-9 for value in ris_worst
                ])),
                "peer_majority_worst": float(np.mean(peer_worst)),
                "peer_majority_qos": float(np.mean([
                    value >= qos_target - 1e-9 for value in peer_worst
                ])),
            })

    payload = {
        "gate": "G24-scalability-comparison",
        "seeds": seeds,
        "qos_target": qos_target,
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "coherence_frames": coherence_frames,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/scalability_comparison_gate.json")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    parser.add_argument("--target-counts", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--uav-ratios", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        grid=args.grid,
        qos_target=args.qos_target,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
        target_counts=args.target_counts,
        uav_ratios=args.uav_ratios,
    )


if __name__ == "__main__":
    main()
