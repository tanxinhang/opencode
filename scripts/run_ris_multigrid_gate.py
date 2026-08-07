"""G5-T gate: coarse-to-fine multigrid RIS placement.

G5-S searched a fixed candidate set.  This gate adds one local refinement
around the best coarse deployment, which is selected by mean worst-target
expected P_D over the seed set.  The multigrid search evaluates
``P_coarse + P_fine`` deployments, each followed by the expected-P_D greedy,
so its complexity is ``O((P_coarse + P_fine) x greedy)``; halving the local
grid spacing doubles the resolution in each axis while adding a bounded
number of deployments.
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

from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


COARSE_POSITIONS = [
    np.array([55.0, 15.0, 12.0]),
    np.array([45.0, 20.0, 8.0]),
    np.array([-35.0, 25.0, 8.0]),
    np.array([-40.0, 30.0, 6.0]),
    np.array([-30.0, 20.0, 10.0]),
    np.array([-50.0, 35.0, 6.0]),
    np.array([0.0, 20.0, 8.0]),
]


def fine_grid(center, span, z_span):
    x = [center[0] + offset for offset in (-span, 0.0, span)]
    y = [center[1] + offset for offset in (-span, 0.0, span)]
    z = [center[2] + offset for offset in (-z_span, 0.0, z_span)]
    return [
        np.array([float(xi), float(yi), float(zi)])
        for xi in x for yi in y for zi in z
    ]


def run_gate(
    *, output: Path, seeds: int, total_budget: int, coherence_frames: int,
    grid: int, ris_elements: int, aperture_scale: float, phase_bits: int,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris = RisConfig(
        position=np.zeros(3),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(ris, coherence_frames=coherence_frames)
    report_budget = int(total_budget - overhead)

    def evaluate_position(position, seeds_for_evaluation):
        config = RisConfig(
            position=position,
            num_elements=ris_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        phases = [ris_beam_phase(target, config) for target in targets]
        gain = ris_physics_gain_matrix(
            config, transmitter_positions, targets, receiver,
            aperture_scale, direct_blockage=0.01, phase_per_target=phases,
        )
        means = []
        worsts = []
        for seed in seeds_for_evaluation:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            vector = np.asarray(selection.expected_pd)
            means.append(float(np.mean(vector)))
            worsts.append(float(np.min(vector)))
        return float(np.mean(means)), float(np.mean(worsts))

    seed_list = [cfg.seed + offset for offset in range(seeds)]
    no_ris_models = [
        build_models(cfg, np.random.default_rng(seed)) for seed in seed_list
    ]
    no_ris_values = [
        expected_pd_greedy_select(
            models, total_budget, false_alarm_rate, qos_pd=qos_pd,
            qos_weights=qos_weights, grid=grid,
        ).expected_pd
        for models in no_ris_models
    ]
    no_ris_mean = float(np.mean([np.mean(v) for v in no_ris_values]))
    no_ris_worst = float(np.mean([np.min(v) for v in no_ris_values]))

    coarse_results = [evaluate_position(position, seed_list) for position in COARSE_POSITIONS]
    coarse_index = int(np.argmax([result[1] for result in coarse_results]))
    coarse_position = COARSE_POSITIONS[coarse_index]
    fine_positions = fine_grid(coarse_position, span=10.0, z_span=2.0)
    fine_results = [evaluate_position(position, seed_list) for position in fine_positions]
    fine_index = int(np.argmax([result[1] for result in fine_results]))
    best_position = fine_positions[fine_index]
    best_mean, best_worst = fine_results[fine_index]
    coarse_mean, coarse_worst = coarse_results[coarse_index]
    fixed_position = COARSE_POSITIONS[0]
    fixed_mean, fixed_worst = evaluate_position(fixed_position, seed_list)

    payload = {
        "gate": "G5-T-ris-multigrid",
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "total_budget_bits": total_budget,
        "coherence_frames": coherence_frames,
        "report_budget_bits": report_budget,
        "evaluations": len(COARSE_POSITIONS) + len(fine_positions),
        "no_ris_mean": no_ris_mean,
        "no_ris_worst": no_ris_worst,
        "fixed_position": fixed_position.tolist(),
        "fixed_mean": fixed_mean,
        "fixed_worst": fixed_worst,
        "coarse_position": coarse_position.tolist(),
        "coarse_mean": coarse_mean,
        "coarse_worst": coarse_worst,
        "best_position": best_position.tolist(),
        "best_mean": best_mean,
        "best_worst": best_worst,
        "mean_gain_best_vs_no_ris": best_mean - no_ris_mean,
        "worst_gain_best_vs_no_ris": best_worst - no_ris_worst,
        "worst_gain_best_vs_fixed": best_worst - fixed_worst,
        "worst_gain_fine_vs_coarse": best_worst - coarse_worst,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "best_position", "best_mean", "best_worst", "no_ris_mean",
        "no_ris_worst", "fixed_worst", "coarse_worst", "evaluations",
        "mean_gain_best_vs_no_ris", "worst_gain_best_vs_no_ris",
        "worst_gain_best_vs_fixed", "worst_gain_fine_vs_coarse",
    )}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_multigrid_gate.json")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        total_budget=args.total_budget,
        coherence_frames=args.coherence_frames,
        grid=args.grid,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
    )


if __name__ == "__main__":
    main()
