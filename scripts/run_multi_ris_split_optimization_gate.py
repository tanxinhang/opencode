"""G28 gate: multi-RIS aperture split and placement optimization.

For a single target and aligned RISs, the reflected power sum is
``sum_r N_r^2 / L_r`` with fixed total ``N``.  Since ``x^2`` is convex, the
maximum over a simplex occurs at an extreme point: all aperture should go to
the RIS with the smallest cascaded loss ``L_r``.  A split can only help when
targets have different geometry losses, so this gate optimizes the split and
the second RIS position on the exact system objective.
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
from uav_otfs_isac.multi_ris import (
    multi_ris_control_overhead,
    multi_ris_physics_gain_matrix,
)
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


BOUNDS = np.array([
    [-20.0, 20.0],
    [10.0, 50.0],
    [0.0, 12.0],
])


def run_gate(
    *, output: Path, seeds: int, grid: int, qos_target: float,
    total_elements: int, aperture_scale: float, phase_bits: int,
    coherence_frames: int, direct_blockage: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    first_position = np.array([0.0, 30.0, 6.0])
    second_position = np.array([20.0, 10.0, 2.0])
    overhead = multi_ris_control_overhead(
        [
            RisConfig(
                position=first_position,
                num_elements=total_elements // 2,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            ),
            RisConfig(
                position=second_position,
                num_elements=total_elements - total_elements // 2,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            ),
        ],
        coherence_frames=coherence_frames,
    )
    report_budget = int(40 - overhead)

    def system_objective(first_elements: int, second_position_value):
        first_ris = RisConfig(
            position=first_position,
            num_elements=first_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        second_ris = RisConfig(
            position=np.asarray(second_position_value, dtype=float),
            num_elements=total_elements - first_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        phases_per_ris = [
            [ris_beam_phase(target, ris) for target in targets]
            for ris in (first_ris, second_ris)
        ]
        gain = multi_ris_physics_gain_matrix(
            [first_ris, second_ris], transmitter_positions, targets,
            receiver, aperture_scale, direct_blockage=direct_blockage,
            phases_per_ris=phases_per_ris,
        )
        worsts = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            worsts.append(float(np.min(selection.expected_pd)))
        return float(np.mean(worsts))

    def one_ris_value():
        ris = RisConfig(
            position=first_position,
            num_elements=total_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        phases = [ris_beam_phase(target, ris) for target in targets]
        gain = multi_ris_physics_gain_matrix(
            [ris], transmitter_positions, targets, receiver,
            aperture_scale, direct_blockage=direct_blockage,
            phases_per_ris=[phases],
        )
        worsts = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            worsts.append(float(np.min(selection.expected_pd)))
        return float(np.mean(worsts))

    best_split = total_elements // 2
    best_position = second_position.copy()
    best_value = system_objective(best_split, best_position)
    history = [{
        "first_elements": best_split,
        "second_position": best_position.tolist(),
        "value": best_value,
    }]
    for step in (16, 8):
        improved = True
        while improved:
            improved = False
            for delta in (-step, step):
                candidate_split = best_split + delta
                if candidate_split <= 0 or candidate_split >= total_elements:
                    continue
                value = system_objective(candidate_split, best_position)
                if value > best_value + 1e-12:
                    best_split = candidate_split
                    best_value = value
                    improved = True
                    history.append({
                        "first_elements": best_split,
                        "second_position": best_position.tolist(),
                        "value": best_value,
                    })
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = best_position.copy()
                    candidate[axis] += sign * step
                    if (candidate[axis] < BOUNDS[axis, 0]
                            or candidate[axis] > BOUNDS[axis, 1]):
                        continue
                    value = system_objective(best_split, candidate)
                    if value > best_value + 1e-12:
                        best_position = candidate
                        best_value = value
                        improved = True
                        history.append({
                            "first_elements": best_split,
                            "second_position": best_position.tolist(),
                            "value": best_value,
                        })

    one_value = one_ris_value()
    equal_value = system_objective(total_elements // 2, np.array([20.0, 10.0, 2.0]))
    optimized_value = system_objective(best_split, best_position)
    summary = {
        "report_budget_bits": report_budget,
        "one_ris_value": one_value,
        "one_ris_qos": float(one_value >= qos_target - 1e-9),
        "equal_split_value": equal_value,
        "equal_split_qos": float(equal_value >= qos_target - 1e-9),
        "optimized_split": best_split,
        "optimized_second_position": best_position.tolist(),
        "optimized_value": optimized_value,
        "optimized_qos": float(optimized_value >= qos_target - 1e-9),
        "history": history,
    }
    payload = {
        "gate": "G28-multi-ris-split-optimization",
        "seeds": seeds,
        "total_elements": total_elements,
        "phase_bits": phase_bits,
        "coherence_frames": coherence_frames,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/multi_ris_split_optimization_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--total-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        grid=args.grid,
        qos_target=args.qos_target,
        total_elements=args.total_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
