"""G27 gate: multi-RIS joint deployment comparison.

The total RIS aperture is fixed at 256 elements.  The gate compares one,
two, and three RIS surfaces under the same total budget and the same
phase-bit/coherence control-overhead identity.  Multi-RIS power is summed
non-coherently, so splitting the aperture reduces coherent gain but adds
placement diversity.
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
from uav_otfs_isac.ris_scenario import RisConfig, ris_beam_phase
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, qos_target: float,
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
    ris_sets = {
        "one_ris": [
            RisConfig(
                position=np.array([0.0, 30.0, 6.0]),
                num_elements=total_elements,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            ),
        ],
        "two_ris": [
            RisConfig(
                position=np.array([0.0, 30.0, 6.0]),
                num_elements=total_elements // 2,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            ),
            RisConfig(
                position=np.array([20.0, 10.0, 2.0]),
                num_elements=total_elements - total_elements // 2,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            ),
        ],
        "three_ris": [
            RisConfig(
                position=np.array([0.0, 30.0, 6.0]),
                num_elements=total_elements // 3,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            ),
            RisConfig(
                position=np.array([20.0, 10.0, 2.0]),
                num_elements=total_elements // 3,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            ),
            RisConfig(
                position=np.array([-15.0, 40.0, 8.0]),
                num_elements=total_elements - 2 * (total_elements // 3),
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            ),
        ],
    }
    summary = []
    for total_budget in budgets:
        overheads = {
            name: multi_ris_control_overhead(
                ris_set, coherence_frames=coherence_frames
            )
            for name, ris_set in ris_sets.items()
        }
        no_ris_worst = []
        no_ris_qos = []
        for seed in seed_list:
            models = build_models(cfg, np.random.default_rng(seed))
            selection = expected_pd_greedy_select(
                models, total_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            no_ris_worst.append(float(np.min(selection.expected_pd)))
            no_ris_qos.append(bool(np.all(selection.expected_pd >= qos_target - 1e-9)))
        cell = {
            "total_budget_bits": total_budget,
            "no_ris_worst": float(np.mean(no_ris_worst)),
            "no_ris_qos": float(np.mean(no_ris_qos)),
            "methods": {},
        }
        for name, ris_set in ris_sets.items():
            report_budget = int(total_budget - overheads[name])
            if report_budget < 0:
                continue
            phases_per_ris = [
                [ris_beam_phase(target, ris) for target in targets]
                for ris in ris_set
            ]
            gain = multi_ris_physics_gain_matrix(
                ris_set, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=direct_blockage,
                phases_per_ris=phases_per_ris,
            )
            worsts = []
            qos = []
            for seed in seed_list:
                models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain
                )
                selection = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                    qos_weights=qos_weights, grid=grid,
                )
                worsts.append(float(np.min(selection.expected_pd)))
                qos.append(bool(np.all(selection.expected_pd >= qos_target - 1e-9)))
            cell["methods"][name] = {
                "report_budget_bits": report_budget,
                "control_overhead_bits": overheads[name],
                "worst_expected_pd": float(np.mean(worsts)),
                "qos_rate": float(np.mean(qos)),
            }
        summary.append(cell)

    payload = {
        "gate": "G27-multi-ris-deployment",
        "seeds": seeds,
        "qos_target": qos_target,
        "total_elements": total_elements,
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
    parser.add_argument("--output", default="results/multi_ris_gate.json")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 28, 40])
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
        budgets=args.budgets,
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
