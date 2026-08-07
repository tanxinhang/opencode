"""G5-S gate: joint RIS placement, phase, and report/control budget.

The fixed G5 RIS position is not necessarily near the blocked weak target.
This gate searches a small deployment candidate set for the RIS position,
keeps the target-aligned phase codebook and the G5-R control/report budget
identity, and selects the position maximizing worst-target expected P_D.
Complexity is ``O(positions x phase_resolutions x greedy)`` and the objective
is the exact expected-P_D metric with the monotone fusion family.
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


CANDIDATE_POSITIONS = [
    np.array([55.0, 15.0, 12.0]),
    np.array([45.0, 20.0, 8.0]),
    np.array([-35.0, 25.0, 8.0]),
    np.array([-40.0, 30.0, 6.0]),
    np.array([-30.0, 20.0, 10.0]),
    np.array([-50.0, 35.0, 6.0]),
    np.array([0.0, 20.0, 8.0]),
]


def run_gate(
    *, output: Path, seeds: int, total_budgets, coherence_frames_options,
    grid: int, ris_elements: int, aperture_scale: float, phase_bits: int,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    rows = []
    for total_budget in total_budgets:
        for coherence_frames in coherence_frames_options:
            for offset in range(seeds):
                seed = cfg.seed + offset
                no_ris_models = build_models(
                    cfg, np.random.default_rng(seed)
                )
                no_ris_selection = expected_pd_greedy_select(
                    no_ris_models, total_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                no_ris_vector = np.asarray(no_ris_selection.expected_pd)
                for position in CANDIDATE_POSITIONS:
                    ris = RisConfig(
                        position=position,
                        num_elements=ris_elements,
                        weak_target_id=cfg.num_targets - 1,
                        phase_bits=phase_bits,
                    )
                    overhead = ris_control_overhead_bits(
                        ris, coherence_frames=coherence_frames
                    )
                    report_budget = int(total_budget - overhead)
                    phases = [ris_beam_phase(target, ris) for target in targets]
                    gain = ris_physics_gain_matrix(
                        ris, transmitter_positions, targets, receiver,
                        aperture_scale, direct_blockage=0.01,
                        phase_per_target=phases,
                    )
                    models = build_models(
                        cfg, np.random.default_rng(seed), snr_gain=gain
                    )
                    selection = expected_pd_greedy_select(
                        models, report_budget, false_alarm_rate,
                        qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                    )
                    vector = np.asarray(selection.expected_pd)
                    rows.append({
                        "total_budget_bits": total_budget,
                        "coherence_frames": coherence_frames,
                        "seed_offset": offset,
                        "ris_position": position.tolist(),
                        "mean_expected_pd": float(np.mean(vector)),
                        "worst_expected_pd": float(np.min(vector)),
                        "no_ris_mean": float(np.mean(no_ris_vector)),
                        "no_ris_worst": float(np.min(no_ris_vector)),
                        "qos_feasible": bool(np.all(
                            vector >= qos_pd - 1e-9
                        )),
                    })

    summary = []
    for total_budget in total_budgets:
        for coherence_frames in coherence_frames_options:
            group = [
                row for row in rows
                if row["total_budget_bits"] == total_budget
                and row["coherence_frames"] == coherence_frames
            ]
            fixed = [row for row in group if row["ris_position"] == [55.0, 15.0, 12.0]]
            by_position: dict[tuple, list] = {}
            for row in group:
                by_position.setdefault(tuple(row["ris_position"]), []).append(row)

            def position_mean_worst(position_rows):
                return float(np.mean([
                    row["worst_expected_pd"] for row in position_rows
                ]))

            best_key = max(
                by_position, key=lambda key: position_mean_worst(by_position[key])
            )
            best = by_position[best_key]
            best_mean = float(np.mean([
                row["mean_expected_pd"] for row in best
            ]))
            best_worst = position_mean_worst(best)
            no_ris_mean = float(np.mean([
                row["no_ris_mean"] for row in group
            ]))
            no_ris_worst = float(np.mean([
                row["no_ris_worst"] for row in group
            ]))
            fixed_worst = float(np.mean([
                row["worst_expected_pd"] for row in fixed
            ]))
            summary.append({
                "total_budget_bits": total_budget,
                "coherence_frames": coherence_frames,
                "no_ris_mean": no_ris_mean,
                "no_ris_worst": no_ris_worst,
                "fixed_position_mean": float(np.mean([
                    row["mean_expected_pd"] for row in fixed
                ])),
                "fixed_position_worst": fixed_worst,
                "best_position": list(best_key),
                "best_mean_expected_pd": best_mean,
                "best_worst_expected_pd": best_worst,
                "mean_gain_best_vs_no_ris": best_mean - no_ris_mean,
                "worst_gain_best_vs_no_ris": best_worst - no_ris_worst,
                "worst_gain_best_vs_fixed": best_worst - fixed_worst,
                "qos_feasible_rate": float(np.mean([
                    row["qos_feasible"] for row in group
                ])),
            })

    payload = {
        "gate": "G5-S-ris-placement",
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "candidate_positions": [position.tolist() for position in CANDIDATE_POSITIONS],
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_placement_gate.json")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--total-budgets", type=int, nargs="+", default=[40, 60])
    parser.add_argument(
        "--coherence-frames", type=int, nargs="+", default=[64]
    )
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        total_budgets=args.total_budgets,
        coherence_frames_options=args.coherence_frames,
        grid=args.grid,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
    )


if __name__ == "__main__":
    main()
