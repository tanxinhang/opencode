"""G5-R gate: joint RIS control-bit and sensing report-bit allocation.

The RIS control plane and the sensing report plane compete for one total bit
budget ``B_total``:

``B_report = B_total - N * phase_bits / coherence_frames``.

The gate sweeps phase resolution and report schedules under this shared
budget, using the physics-based RIS channel (G5-P) and the expected-P_D
greedy selector.  It reports the best phase resolution per operating point and
the net gain over no-RIS with the full budget spent on reports.
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


def run_gate(
    *, output: Path, seeds: int, total_budgets, coherence_frames_options,
    grid: int, ris_elements: int, aperture_scale: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    phase_bits_options = (None, 1, 2, 3)
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
                for phase_bits in phase_bits_options:
                    ris = RisConfig(
                        position=np.array([55.0, 15.0, 12.0]),
                        num_elements=ris_elements,
                        weak_target_id=cfg.num_targets - 1,
                        phase_bits=phase_bits,
                    )
                    overhead = ris_control_overhead_bits(
                        ris, coherence_frames=coherence_frames
                    )
                    report_budget = int(total_budget - overhead)
                    if report_budget < 0:
                        continue
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
                        "phase_bits": phase_bits,
                        "control_overhead_bits": overhead,
                        "report_budget_bits": report_budget,
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
            best_by_mean = max(
                group, key=lambda row: row["mean_expected_pd"]
            )
            best_by_worst = max(
                group, key=lambda row: row["worst_expected_pd"]
            )
            quantized = [row for row in group if row["phase_bits"] is not None]
            best_quantized_by_mean = max(
                quantized, key=lambda row: row["mean_expected_pd"]
            )
            best_quantized_by_worst = max(
                quantized, key=lambda row: row["worst_expected_pd"]
            )
            summary.append({
                "total_budget_bits": total_budget,
                "coherence_frames": coherence_frames,
                "no_ris_mean": float(np.mean([
                    row["no_ris_mean"] for row in group
                ])),
                "no_ris_worst": float(np.mean([
                    row["no_ris_worst"] for row in group
                ])),
                "best_phase_bits_by_mean": best_by_mean["phase_bits"],
                "best_report_budget_by_mean": best_by_mean["report_budget_bits"],
                "best_mean_expected_pd": best_by_mean["mean_expected_pd"],
                "best_mean_gain_vs_no_ris": (
                    best_by_mean["mean_expected_pd"]
                    - best_by_mean["no_ris_mean"]
                ),
                "best_phase_bits_by_worst": best_by_worst["phase_bits"],
                "best_worst_expected_pd": best_by_worst["worst_expected_pd"],
                "best_worst_gain_vs_no_ris": (
                    best_by_worst["worst_expected_pd"]
                    - best_by_worst["no_ris_worst"]
                ),
                "best_quantized_phase_bits_by_mean": best_quantized_by_mean["phase_bits"],
                "best_quantized_report_budget_by_mean": (
                    best_quantized_by_mean["report_budget_bits"]
                ),
                "best_quantized_mean_expected_pd": best_quantized_by_mean["mean_expected_pd"],
                "best_quantized_mean_gain_vs_no_ris": (
                    best_quantized_by_mean["mean_expected_pd"]
                    - best_quantized_by_mean["no_ris_mean"]
                ),
                "best_quantized_phase_bits_by_worst": best_quantized_by_worst["phase_bits"],
                "best_quantized_worst_expected_pd": best_quantized_by_worst["worst_expected_pd"],
                "best_quantized_worst_gain_vs_no_ris": (
                    best_quantized_by_worst["worst_expected_pd"]
                    - best_quantized_by_worst["no_ris_worst"]
                ),
                "qos_feasible_rate": float(np.mean([
                    row["qos_feasible"] for row in group
                ])),
            })

    payload = {
        "gate": "G5-R-ris-joint-budget",
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_joint_budget_gate.json")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--total-budgets", type=int, nargs="+", default=[40, 60])
    parser.add_argument(
        "--coherence-frames", type=int, nargs="+", default=[64, 256]
    )
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        total_budgets=args.total_budgets,
        coherence_frames_options=args.coherence_frames,
        grid=args.grid,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
    )


if __name__ == "__main__":
    main()
