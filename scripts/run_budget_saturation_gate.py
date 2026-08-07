"""G6 gate: budget saturation frontier for selection and architecture.

The question is whether a large report budget is really required.  This gate
sweeps total budget under the joint RIS control/report identity and compares
forward greedy, discrete coordinate ascent, and the all-scheduled upper
bound, with and without the G5-T RIS deployment.  It reports the minimum
total budget at which the worst-target expected P_D reaches the QoS target,
so saturation can be attributed to selection or to the sensing architecture.
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
from uav_otfs_isac.discrete_descent import discrete_gradient_select
from uav_otfs_isac.expected_pd import (
    expected_gaussian_detection_probability,
    expected_pd_greedy_select,
)
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def all_scheduled_pd(models, false_alarm_rate, grid):
    return np.asarray([
        expected_gaussian_detection_probability(
            model, set(range(model.num_uavs)), false_alarm_rate,
            pd_mode="optimal", grid=grid,
        )
        for model in models
    ])


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, qos_target: float,
    ris_elements: int, aperture_scale: float, phase_bits: int,
    coherence_frames: int, direct_blockage: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris_position = np.array([0.0, 30.0, 6.0])
    ris = RisConfig(
        position=ris_position,
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(ris, coherence_frames=coherence_frames)

    rows = []
    for offset in range(seeds):
        seed = cfg.seed + offset
        no_ris_models = build_models(cfg, np.random.default_rng(seed))
        phases = [ris_beam_phase(target, ris) for target in targets]
        gain = ris_physics_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase_per_target=phases,
        )
        ris_models = build_models(
            cfg, np.random.default_rng(seed), snr_gain=gain
        )
        for scenario, models, report_budget_fn in (
            ("no_ris", no_ris_models, lambda total: total),
            ("ris", ris_models, lambda total: int(total - overhead)),
        ):
            for total_budget in budgets:
                report_budget = report_budget_fn(total_budget)
                if report_budget < 0:
                    continue
                greedy = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                    qos_weights=qos_weights, grid=grid,
                )
                descent = discrete_gradient_select(
                    models, report_budget, false_alarm_rate,
                    init_schedule=greedy.scheduled, qos_pd=qos_pd,
                    qos_weights=qos_weights, grid=grid,
                )
                all_values = all_scheduled_pd(
                    models, false_alarm_rate, grid
                )
                rows.append({
                    "seed_offset": offset,
                    "scenario": scenario,
                    "total_budget_bits": total_budget,
                    "report_budget_bits": report_budget,
                    "greedy_mean": float(np.mean(greedy.expected_pd)),
                    "greedy_worst": float(np.min(greedy.expected_pd)),
                    "descent_mean": float(np.mean(descent.expected_pd)),
                    "descent_worst": float(np.min(descent.expected_pd)),
                    "all_mean": float(np.mean(all_values)),
                    "all_worst": float(np.min(all_values)),
                    "greedy_qos": bool(np.all(greedy.expected_pd >= qos_target - 1e-9)),
                    "descent_qos": bool(np.all(descent.expected_pd >= qos_target - 1e-9)),
                })

    summary = []
    for scenario in ("no_ris", "ris"):
        for total_budget in budgets:
            group = [
                row for row in rows
                if row["scenario"] == scenario
                and row["total_budget_bits"] == total_budget
            ]
            if not group:
                continue
            greedy_worst = np.asarray([row["greedy_worst"] for row in group])
            descent_worst = np.asarray([row["descent_worst"] for row in group])
            all_worst = np.asarray([row["all_worst"] for row in group])
            summary.append({
                "scenario": scenario,
                "total_budget_bits": total_budget,
                "report_budget_bits": group[0]["report_budget_bits"],
                "greedy_mean": float(np.mean([row["greedy_mean"] for row in group])),
                "greedy_worst": float(np.mean(greedy_worst)),
                "descent_mean": float(np.mean([row["descent_mean"] for row in group])),
                "descent_worst": float(np.mean(descent_worst)),
                "all_mean": float(np.mean([row["all_mean"] for row in group])),
                "all_worst": float(np.mean(all_worst)),
                "selection_gain_mean": float(np.mean([
                    row["descent_mean"] - row["greedy_mean"] for row in group
                ])),
                "selection_gain_worst": float(np.mean([
                    row["descent_worst"] - row["greedy_worst"] for row in group
                ])),
                "gap_to_all_scheduled_worst": float(np.mean(
                    all_worst - descent_worst
                )),
                "greedy_qos_rate": float(np.mean([
                    row["greedy_qos"] for row in group
                ])),
                "descent_qos_rate": float(np.mean([
                    row["descent_qos"] for row in group
                ])),
            })

    minimum_budget = {}
    for scenario in ("no_ris", "ris"):
        budgets_by_seed = []
        for offset in range(seeds):
            found = None
            for total_budget in sorted(budgets):
                match = [
                    row for row in rows
                    if row["seed_offset"] == offset
                    and row["scenario"] == scenario
                    and row["total_budget_bits"] == total_budget
                ]
                if match and match[0]["descent_qos"]:
                    found = total_budget
                    break
            budgets_by_seed.append(found)
        minimum_budget[scenario] = {
            "per_seed": budgets_by_seed,
            "mean": float(np.mean([value for value in budgets_by_seed if value is not None])) if any(
                value is not None for value in budgets_by_seed
            ) else None,
            "achieved_seed_rate": float(np.mean([
                value is not None for value in budgets_by_seed
            ])),
        }

    payload = {
        "gate": "G6-budget-saturation",
        "qos_target": qos_target,
        "seeds": seeds,
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "coherence_frames": coherence_frames,
        "minimum_budget_for_qos": minimum_budget,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "minimum_budget_for_qos": minimum_budget,
        "summary": summary,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/budget_saturation_gate.json")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--budgets", type=int, nargs="+", default=[
        20, 24, 28, 32, 36, 40, 44,
    ])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--ris-elements", type=int, default=256)
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
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
