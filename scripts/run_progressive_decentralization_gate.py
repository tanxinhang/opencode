"""G19 gate: progressive decentralization ablation.

The centralized baseline uses global expected-P_D greedy scheduling and the
P_D-optimal linear fusion family.  Decentralization is opened in stages:

1. ``local_schedule_optimal``: fair round-robin local scheduling, optimal
   fusion at the center;
2. ``local_schedule_deflection``: fair round-robin scheduling, deflection
   fusion;
3. ``owner_only``: no cross-UAV reports, owner decision only;
4. ``hard_decision_local``: per-target 1-bit hard decisions plus counting
   fusion at the center.

Each stage isolates the loss from scheduling coordination, fusion optimality,
reporting, and soft-information quality respectively.
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

from uav_otfs_isac.baselines import sensing_quality_score
from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.ris_subarray import multi_beam_phase
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import (
    evaluate_schedule_expected_pd,
    hard_decision_fusion,
)


def round_robin_schedule(
    models,
    budget_bits: int,
    cost_bits: int = 5,
):
    """Fair local scheduling: round-robin over targets, static local scores."""
    scheduled = [{model.owner} for model in models]
    used = 0
    while True:
        changed = False
        for q, model in enumerate(models):
            if used + cost_bits > budget_bits:
                continue
            candidates = sorted(
                (
                    float(sensing_quality_score(model, i)),
                    i,
                )
                for i in range(model.num_uavs)
                if i != model.owner and i not in scheduled[q]
            )
            candidates.reverse()
            if candidates:
                scheduled[q].add(candidates[0][1])
                used += cost_bits
                changed = True
        if not changed:
            break
    return tuple(frozenset(group) for group in scheduled), used


def local_hard_decision_schedule(models, budget_bits: int):
    """Per-target equal 1-bit schedule without global coordination."""
    per_target = max(1, budget_bits // len(models))
    scheduled = []
    used = 0
    for model in models:
        candidates = sorted(
            (
                float(sensing_quality_score(model, i)),
                i,
            )
            for i in range(model.num_uavs)
            if i != model.owner
        )
        candidates.reverse()
        chosen = {model.owner}
        for _, uav in candidates[:per_target]:
            chosen.add(uav)
            used += 1
        scheduled.append(chosen)
    return tuple(frozenset(group) for group in scheduled), used


def run_gate(
    *, output: Path, seeds: int, grid: int, qos_target: float,
    aperture_scale: float, direct_blockage: float, g18_result: Path,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    with g18_result.open(encoding="utf-8") as handle:
        g18 = json.load(handle)
    rows = []
    for config_cell in g18["summary"]:
        num_elements = config_cell["num_elements"]
        phase_bits = config_cell["phase_bits"]
        coherence_frames = config_cell["coherence_frames"]
        total_budget = config_cell["total_budget_bits"]
        report_budget = config_cell["report_budget_bits"]
        position = config_cell["final_position"]
        allocation = config_cell["final_allocation"]
        ris = RisConfig(
            position=np.asarray(position, dtype=float),
            num_elements=num_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        phase = multi_beam_phase(ris, targets, allocation)
        gain = shared_phase_gain_matrix(
            ris, transmitter_positions, targets, receiver,
            aperture_scale,
            direct_blockage=direct_blockage, phase=phase,
        )
        for offset in range(seeds):
            seed = cfg.seed + offset
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            centralized = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            local_schedule, local_used = round_robin_schedule(
                models, report_budget
            )
            local_optimal = evaluate_schedule_expected_pd(
                models, local_schedule, false_alarm_rate,
                pd_mode="optimal", grid=grid,
            )
            local_deflection = evaluate_schedule_expected_pd(
                models, local_schedule, false_alarm_rate,
                pd_mode="deflection", grid=grid,
            )
            owner_only = evaluate_schedule_expected_pd(
                models, tuple({model.owner} for model in models),
                false_alarm_rate, pd_mode="optimal", grid=grid,
            )
            hard_schedule, hard_used = local_hard_decision_schedule(
                models, report_budget
            )
            hard_values = np.asarray([
                hard_decision_fusion(
                    model, hard_schedule[q], false_alarm_rate
                )["pd"]
                for q, model in enumerate(models)
            ])
            methods = {
                "centralized_full": (
                    np.asarray(centralized.expected_pd), report_budget
                ),
                "local_schedule_optimal": (local_optimal, local_used),
                "local_schedule_deflection": (local_deflection, local_used),
                "owner_only": (
                    owner_only, 0
                ),
                "hard_decision_local": (hard_values, hard_used),
            }
            row = {
                "seed_offset": offset,
                "num_elements": num_elements,
                "phase_bits": phase_bits,
                "coherence_frames": coherence_frames,
                "total_budget_bits": total_budget,
                "report_budget_bits": report_budget,
                "position": position,
                "allocation": allocation,
            }
            for name, (values, used) in methods.items():
                row[f"{name}_mean"] = float(np.mean(values))
                row[f"{name}_worst"] = float(np.min(values))
                row[f"{name}_used_bits"] = int(used)
                row[f"{name}_qos"] = bool(np.all(values >= qos_target - 1e-9))
            rows.append(row)

    summary = []
    for config_key in sorted({
        (
            row["num_elements"],
            row["phase_bits"],
            row["coherence_frames"],
            row["total_budget_bits"],
        )
        for row in rows
    }):
        group = [
            row for row in rows
            if (
                row["num_elements"],
                row["phase_bits"],
                row["coherence_frames"],
                row["total_budget_bits"],
            ) == config_key
        ]
        cell = {
            "num_elements": config_key[0],
            "phase_bits": config_key[1],
            "coherence_frames": config_key[2],
            "total_budget_bits": config_key[3],
            "report_budget_bits": group[0]["report_budget_bits"],
            "position": group[0]["position"],
            "allocation": group[0]["allocation"],
            "methods": {},
        }
        for method in ("centralized_full", "local_schedule_optimal",
                       "local_schedule_deflection", "owner_only",
                       "hard_decision_local"):
            values = {
                metric: float(np.mean([
                    row[f"{method}_{metric}"] for row in group
                ]))
                for metric in ("mean", "worst")
            }
            values["used_bits"] = int(group[0][f"{method}_used_bits"])
            values["qos_rate"] = float(np.mean([
                row[f"{method}_qos"] for row in group
            ]))
            values["mean_loss_vs_centralized"] = (
                values["mean"]
                - float(np.mean([row["centralized_full_mean"] for row in group]))
            )
            values["worst_loss_vs_centralized"] = (
                values["worst"]
                - float(np.mean([row["centralized_full_worst"] for row in group]))
            )
            cell["methods"][method] = values
        summary.append(cell)

    payload = {
        "gate": "G19-progressive-decentralization",
        "seeds": seeds,
        "qos_target": qos_target,
        "aperture_scale": aperture_scale,
        "direct_blockage": direct_blockage,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/progressive_decentralization_gate.json")
    parser.add_argument("--g18-result", type=Path,
                        default="results/joint_placement_allocation_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        g18_result=args.g18_result,
        seeds=args.seeds,
        grid=args.grid,
        qos_target=args.qos_target,
        aperture_scale=args.aperture_scale,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
