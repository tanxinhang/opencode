"""G8 gate: exact quota-constrained selection upper bound.

Report costs are equal in the audited model, so the selection layer is a
cardinality-constrained submodular problem.  This gate evaluates every
per-target report subset exactly, finds the best subset of each size, and
searches all per-target report quotas.  It compares this exact upper bound
with forward greedy under the same RIS/no-RIS channel and budget identity.
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
from uav_otfs_isac.exact_quota_selection import exact_quota_select
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


def bootstrap_ci(values, seed=20260805, replicates=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = np.arange(values.size)
    samples = []
    for _ in range(replicates):
        sample = rng.choice(indices, size=values.size, replace=True)
        samples.append(float(np.mean(values[sample])))
    return {
        "mean": float(np.mean(values)),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "win_rate": float(np.mean(values > 1e-6)),
        "pairs": int(values.size),
    }


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
    ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
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
        for scenario, models in (
            ("no_ris", no_ris_models),
            ("ris", ris_models),
        ):
            for total_budget in budgets:
                report_budget = (
                    total_budget
                    if scenario == "no_ris"
                    else int(total_budget - overhead)
                )
                if report_budget < 0:
                    continue
                greedy = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                    qos_weights=qos_weights, grid=grid,
                )
                exact = exact_quota_select(
                    models, report_budget, false_alarm_rate, qos_pd=qos_pd,
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
                    "exact_mean": float(np.mean(exact.expected_pd)),
                    "exact_worst": float(np.min(exact.expected_pd)),
                    "all_mean": float(np.mean(all_values)),
                    "all_worst": float(np.min(all_values)),
                    "greedy_qos": bool(np.all(
                        greedy.expected_pd >= qos_target - 1e-9
                    )),
                    "exact_qos": bool(np.all(
                        exact.expected_pd >= qos_target - 1e-9
                    )),
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
            summary.append({
                "scenario": scenario,
                "total_budget_bits": total_budget,
                "report_budget_bits": group[0]["report_budget_bits"],
                "greedy_mean": float(np.mean([
                    row["greedy_mean"] for row in group
                ])),
                "greedy_worst": float(np.mean([
                    row["greedy_worst"] for row in group
                ])),
                "exact_mean": float(np.mean([
                    row["exact_mean"] for row in group
                ])),
                "exact_worst": float(np.mean([
                    row["exact_worst"] for row in group
                ])),
                "all_mean": float(np.mean([
                    row["all_mean"] for row in group
                ])),
                "all_worst": float(np.mean([
                    row["all_worst"] for row in group
                ])),
                "selection_gain_mean": float(np.mean([
                    row["exact_mean"] - row["greedy_mean"] for row in group
                ])),
                "selection_gain_worst": float(np.mean([
                    row["exact_worst"] - row["greedy_worst"] for row in group
                ])),
                "gap_to_all_worst": float(np.mean([
                    row["all_worst"] - row["exact_worst"] for row in group
                ])),
                "greedy_qos_rate": float(np.mean([
                    row["greedy_qos"] for row in group
                ])),
                "exact_qos_rate": float(np.mean([
                    row["exact_qos"] for row in group
                ])),
            })

    sections = {}
    for scenario in ("no_ris", "ris"):
        for budget in budgets:
            for metric in ("mean", "worst"):
                differences = [
                    row[f"exact_{metric}"] - row[f"greedy_{metric}"]
                    for row in rows
                    if row["scenario"] == scenario
                    and row["total_budget_bits"] == budget
                ]
                if differences:
                    sections[f"exact_vs_greedy_{scenario}_B{budget}_{metric}"] = (
                        bootstrap_ci(differences)
                    )

    payload = {
        "gate": "G8-exact-quota-selection",
        "qos_target": qos_target,
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "coherence_frames": coherence_frames,
        "summary": summary,
        "sections": sections,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "sections": sections}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/exact_quota_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 28, 40])
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
