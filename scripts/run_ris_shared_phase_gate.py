"""G7 gate: continuous shared-phase RIS optimization under tight budgets.

The G5 chain configures a different phase profile per target.  This gate
instead optimizes one physical phase profile shared by all targets, using
analytic array-power gradients.  It evaluates the optimized profile at total
budgets 20/28/40 under the same control/report identity, so the result shows
whether a single RIS phase profile can carry the tight-budget QoS claim
without per-target time multiplexing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.ris_optimization import (
    projected_gradient_shared_phase,
    ris_beam_phase_from_cosine,
    shared_phase_gain_matrix,
    target_direction_cosines,
)
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


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
    target_cosines = target_direction_cosines(ris, targets)
    weak_cosine = float(target_cosines[cfg.num_targets - 1])
    optimization = projected_gradient_shared_phase(
        ris, targets, surrogate="worst", max_steps=120
    )
    optimized_cosine = float(optimization["steering_cosine"])
    optimized_phase = ris_beam_phase_from_cosine(ris, optimized_cosine)
    weak_phase = ris_beam_phase_from_cosine(ris, weak_cosine)
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    system_optimization_by_budget = {}
    for total_budget in budgets:
        report_budget = int(total_budget - overhead)
        if report_budget < 0:
            continue

        def system_objective(
            steering_cosine: float,
            report_budget: int = report_budget,
        ) -> float:
            phase = ris_beam_phase_from_cosine(ris, float(steering_cosine))
            gain = shared_phase_gain_matrix(
                ris, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=direct_blockage, phase=phase,
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
            return -float(np.mean(worsts))

        cosine_grid = np.linspace(-1.0, 1.0, 101)
        grid_values = [system_objective(float(u)) for u in cosine_grid]
        best_index = int(np.argmin(grid_values))
        lo = cosine_grid[max(best_index - 1, 0)]
        hi = cosine_grid[min(best_index + 1, cosine_grid.size - 1)]
        scalar_result = minimize_scalar(
            system_objective, bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-3, "maxiter": 30},
        )
        system_optimization_by_budget[total_budget] = {
            "steering_cosine": float(scalar_result.x),
            "system_worst_pd": float(-scalar_result.fun),
            "report_budget_bits": report_budget,
            "evaluations": len(grid_values) + int(scalar_result.nfev),
        }

    rows = []
    for offset in range(seeds):
        seed = cfg.seed + offset
        rng_phase = np.random.default_rng(seed + 800000)
        random_shared_phase = rng_phase.uniform(
            0.0, 2.0 * np.pi, ris_elements
        )
        no_ris_models = build_models(cfg, np.random.default_rng(seed))
        ideal_phases = [ris_beam_phase(target, ris) for target in targets]
        ideal_gain = ris_physics_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase_per_target=ideal_phases,
        )
        random_gain = shared_phase_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase=random_shared_phase,
        )
        weak_gain = shared_phase_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase=weak_phase,
        )
        optimized_gain = shared_phase_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase=optimized_phase,
        )
        scenario_models = {
            "no_ris": (no_ris_models, None),
            "random_shared_phase": (build_models(
                cfg, np.random.default_rng(seed), snr_gain=random_gain
            ), overhead),
            "per_target_ideal_phase": (build_models(
                cfg, np.random.default_rng(seed), snr_gain=ideal_gain
            ), overhead),
            "shared_weak_aligned": (build_models(
                cfg, np.random.default_rng(seed), snr_gain=weak_gain
            ), overhead),
            "shared_surrogate_optimized": (build_models(
                cfg, np.random.default_rng(seed), snr_gain=optimized_gain
            ), overhead),
        }
        for scenario, (models, ris_overhead) in scenario_models.items():
            for total_budget in budgets:
                report_budget = (
                    total_budget
                    if ris_overhead is None
                    else int(total_budget - ris_overhead)
                )
                if report_budget < 0:
                    continue
                selection = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                    qos_weights=qos_weights, grid=grid,
                )
                values = np.asarray(selection.expected_pd)
                rows.append({
                    "seed_offset": offset,
                    "scenario": scenario,
                    "total_budget_bits": total_budget,
                    "report_budget_bits": report_budget,
                    "mean_expected_pd": float(np.mean(values)),
                    "worst_expected_pd": float(np.min(values)),
                    "qos_feasible": bool(np.all(values >= qos_target - 1e-9)),
                })
        for total_budget in budgets:
            report_budget = int(total_budget - overhead)
            if report_budget < 0:
                continue
            system_cosine = system_optimization_by_budget[total_budget][
                "steering_cosine"
            ]
            phase = ris_beam_phase_from_cosine(ris, system_cosine)
            gain = shared_phase_gain_matrix(
                ris, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=direct_blockage, phase=phase,
            )
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            values = np.asarray(selection.expected_pd)
            rows.append({
                "seed_offset": offset,
                "scenario": "shared_system_optimized",
                "total_budget_bits": total_budget,
                "report_budget_bits": report_budget,
                "mean_expected_pd": float(np.mean(values)),
                "worst_expected_pd": float(np.min(values)),
                "qos_feasible": bool(np.all(values >= qos_target - 1e-9)),
            })

    summary = []
    for scenario in ("no_ris", "random_shared_phase", "per_target_ideal_phase",
                     "shared_weak_aligned", "shared_surrogate_optimized",
                     "shared_system_optimized"):
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
                "mean_expected_pd": float(np.mean([
                    row["mean_expected_pd"] for row in group
                ])),
                "worst_expected_pd": float(np.mean([
                    row["worst_expected_pd"] for row in group
                ])),
                "qos_feasible_rate": float(np.mean([
                    row["qos_feasible"] for row in group
                ])),
            })

    sections = {}
    for budget in budgets:
        for baseline in ("no_ris", "random_shared_phase",
                         "per_target_ideal_phase", "shared_weak_aligned",
                         "shared_surrogate_optimized"):
            for metric in ("mean", "worst"):
                differences = [
                    row["mean_expected_pd" if metric == "mean" else "worst_expected_pd"]
                    for row in rows
                    if row["scenario"] == "shared_system_optimized"
                    and row["total_budget_bits"] == budget
                ]
                baseline_values = [
                    row["mean_expected_pd" if metric == "mean" else "worst_expected_pd"]
                    for row in rows
                    if row["scenario"] == baseline
                    and row["total_budget_bits"] == budget
                ]
                sections[f"shared_optimized_vs_{baseline}_B{budget}_{metric}"] = (
                    bootstrap_ci([
                        value - base
                        for value, base in zip(differences, baseline_values)
                    ])
                )

    payload = {
        "gate": "G7-ris-shared-phase-gradient",
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "coherence_frames": coherence_frames,
        "qos_target": qos_target,
        "projected_gradient": optimization,
        "system_optimization_by_budget": system_optimization_by_budget,
        "target_cosines": target_cosines.tolist(),
        "weak_target_cosine": weak_cosine,
        "summary": summary,
        "sections": sections,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "projected_gradient": optimization,
        "system_optimization_by_budget": system_optimization_by_budget,
        "summary": summary,
        "sections": sections,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_shared_phase_gate.json")
    parser.add_argument("--seeds", type=int, default=6)
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
