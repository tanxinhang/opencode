"""G10 gate: per-subarray steering-cosine optimization.

The G9 gate optimizes how many elements each target gets.  This gate fixes
those allocations and additionally optimizes each subarray's continuous
steering cosine by coordinate ascent, which trades cross-block interference
against self-block gain without changing total aperture or phase-bits.
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
from uav_otfs_isac.ris_optimization import (
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
from uav_otfs_isac.ris_subarray import (
    coordinate_block_steering_ascent,
    multi_beam_phase,
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
    g9_result: Path,
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
    weak_phase = ris_beam_phase_from_cosine(ris, weak_cosine)
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    with g9_result.open(encoding="utf-8") as handle:
        g9 = json.load(handle)
    allocations = {
        int(budget): optimization["allocation"]
        for budget, optimization in g9["optimization_by_budget"].items()
    }

    steering_by_budget = {}
    for total_budget in budgets:
        report_budget = int(total_budget - overhead)
        allocation = allocations[total_budget]
        if report_budget < 0:
            continue

        def objective(
            cosines: tuple[float, ...],
            report_budget: int = report_budget,
            allocation: tuple[int, ...] = tuple(allocation),
        ) -> float:
            phase = multi_beam_phase(
                ris, targets, allocation, steering_cosines=cosines
            )
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
            return float(np.mean(worsts))

        result = coordinate_block_steering_ascent(
            ris, targets, allocation, objective,
            step=0.1, grid_points=7, max_rounds=2,
            initial_cosines=target_cosines,
        )
        result["report_budget_bits"] = report_budget
        steering_by_budget[total_budget] = result

    rows = []
    for offset in range(seeds):
        seed = cfg.seed + offset
        rng_phase = np.random.default_rng(seed + 1000000)
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
        scenario_models = {
            "no_ris": (no_ris_models, None),
            "random_shared_phase": (build_models(
                cfg, np.random.default_rng(seed), snr_gain=random_gain
            ), overhead),
            "shared_weak_aligned": (build_models(
                cfg, np.random.default_rng(seed), snr_gain=weak_gain
            ), overhead),
            "per_target_ideal_phase": (build_models(
                cfg, np.random.default_rng(seed), snr_gain=ideal_gain
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
            allocation = allocations[total_budget]
            default_phase = multi_beam_phase(ris, targets, allocation)
            default_gain = shared_phase_gain_matrix(
                ris, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=direct_blockage,
                phase=default_phase,
            )
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=default_gain
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            values = np.asarray(selection.expected_pd)
            rows.append({
                "seed_offset": offset,
                "scenario": "g9_subarray",
                "total_budget_bits": total_budget,
                "report_budget_bits": report_budget,
                "mean_expected_pd": float(np.mean(values)),
                "worst_expected_pd": float(np.min(values)),
                "qos_feasible": bool(np.all(values >= qos_target - 1e-9)),
            })
            cosines = steering_by_budget[total_budget]["steering_cosines"]
            steering_phase = multi_beam_phase(
                ris, targets, allocation, steering_cosines=cosines
            )
            steering_gain = shared_phase_gain_matrix(
                ris, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=direct_blockage,
                phase=steering_phase,
            )
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=steering_gain
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            values = np.asarray(selection.expected_pd)
            rows.append({
                "seed_offset": offset,
                "scenario": "g10_steering_optimized",
                "total_budget_bits": total_budget,
                "report_budget_bits": report_budget,
                "mean_expected_pd": float(np.mean(values)),
                "worst_expected_pd": float(np.min(values)),
                "qos_feasible": bool(np.all(values >= qos_target - 1e-9)),
            })

    summary = []
    for scenario in ("no_ris", "random_shared_phase", "shared_weak_aligned",
                     "per_target_ideal_phase", "g9_subarray",
                     "g10_steering_optimized"):
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
                         "shared_weak_aligned", "per_target_ideal_phase",
                         "g9_subarray"):
            for metric in ("mean", "worst"):
                proposed = [
                    row["mean_expected_pd" if metric == "mean" else "worst_expected_pd"]
                    for row in rows
                    if row["scenario"] == "g10_steering_optimized"
                    and row["total_budget_bits"] == budget
                ]
                baseline_values = [
                    row["mean_expected_pd" if metric == "mean" else "worst_expected_pd"]
                    for row in rows
                    if row["scenario"] == baseline
                    and row["total_budget_bits"] == budget
                ]
                sections[f"g10_vs_{baseline}_B{budget}_{metric}"] = bootstrap_ci([
                    value - base for value, base in zip(proposed, baseline_values)
                ])

    payload = {
        "gate": "G10-ris-subarray-steering",
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "coherence_frames": coherence_frames,
        "qos_target": qos_target,
        "steering_by_budget": steering_by_budget,
        "summary": summary,
        "sections": sections,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "steering_by_budget": steering_by_budget,
        "summary": summary,
        "sections": sections,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_subarray_steering_gate.json")
    parser.add_argument("--g9-result", type=Path,
                        default="results/ris_subarray_gate.json")
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
        g9_result=args.g9_result,
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
