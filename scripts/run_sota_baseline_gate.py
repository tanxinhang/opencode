"""G5-SOTA gate: literature-style baselines under the same resource budget.

All methods use the same physics-based RIS channel, the same total budget
identity ``B_total = B_report + N_ris * phase_bits / coherence_frames``, and
the same per-seed geometry/reporting randomness.  The proposed method is
expected-P_D greedy plus the P_D-optimal linear fusion family; the baselines
are static deflection Top-K, uniform per-target soft allocation, random RIS
phase, and 1-bit hard-decision counting fusion.
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
from uav_otfs_isac.sota_baselines import (
    evaluate_schedule_expected_pd,
    hard_decision_fusion,
    hard_decision_schedule,
    static_deflection_schedule,
    uniform_soft_schedule,
)


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
    *, output: Path, seeds: int, total_budget: int, coherence_frames: int,
    grid: int, ris_elements: int, aperture_scale: float, phase_bits: int,
    direct_blockage: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
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
    report_budget = int(total_budget - overhead)

    rows = []
    for offset in range(seeds):
        seed = cfg.seed + offset
        rng_phase = np.random.default_rng(seed + 700000)
        no_ris_models = build_models(cfg, np.random.default_rng(seed))
        phases = [ris_beam_phase(target, ris) for target in targets]
        aligned_gain = ris_physics_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase_per_target=phases,
        )
        ris_models = build_models(
            cfg, np.random.default_rng(seed), snr_gain=aligned_gain
        )
        random_phases = [
            rng_phase.uniform(0.0, 2.0 * np.pi, ris_elements)
            for _ in targets
        ]
        random_gain = ris_physics_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase_per_target=random_phases,
        )
        random_models = build_models(
            cfg, np.random.default_rng(seed), snr_gain=random_gain
        )

        proposed = expected_pd_greedy_select(
            ris_models, report_budget, false_alarm_rate, qos_pd=qos_pd,
            qos_weights=qos_weights, grid=grid,
        )
        proposed_pd = np.asarray(proposed.expected_pd)

        s1_schedule, s1_used = static_deflection_schedule(ris_models, report_budget)
        s1_pd = evaluate_schedule_expected_pd(
            ris_models, s1_schedule, false_alarm_rate,
            pd_mode="deflection", grid=grid,
        )
        s2_schedule, s2_used = static_deflection_schedule(
            no_ris_models, report_budget
        )
        s2_pd = evaluate_schedule_expected_pd(
            no_ris_models, s2_schedule, false_alarm_rate,
            pd_mode="deflection", grid=grid,
        )
        s3_schedule, s3_used = static_deflection_schedule(
            random_models, report_budget
        )
        s3_pd = evaluate_schedule_expected_pd(
            random_models, s3_schedule, false_alarm_rate,
            pd_mode="deflection", grid=grid,
        )
        s4_schedule, s4_used = uniform_soft_schedule(
            no_ris_models, reports_per_target=1
        )
        s4_pd = evaluate_schedule_expected_pd(
            no_ris_models, s4_schedule, false_alarm_rate,
            pd_mode="deflection", grid=grid,
        )
        proposed_deflection_pd = evaluate_schedule_expected_pd(
            ris_models, proposed.scheduled, false_alarm_rate,
            pd_mode="deflection", grid=grid,
        )

        hard_no_ris_schedule, hard_no_ris_used = hard_decision_schedule(
            no_ris_models, report_budget
        )
        hard_no_ris_pd = np.asarray([
            hard_decision_fusion(
                model, hard_no_ris_schedule[q], false_alarm_rate
            )["pd"]
            for q, model in enumerate(no_ris_models)
        ])
        hard_no_ris_pfa = np.asarray([
            hard_decision_fusion(
                model, hard_no_ris_schedule[q], false_alarm_rate
            )["pfa"]
            for q, model in enumerate(no_ris_models)
        ])
        hard_ris_schedule, hard_ris_used = hard_decision_schedule(
            ris_models, report_budget
        )
        hard_ris_pd = np.asarray([
            hard_decision_fusion(
                model, hard_ris_schedule[q], false_alarm_rate
            )["pd"]
            for q, model in enumerate(ris_models)
        ])
        hard_ris_pfa = np.asarray([
            hard_decision_fusion(
                model, hard_ris_schedule[q], false_alarm_rate
            )["pfa"]
            for q, model in enumerate(ris_models)
        ])

        methods = {
            "proposed": (proposed_pd, report_budget),
            "proposed_schedule_deflection": (proposed_deflection_pd, report_budget),
            "s1_ris_deflection_topk": (s1_pd, s1_used),
            "s2_no_ris_deflection_topk": (s2_pd, s2_used),
            "s3_random_ris_deflection_topk": (s3_pd, s3_used),
            "s4_uniform_soft_no_ris": (s4_pd, s4_used),
            "hard_no_ris": (hard_no_ris_pd, hard_no_ris_used),
            "hard_ris": (hard_ris_pd, hard_ris_used),
        }
        row = {"seed_offset": offset}
        for name, (values, used) in methods.items():
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_worst"] = float(np.min(values))
            row[f"{name}_used_bits"] = int(used)
        row["hard_no_ris_pfa"] = float(np.mean(hard_no_ris_pfa))
        row["hard_ris_pfa"] = float(np.mean(hard_ris_pfa))
        rows.append(row)

    sections = {}
    for baseline in (
        "s1_ris_deflection_topk",
        "s2_no_ris_deflection_topk",
        "s3_random_ris_deflection_topk",
        "s4_uniform_soft_no_ris",
        "hard_no_ris",
        "hard_ris",
        "proposed_schedule_deflection",
    ):
        for metric in ("mean", "worst"):
            differences = [
                row[f"proposed_{metric}"] - row[f"{baseline}_{metric}"]
                for row in rows
            ]
            sections[f"proposed_vs_{baseline}_{metric}"] = bootstrap_ci(
                differences
            )

    payload = {
        "gate": "G5-SOTA-baselines",
        "ris_position": ris_position.tolist(),
        "total_budget_bits": total_budget,
        "report_budget_bits": report_budget,
        "coherence_frames": coherence_frames,
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "direct_blockage": direct_blockage,
        "sections": sections,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(sections, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/sota_baseline_gate.json")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
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
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
