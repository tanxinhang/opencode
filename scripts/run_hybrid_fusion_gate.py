"""G31 gate: soft/hard hybrid fusion within each target.

Each target can schedule a small number of 5-bit soft reports and additional
1-bit hard reports under the same per-target budget.  The fusion score is an
exact Gaussian-plus-hard LLR rule, so the hybrid detector is a true
soft/hard combination rather than a best-of selection.
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
from uav_otfs_isac.hybrid_fusion import hybrid_gaussian_hard_pd
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import optimized_hard_decision_fusion


def quality_rank(model):
    return sorted(
        (
            float(model.delta[i] ** 2 / model.sigma0[i, i]),
            i,
        )
        for i in range(model.num_uavs)
        if i != model.owner
    )


def hybrid_schedule(model, budget_bits, num_targets):
    per_target = budget_bits // num_targets
    ranked = quality_rank(model)
    ranked.reverse()
    soft_count = min(
        len(ranked),
        max(0, (per_target - 5) // 5) + (1 if per_target >= 5 else 0),
    )
    soft = {model.owner}
    used = 0
    for _, uav in ranked[:soft_count]:
        soft.add(uav)
        used += 5
    hard = set()
    for _, uav in ranked[soft_count:]:
        if used + 1 > per_target:
            break
        hard.add(uav)
        used += 1
    return soft, hard


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
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(ris, coherence_frames=coherence_frames)
    phases = [ris_beam_phase(target, ris) for target in targets]
    gain = ris_physics_gain_matrix(
        ris, transmitter_positions, targets, receiver, aperture_scale,
        direct_blockage=direct_blockage, phase_per_target=phases,
    )
    summary = []
    for total_budget in budgets:
        report_budget = int(total_budget - overhead)
        methods = {"soft5": [], "hard1": [], "hybrid": []}
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            soft_selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            methods["soft5"].append(float(np.min(soft_selection.expected_pd)))
            hard_values = []
            hybrid_values = []
            for model in models:
                ranked = quality_rank(model)
                ranked.reverse()
                per_target = max(1, report_budget // cfg.num_targets)
                hard_schedule = {model.owner}
                for _, uav in ranked[:per_target]:
                    hard_schedule.add(uav)
                hard_values.append(float(optimized_hard_decision_fusion(
                    model, hard_schedule, false_alarm_rate
                )["pd"]))
                soft, hard = hybrid_schedule(
                    model, report_budget, cfg.num_targets
                )
                hybrid_values.append(float(hybrid_gaussian_hard_pd(
                    model, soft, hard, false_alarm_rate, grid=grid
                )["pd"]))
            methods["hard1"].append(float(np.min(hard_values)))
            methods["hybrid"].append(float(np.min(hybrid_values)))
        cell = {
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "methods": {},
        }
        for name, values in methods.items():
            worst = float(np.mean(values))
            cell["methods"][name] = {
                "worst_expected_pd": worst,
                "qos_rate": float(worst >= qos_target - 1e-9),
            }
        summary.append(cell)

    payload = {
        "gate": "G31-hybrid-soft-hard-fusion",
        "seeds": seeds,
        "qos_target": qos_target,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/hybrid_fusion_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[28, 40])
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
