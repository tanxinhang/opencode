"""G20 gate: amplified distributed hard-decision detection.

The distributed branch is no longer a fixed baseline.  Each UAV uses a local
1-bit threshold whose local false-alarm rate is a designed parameter, and the
owner fuses the received hard decisions with an exact counting threshold.
The local P_FA and vote threshold are optimized per target to maximize P_D
under the global P_FA constraint, so the distributed detector has a derived
design objective instead of a fixed heuristic threshold.
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
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.ris_subarray import multi_beam_phase
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import (
    evaluate_schedule_expected_pd,
    hard_decision_fusion,
    optimized_hard_decision_fusion,
)


def local_hard_decision_schedule(models, budget_bits: int):
    """Per-target equal 1-bit schedule without global coordination."""
    per_target = max(1, budget_bits // len(models))
    scheduled = []
    used = 0
    for model in models:
        candidates = sorted(
            (
                float(model.delta[i] ** 2 / model.sigma0[i, i]),
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
    summary = []
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
        central_values = []
        owner_values = []
        hard_default_values = []
        hard_optimized_values = []
        hard_used = 0
        optimized_alphas = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            central = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            central_values.append(float(np.min(central.expected_pd)))
            owner = evaluate_schedule_expected_pd(
                models, tuple({model.owner} for model in models),
                false_alarm_rate, pd_mode="optimal", grid=grid,
            )
            owner_values.append(float(np.min(owner)))
            hard_schedule, hard_used = local_hard_decision_schedule(
                models, report_budget
            )
            defaults = []
            optimized = []
            for q, model in enumerate(models):
                defaults.append(hard_decision_fusion(
                    model, hard_schedule[q], false_alarm_rate
                )["pd"])
                best = optimized_hard_decision_fusion(
                    model, hard_schedule[q], false_alarm_rate
                )
                optimized.append(float(best["pd"]))
                optimized_alphas.append(float(best["local_false_alarm_rate"] or 0.1))
            hard_default_values.append(float(np.min(defaults)))
            hard_optimized_values.append(float(np.min(optimized)))
        methods = {
            "centralized_full": (
                np.mean(central_values), 1.0
            ),
            "owner_only": (np.mean(owner_values), 1.0),
            "hard_default": (np.mean(hard_default_values), 1.0),
            "hard_optimized": (np.mean(hard_optimized_values), 1.0),
        }
        cell = {
            "num_elements": num_elements,
            "phase_bits": phase_bits,
            "coherence_frames": coherence_frames,
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "position": position,
            "allocation": allocation,
            "hard_used_bits": hard_used,
            "mean_optimized_local_pfa": float(np.mean(optimized_alphas)),
            "methods": {},
        }
        for name, (worst, _) in methods.items():
            cell["methods"][name] = {
                "worst_expected_pd": float(worst),
                "qos_rate": float(worst >= qos_target - 1e-9),
            }
        summary.append(cell)

    payload = {
        "gate": "G20-amplified-distributed-detection",
        "seeds": seeds,
        "qos_target": qos_target,
        "aperture_scale": aperture_scale,
        "direct_blockage": direct_blockage,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/amplified_distributed_gate.json")
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
