"""G21 gate: progressive network-level decentralization.

This gate relaxes the distributed assumptions further than G19/G20:

1. ``centralized_soft``: centralized soft fusion upper bound;
2. ``hard_full_links``: optimized 1-bit hard fusion with all scheduled links;
3. ``hard_top_k``: optimized 1-bit fusion using only the best K reporting
   links to each owner;
4. ``peer_majority``: every UAV votes locally and the target is declared by
   an optimized majority threshold, with no owner fusion center, no report
   links, and no global scheduling.

The stages isolate the value of connectivity, topology, and a fusion center.
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
    optimized_hard_decision_fusion,
    peer_majority_fusion,
)


def local_hard_schedule(models, budget_bits):
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


def top_k_links(model, schedule, k):
    reports = [
        i for i in schedule
        if i != model.owner
    ]
    ranked = sorted(
        reports,
        key=lambda i: (
            float(model.success_prob[i]) * (1.0 - float(model.bit_flip_prob[i]))
            * float(model.delta[i] ** 2 / model.sigma0[i, i])
        ),
        reverse=True,
    )
    return frozenset([model.owner, *ranked[:k]])


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
        accumulators = {
            "centralized_soft": [],
            "hard_full_links": [],
            "hard_top5": [],
            "hard_top3": [],
            "hard_top1": [],
            "peer_majority": [],
        }
        used_bits = 0
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain
            )
            central = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            accumulators["centralized_soft"].append(
                float(np.min(central.expected_pd))
            )
            hard_schedule, used_bits = local_hard_schedule(
                models, report_budget
            )
            full = []
            top5 = []
            top3 = []
            top1 = []
            peer_values = []
            for q, model in enumerate(models):
                full.append(float(optimized_hard_decision_fusion(
                    model, hard_schedule[q], false_alarm_rate
                )["pd"]))
                for key, k in (("hard_top5", 5), ("hard_top3", 3), ("hard_top1", 1)):
                    limited = top_k_links(model, hard_schedule[q], k)
                    value = optimized_hard_decision_fusion(
                        model, limited, false_alarm_rate
                    )["pd"]
                    {
                        "hard_top5": top5,
                        "hard_top3": top3,
                        "hard_top1": top1,
                    }[key].append(float(value))
                peer = peer_majority_fusion(model, false_alarm_rate)
                peer_values.append(float(peer["pd"]))
            accumulators["hard_full_links"].append(float(np.min(full)))
            accumulators["hard_top5"].append(float(np.min(top5)))
            accumulators["hard_top3"].append(float(np.min(top3)))
            accumulators["hard_top1"].append(float(np.min(top1)))
            accumulators["peer_majority"].append(float(np.min(peer_values)))
        cell = {
            "num_elements": num_elements,
            "phase_bits": phase_bits,
            "coherence_frames": coherence_frames,
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "position": position,
            "allocation": allocation,
            "hard_used_bits": used_bits,
            "methods": {},
        }
        for name, values in accumulators.items():
            worst = float(np.mean(values))
            cell["methods"][name] = {
                "worst_expected_pd": worst,
                "qos_rate": float(worst >= qos_target - 1e-9),
            }
        summary.append(cell)

    payload = {
        "gate": "G21-network-decentralization",
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
    parser.add_argument("--output", default="results/network_decentralization_gate.json")
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
