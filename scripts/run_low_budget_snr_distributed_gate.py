"""G40 gate: low-budget/low-SNR distributed comparison.

The RIS aperture is reduced to 128 elements, the total budget is lowered to
12/16/20, and a spatial interference source is added.  This pushes the
centralized worst P_D into the 0.6-0.7 regime so the distributed gap is
visible.
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
    degraded_peer_majority_fusion,
    optimized_hard_decision_fusion,
    peer_majority_fusion,
)


def inr_profile(transmitter_positions, inr_ref=0.5):
    source = np.array([60.0, -20.0, 0.0])
    distances = np.linalg.norm(
        transmitter_positions - source, axis=1
    )
    return inr_ref * (100.0 / np.maximum(distances, 1e-9)) ** 2


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
    inr = inr_profile(transmitter_positions)
    ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(
        ris, coherence_frames=coherence_frames
    )
    phases = [ris_beam_phase(target, ris) for target in targets]
    gain = ris_physics_gain_matrix(
        ris, transmitter_positions, targets, receiver, aperture_scale,
        direct_blockage=direct_blockage, phase_per_target=phases,
    )
    summary = []
    for total_budget in budgets:
        report_budget = int(total_budget - overhead)
        if report_budget < 0:
            continue
        methods = {
            "centralized_soft": [],
            "peer_clean": [],
            "peer_multihop": [],
            "hard_optimized": [],
        }
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                interference_to_noise=inr,
            )
            central = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            methods["centralized_soft"].append(
                float(np.min(central.expected_pd))
            )
            methods["peer_clean"].append(float(np.min([
                float(peer_majority_fusion(model, false_alarm_rate)["pd"])
                for model in models
            ])))
            methods["peer_multihop"].append(float(np.min([
                float(degraded_peer_majority_fusion(
                    model, false_alarm_rate,
                    per_hop_reliability=0.8, hops=3,
                )["pd"])
                for model in models
            ])))
            hard_values = []
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
                per_target = max(1, report_budget // cfg.num_targets)
                schedule = {model.owner}
                for _, uav in candidates[:per_target]:
                    schedule.add(uav)
                hard_values.append(float(optimized_hard_decision_fusion(
                    model, schedule, false_alarm_rate
                )["pd"]))
            methods["hard_optimized"].append(float(np.min(hard_values)))
        cell = {
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "ris_elements": ris_elements,
            "mean_inr": float(np.mean(inr)),
            "qos_target": qos_target,
            "methods": {},
        }
        for name, values in methods.items():
            worst = float(np.mean(values))
            cell["methods"][name] = {
                "worst_expected_pd": worst,
                "qos_feasible": worst >= qos_target - 1e-9,
            }
        summary.append(cell)

    payload = {
        "gate": "G40-low-budget-snr-distributed",
        "seeds": seeds,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/low_budget_snr_distributed_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.70)
    parser.add_argument("--ris-elements", type=int, default=128)
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
