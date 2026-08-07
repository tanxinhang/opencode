"""G44 gate: fundamental information-budget view.

All methods are mapped to one normalized information budget:

``rho = J_method / J_full``,

where ``J_full`` is the all-UAV deflection upper bound, ``J_soft`` is the
deflection of the greedy soft schedule, and ``J_hard`` is the total KL
information of the hard reports.  The gate checks whether P_D is monotone in
this budget across centralized soft, optimized hard, and peer consensus.
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
from uav_otfs_isac.fundamental_info import (
    full_info_deflection,
    hard_consensus_information,
    hard_kl_information,
    schedule_deflection,
)
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import (
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
        soft_pd = []
        hard_pd = []
        peer_pd = []
        soft_info = []
        hard_info = []
        peer_info = []
        full_info = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                interference_to_noise=inr,
            )
            full = full_info_deflection(models)
            full_info.append(float(np.mean(full)))
            soft = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            soft_pd.append(float(np.min(soft.expected_pd)))
            soft_info.append(float(np.mean(
                schedule_deflection(models, soft.scheduled)
            )))
            hard_values = []
            hard_kl = []
            for q, model in enumerate(models):
                candidates = sorted(
                    (
                        float(model.delta[i] ** 2 / model.sigma0[i, i]),
                        i,
                    )
                    for i in range(model.num_uavs)
                    if i != model.owner
                )
                candidates.reverse()
                per_target = min(
                    report_budget // cfg.num_targets,
                    model.num_uavs - 1,
                )
                schedule = {model.owner}
                for _, uav in candidates[:per_target]:
                    schedule.add(uav)
                hard_values.append(float(optimized_hard_decision_fusion(
                    model, schedule, false_alarm_rate
                )["pd"]))
                hard_kl.append(sum(
                    hard_kl_information(model, uav)
                    for uav in schedule if uav != model.owner
                ))
            hard_pd.append(float(np.min(hard_values)))
            hard_info.append(float(np.mean(hard_kl)))
            peer_pd.append(float(np.min([
                float(peer_majority_fusion(model, false_alarm_rate)["pd"])
                for model in models
            ])))
            peer_info.append(float(np.mean(hard_consensus_information(models))))
        summary.append({
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "soft_pd": float(np.mean(soft_pd)),
            "soft_info_norm": float(np.mean(soft_info) / max(np.mean(full_info), 1e-12)),
            "hard_pd": float(np.mean(hard_pd)),
            "hard_info_norm": float(np.mean(hard_info) / max(np.mean(full_info), 1e-12)),
            "peer_pd": float(np.mean(peer_pd)),
            "peer_info_norm": float(np.mean(peer_info) / max(np.mean(full_info), 1e-12)),
            "full_info": float(np.mean(full_info)),
        })

    payload = {
        "gate": "G44-fundamental-information-budget",
        "seeds": seeds,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/fundamental_information_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[8, 12, 16, 28, 40])
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
