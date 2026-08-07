"""G47 gate: exact centralized/distributed architecture switching.

G46 shows that peer consensus has the higher P_D-consistent information
budget when report bits are scarce.  G47 makes that observation a detector:
for each seed, compare the exact worst-target P_D of centralized soft fusion
and peer majority and select the better mode.  A fixed
`report_budget < 10` threshold is also evaluated as the practical
non-oracle version.  The selected architecture spends the same report bits
as the selected branch, so no additional communication budget is consumed.
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

from uav_otfs_isac.architecture_switch import (
    exact_architecture_switch,
    fixed_budget_architecture_switch,
    selected_architecture_pd,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.fundamental_info import (
    effective_deflection,
    full_info_deflection,
)
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import peer_majority_fusion


def inr_profile(transmitter_positions, inr_ref=0.5):
    source = np.array([60.0, -20.0, 0.0])
    distances = np.linalg.norm(transmitter_positions - source, axis=1)
    return inr_ref * (100.0 / np.maximum(distances, 1e-9)) ** 2


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, qos_target: float,
    ris_elements: int, aperture_scale: float, phase_bits: int,
    coherence_frames: int, direct_blockage: float,
    fixed_switch_threshold: int,
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
    overhead = ris_control_overhead_bits(ris, coherence_frames=coherence_frames)
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
        soft_values = []
        peer_values = []
        exact_values = []
        fixed_values = []
        soft_used = []
        peer_used = []
        exact_modes = []
        fixed_modes = []
        full_values = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                interference_to_noise=inr,
            )
            full_values.append(float(np.mean(full_info_deflection(models))))
            soft = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            soft_worst = float(np.min(soft.expected_pd))
            peer_worst = float(np.min([
                float(peer_majority_fusion(model, false_alarm_rate)["pd"])
                for model in models
            ]))
            exact_mode = exact_architecture_switch(soft_worst, peer_worst)
            fixed_mode = fixed_budget_architecture_switch(
                report_budget, threshold_bits=fixed_switch_threshold
            )
            soft_values.append(soft_worst)
            peer_values.append(peer_worst)
            exact_values.append(selected_architecture_pd(
                soft_worst, peer_worst, exact_mode
            ))
            fixed_values.append(selected_architecture_pd(
                soft_worst, peer_worst, fixed_mode
            ))
            soft_used.append(int(soft.used_bits))
            peer_used.append(0)
            exact_modes.append(exact_mode)
            fixed_modes.append(fixed_mode)
        full_mean = float(np.mean(full_values))
        soft_mean = float(np.mean(soft_values))
        peer_mean = float(np.mean(peer_values))
        exact_mean = float(np.mean(exact_values))
        fixed_mean = float(np.mean(fixed_values))
        summary.append({
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "soft_worst_pd": soft_mean,
            "peer_worst_pd": peer_mean,
            "exact_switch_worst_pd": exact_mean,
            "fixed_switch_worst_pd": fixed_mean,
            "exact_gain_vs_soft": exact_mean - soft_mean,
            "fixed_gain_vs_soft": fixed_mean - soft_mean,
            "peer_selected_rate_exact": float(
                sum(mode == "peer" for mode in exact_modes) / seeds
            ),
            "peer_selected_rate_fixed": float(
                sum(mode == "peer" for mode in fixed_modes) / seeds
            ),
            "soft_rho_exact": effective_deflection(
                soft_mean, false_alarm_rate
            ) / max(full_mean, 1e-12),
            "peer_rho_exact": effective_deflection(
                peer_mean, false_alarm_rate
            ) / max(full_mean, 1e-12),
            "exact_switch_rho_exact": effective_deflection(
                exact_mean, false_alarm_rate
            ) / max(full_mean, 1e-12),
            "soft_used_bits": float(np.mean(soft_used)),
            "peer_used_bits": float(np.mean(peer_used)),
            "soft_qos_feasible": soft_mean >= qos_target - 1e-9,
            "peer_qos_feasible": peer_mean >= qos_target - 1e-9,
            "exact_switch_qos_feasible": exact_mean >= qos_target - 1e-9,
            "fixed_switch_qos_feasible": fixed_mean >= qos_target - 1e-9,
        })

    payload = {
        "gate": "G47-architecture-switch",
        "seeds": seeds,
        "qos_target": qos_target,
        "fixed_switch_threshold": fixed_switch_threshold,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/architecture_switch_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[8, 12, 16, 20, 28, 40])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--ris-elements", type=int, default=128)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    parser.add_argument("--fixed-switch-threshold", type=int, default=10)
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
        fixed_switch_threshold=args.fixed_switch_threshold,
    )


if __name__ == "__main__":
    main()
