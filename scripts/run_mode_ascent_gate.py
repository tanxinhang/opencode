"""G50 gate: two-sided mode ascent.

G49 only adds freed bits to targets already in centralized mode.  G50 adds a
second update direction: a peer target may spend unused report bits on its
original soft schedule and switch back to centralized soft whenever its
exact P_D strictly exceeds the peer majority P_D.  Failed upgrade attempts
are discarded, so every accepted step is monotone and the worst-target P_D
cannot decrease.
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
    target_wise_architecture_switch,
    two_sided_mode_ascent,
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
        soft_worst = []
        peer_worst = []
        global_worst = []
        target_wise_worst = []
        ascent_worst = []
        ascent_used = []
        peer_to_soft_switches = []
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
            soft_pds = [float(value) for value in soft.expected_pd]
            peer_pds = [
                float(peer_majority_fusion(model, false_alarm_rate)["pd"])
                for model in models
            ]
            g48_modes, g48_values = target_wise_architecture_switch(
                soft_pds, peer_pds
            )
            soft_worst.append(float(np.min(soft_pds)))
            peer_worst.append(float(np.min(peer_pds)))
            global_worst.append(max(
                float(np.min(soft_pds)),
                float(np.min(peer_pds)),
            ))
            target_wise_worst.append(float(np.min(g48_values)))

            ascent_modes, _, ascent_quality, ascent_bits = (
                two_sided_mode_ascent(
                    models, peer_pds, soft.scheduled, report_budget,
                    false_alarm_rate, grid=grid,
                )
            )
            ascent_values = [
                ascent_quality[q] if ascent_modes[q] == "soft" else peer_pds[q]
                for q in range(len(models))
            ]
            ascent_worst.append(float(np.min(ascent_values)))
            ascent_used.append(int(ascent_bits))
            peer_to_soft_switches.append(float(sum(
                g48_modes[q] == "peer" and ascent_modes[q] == "soft"
                for q in range(len(models))
            )))
        full_mean = float(np.mean(full_values))
        soft_mean = float(np.mean(soft_worst))
        peer_mean = float(np.mean(peer_worst))
        global_mean = float(np.mean(global_worst))
        target_mean = float(np.mean(target_wise_worst))
        ascent_mean = float(np.mean(ascent_worst))
        summary.append({
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "soft_worst_pd": soft_mean,
            "peer_worst_pd": peer_mean,
            "global_switch_worst_pd": global_mean,
            "target_wise_switch_worst_pd": target_mean,
            "mode_ascent_worst_pd": ascent_mean,
            "mode_ascent_gain_vs_soft": ascent_mean - soft_mean,
            "mode_ascent_gain_vs_target_wise": ascent_mean - target_mean,
            "soft_rho_exact": effective_deflection(
                soft_mean, false_alarm_rate
            ) / max(full_mean, 1e-12),
            "mode_ascent_rho_exact": effective_deflection(
                ascent_mean, false_alarm_rate
            ) / max(full_mean, 1e-12),
            "mode_ascent_used_bits": float(np.mean(ascent_used)),
            "peer_to_soft_switches": float(np.mean(peer_to_soft_switches)),
            "soft_qos_feasible": soft_mean >= qos_target - 1e-9,
            "mode_ascent_qos_feasible": ascent_mean >= qos_target - 1e-9,
        })

    payload = {
        "gate": "G50-two-sided-mode-ascent",
        "seeds": seeds,
        "qos_target": qos_target,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/mode_ascent_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[8, 12, 16, 20, 28, 40])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
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
