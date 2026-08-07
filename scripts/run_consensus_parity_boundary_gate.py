"""G41 gate: consensus-versus-centralized parity boundary.

For i.i.d. local 1-bit decisions with probabilities ``p0, p1``, the majority
rule satisfies ``P_D >= beta`` and ``P_FA <= alpha`` when, by the Gaussian
approximation,

``M >= M_min = p1 (1 - p1) (z_alpha + z_beta)^2 / (p1 - p0)^2``.

This gate sweeps UAV count and report budget, evaluates the exact system, and
records where consensus outperforms centralized soft fusion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

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
    hard_decision_local_probabilities,
    peer_majority_fusion,
)


def spaced_owners(num_uavs: int, num_targets: int) -> tuple[int, ...]:
    if num_targets == 1:
        return (0,)
    return tuple(
        int(round(index * (num_uavs - 1) / (num_targets - 1)))
        for index in range(num_targets)
    )


def inr_profile(transmitter_positions, inr_ref=0.5):
    source = np.array([60.0, -20.0, 0.0])
    distances = np.linalg.norm(
        transmitter_positions - source, axis=1
    )
    return inr_ref * (100.0 / np.maximum(distances, 1e-9)) ** 2


def theoretical_min_uavs(p0: float, p1: float, alpha: float, beta: float) -> float:
    if p1 <= p0:
        return np.inf
    numerator = p1 * (1.0 - p1) * (
        norm.ppf(1.0 - alpha) + norm.ppf(beta)
    ) ** 2
    denominator = (p1 - p0) ** 2
    return float(numerator / denominator)


def run_gate(
    *, output: Path, seeds: int, budgets, uav_counts, grid: int,
    qos_target: float, ris_elements: int, aperture_scale: float,
    phase_bits: int, coherence_frames: int, direct_blockage: float,
) -> None:
    base = load_config("config/demo.yaml")
    false_alarm_rate = base.false_alarm_rate
    alpha = false_alarm_rate
    beta = qos_target
    summary = []
    for num_uavs in uav_counts:
        owners = spaced_owners(num_uavs, base.num_targets)
        cfg = replace(
            base,
            num_uavs=num_uavs,
            owners=owners,
            target_present=tuple([True] * base.num_targets),
            qos_min_deflection=tuple([3.0] * base.num_targets),
            qos_weights=tuple([1.0] * base.num_targets),
            performance_weights=tuple([1.0] * base.num_targets),
        )
        cfg.validate()
        qos_pd = np.full(base.num_targets, qos_target)
        qos_weights = np.ones(base.num_targets)
        transmitter_positions = uav_geometry(num_uavs)
        targets = [target_geometry(q) for q in range(base.num_targets)]
        receiver = np.array([0.0, 0.0, 0.0])
        seed_list = [cfg.seed + offset for offset in range(seeds)]
        inr = inr_profile(transmitter_positions)
        ris = RisConfig(
            position=np.array([0.0, 30.0, 6.0]),
            num_elements=ris_elements,
            weak_target_id=base.num_targets - 1,
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
        reference_models = build_models(
            cfg, np.random.default_rng(cfg.seed), snr_gain=gain,
            interference_to_noise=inr,
        )
        p_values = [
            hard_decision_local_probabilities(model, uav, 0.1)
            for model in reference_models
            for uav in range(num_uavs)
            if uav != model.owner
        ]
        p0_mean = float(np.mean([value[0] for value in p_values]))
        p1_mean = float(np.mean([value[1] for value in p_values]))
        m_theory = theoretical_min_uavs(p0_mean, p1_mean, alpha, beta)
        for total_budget in budgets:
            report_budget = int(total_budget - overhead)
            if report_budget < 0:
                continue
            central_values = []
            peer_values = []
            for seed in seed_list:
                models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain,
                    interference_to_noise=inr,
                )
                central = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                central_values.append(float(np.min(central.expected_pd)))
                peer_values.append(float(np.min([
                    float(peer_majority_fusion(model, false_alarm_rate)["pd"])
                    for model in models
                ])))
            central_worst = float(np.mean(central_values))
            peer_worst = float(np.mean(peer_values))
            summary.append({
                "num_uavs": num_uavs,
                "total_budget_bits": total_budget,
                "report_budget_bits": report_budget,
                "p0_mean": p0_mean,
                "p1_mean": p1_mean,
                "theoretical_min_uavs": m_theory,
                "centralized_worst": central_worst,
                "peer_worst": peer_worst,
                "consensus_wins": peer_worst > central_worst + 1e-9,
            })

    payload = {
        "gate": "G41-consensus-parity-boundary",
        "seeds": seeds,
        "qos_target": qos_target,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/consensus_parity_boundary_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--uav-counts", type=int, nargs="+",
                        default=[3, 6, 8, 12, 16])
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
        uav_counts=args.uav_counts,
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
