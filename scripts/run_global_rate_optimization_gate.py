"""G30 gate: global variable-rate profile optimization.

Each UAV uses one quantizer-bit rate for all target reports.  The objective
is the exact system function

``F(bits) = mean_seed min_q E_PD(q, S_q(bits))``,

where ``S_q(bits)`` is the expected-P_D greedy schedule under report costs
``bits + 2``.  Coordinate ascent changes one UAV's bit count at a time and
accepts only strict improvements; the final point is certified against every
single-rate change.
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

        def objective(bits_profile):
            bits_profile = tuple(int(value) for value in bits_profile)
            worsts = []
            for seed in seed_list:
                models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain,
                    quantizer_bits_per_uav=bits_profile,
                )
                selection = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                worsts.append(float(np.min(selection.expected_pd)))
            return float(np.mean(worsts))

        fixed3 = tuple([1] * cfg.num_uavs)
        fixed5 = tuple([3] * cfg.num_uavs)
        adaptive = tuple([
            3 if index < max(1, report_budget // cfg.num_targets) else 1
            for index in range(cfg.num_uavs)
        ])
        best_profile = list(fixed5)
        best_value = objective(best_profile)
        history = [{
            "bits": list(best_profile),
            "value": best_value,
        }]
        improved = True
        while improved:
            improved = False
            best_candidate = None
            best_candidate_value = best_value
            for uav in range(cfg.num_uavs):
                for new_bits in (1, 2, 3):
                    if new_bits == best_profile[uav]:
                        continue
                    trial = list(best_profile)
                    trial[uav] = new_bits
                    value = objective(trial)
                    if value > best_candidate_value + 1e-12:
                        best_candidate_value = value
                        best_candidate = trial
            if best_candidate is not None:
                best_profile = best_candidate
                best_value = best_candidate_value
                improved = True
                history.append({
                    "bits": list(best_profile),
                    "value": best_value,
                })

        certificate = []
        for uav in range(cfg.num_uavs):
            for new_bits in (1, 2, 3):
                if new_bits == best_profile[uav]:
                    continue
                trial = list(best_profile)
                trial[uav] = new_bits
                value = objective(trial)
                certificate.append({
                    "uav": uav,
                    "new_bits": new_bits,
                    "gradient": value - best_value,
                })
        summary.append({
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "fixed3": objective(fixed3),
            "fixed5": objective(fixed5),
            "adaptive": objective(adaptive),
            "optimized_bits": list(best_profile),
            "optimized_value": best_value,
            "optimized_qos": float(best_value >= qos_target - 1e-9),
            "single_change_local_optimal": bool(
                max(item["gradient"] for item in certificate) <= 1e-9
            ),
            "history": history,
        })

    payload = {
        "gate": "G30-global-rate-optimization",
        "seeds": seeds,
        "qos_target": qos_target,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/global_rate_optimization_gate.json")
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
