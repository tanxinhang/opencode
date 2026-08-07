"""G32 gate: interference/clutter sensitivity.

The moment-matched model is extended with a per-UAV interference-to-noise
ratio.  Effective SINR becomes ``SNR / (1 + INR)``.  The gate sweeps INR and
compares no-RIS centralized soft fusion, RIS ideal phase, and peer majority.
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
from uav_otfs_isac.sota_baselines import peer_majority_fusion


def run_gate(
    *, output: Path, seeds: int, inr_db_options, grid: int,
    qos_target: float, total_budget: int, ris_elements: int,
    aperture_scale: float, phase_bits: int, coherence_frames: int,
    direct_blockage: float,
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
    report_budget = int(total_budget - overhead)
    phases = [ris_beam_phase(target, ris) for target in targets]
    gain = ris_physics_gain_matrix(
        ris, transmitter_positions, targets, receiver, aperture_scale,
        direct_blockage=direct_blockage, phase_per_target=phases,
    )
    summary = []
    for inr_db in inr_db_options:
        inr = float(10.0 ** (inr_db / 10.0))
        methods = {"no_ris": [], "ris_ideal": [], "peer_majority": []}
        for seed in seed_list:
            no_ris_models = build_models(
                cfg, np.random.default_rng(seed),
                interference_to_noise=np.full(cfg.num_uavs, inr),
            )
            no_ris_selection = expected_pd_greedy_select(
                no_ris_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["no_ris"].append(float(np.min(no_ris_selection.expected_pd)))
            ris_models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                interference_to_noise=np.full(cfg.num_uavs, inr),
            )
            ris_selection = expected_pd_greedy_select(
                ris_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["ris_ideal"].append(float(np.min(ris_selection.expected_pd)))
            peer_values = [
                float(peer_majority_fusion(model, false_alarm_rate)["pd"])
                for model in ris_models
            ]
            methods["peer_majority"].append(float(np.min(peer_values)))
        cell = {
            "inr_db": inr_db,
            "inr_linear": inr,
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
        "gate": "G32-interference-sensitivity",
        "seeds": seeds,
        "qos_target": qos_target,
        "total_budget_bits": total_budget,
        "report_budget_bits": report_budget,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/interference_sensitivity_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--inr-db", type=float, nargs="+", default=[0, 3, 10, 20])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        inr_db_options=args.inr_db,
        grid=args.grid,
        qos_target=args.qos_target,
        total_budget=args.total_budget,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
