"""G35 gate: 1-D ULA versus 2-D UPA RIS.

Both apertures have 256 elements and the same control overhead.  The gate
compares no-RIS, 1-D ULA RIS, and 2-D UPA RIS under clean and spatial
interference scenarios.
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
from uav_otfs_isac.ris_upd import (
    upd_ideal_phase,
    upd_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def inr_profile(transmitter_positions, inr_ref=0.1):
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
    ula = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    upa = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        aperture_shape=(16, 16),
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(
        ula, coherence_frames=coherence_frames
    )
    ula_phases = [ris_beam_phase(target, ula) for target in targets]
    ula_gain = ris_physics_gain_matrix(
        ula, transmitter_positions, targets, receiver, aperture_scale,
        direct_blockage=direct_blockage, phase_per_target=ula_phases,
    )
    upa_phases = [upd_ideal_phase(upa, target) for target in targets]
    upa_gain = upd_physics_gain_matrix(
        upa, transmitter_positions, targets, receiver, aperture_scale,
        direct_blockage=direct_blockage, phase_per_target=upa_phases,
    )
    summary = []
    for scenario, inr in (
        ("clean", None),
        ("interference", inr_profile(transmitter_positions)),
    ):
        for total_budget in budgets:
            report_budget = int(total_budget - overhead)
            methods = {"no_ris": [], "ula": [], "upa": []}
            for seed in seed_list:
                no_ris_models = build_models(
                    cfg, np.random.default_rng(seed),
                    interference_to_noise=inr,
                )
                no_ris_selection = expected_pd_greedy_select(
                    no_ris_models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                methods["no_ris"].append(
                    float(np.min(no_ris_selection.expected_pd))
                )
                for name, gain in (("ula", ula_gain), ("upa", upa_gain)):
                    models = build_models(
                        cfg, np.random.default_rng(seed), snr_gain=gain,
                        interference_to_noise=inr,
                    )
                    selection = expected_pd_greedy_select(
                        models, report_budget, false_alarm_rate,
                        qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                    )
                    methods[name].append(float(np.min(selection.expected_pd)))
            cell = {
                "scenario": scenario,
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
        "gate": "G35-ula-vs-upa",
        "seeds": seeds,
        "qos_target": qos_target,
        "ris_elements": ris_elements,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/upd_vs_ula_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 28, 40])
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
