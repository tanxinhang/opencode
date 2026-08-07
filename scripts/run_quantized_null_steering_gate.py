"""G37 gate: directly quantized null-steering phases.

G36 optimized continuous phases and quantized them at evaluation.  This gate
additionally runs discrete coordinate ascent directly on the quantized phase
levels, so the nulling design respects the real control overhead.
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
from uav_otfs_isac.ris_null_steering import (
    array_power,
    optimize_null_steering_phases,
    quantized_null_steering_phases,
    reflected_interference_inr,
)
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    quantize_phase,
    ris_control_overhead_bits,
)
from uav_otfs_isac.ris_upd import (
    upd_ideal_phase,
    upd_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def direct_inr_profile(transmitter_positions, sources, source_refs):
    total = np.zeros(transmitter_positions.shape[0], dtype=float)
    for position, inr_ref in zip(sources, source_refs):
        distances = np.linalg.norm(
            transmitter_positions - np.asarray(position, dtype=float),
            axis=1,
        )
        total += inr_ref * (100.0 / np.maximum(distances, 1e-9)) ** 2
    return total


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, qos_target: float,
    ris_elements: int, aperture_scale: float, phase_bits: int,
    coherence_frames: int, direct_blockage: float, sources, source_refs,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    upa = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        aperture_shape=(16, 16),
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(
        upa, coherence_frames=coherence_frames
    )
    direct_inr = direct_inr_profile(
        transmitter_positions, sources, source_refs
    )
    aligned_phases = [
        quantize_phase(upd_ideal_phase(upa, target), phase_bits)
        for target in targets
    ]
    continuous_phases = [
        quantize_phase(
            optimize_null_steering_phases(upa, target, sources, lambda_=1.0),
            phase_bits,
        )
        for target in targets
    ]
    quantized_phases = [
        quantized_null_steering_phases(
            upa, target, sources, lambda_=1.0
        )
        for target in targets
    ]
    phase_sets = {
        "aligned_quantized": aligned_phases,
        "continuous_quantized": continuous_phases,
        "quantized_optimized": quantized_phases,
    }
    interference_info = {}
    gain_sets = {}
    reflected_sets = {}
    for name, phases in phase_sets.items():
        reflected = np.mean([
            reflected_interference_inr(
                upa, phases[q], sources, transmitter_positions, inr_ref=1.0
            )
            for q in range(cfg.num_targets)
        ], axis=0)
        reflected_sets[name] = reflected
        gain_sets[name] = upd_physics_gain_matrix(
            upa, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase_per_target=phases,
        )
        interference_info[name] = {
            "mean_reflected_inr": float(np.mean(reflected)),
            "mean_target_gain": float(np.mean([
                array_power(phases[q], upd_ideal_phase(upa, targets[q]))
                for q in range(cfg.num_targets)
            ])),
        }
    summary = []
    for total_budget in budgets:
        report_budget = int(total_budget - overhead)
        methods = {"no_ris": [], **{name: [] for name in phase_sets}}
        for seed in seed_list:
            no_ris_models = build_models(
                cfg, np.random.default_rng(seed),
                interference_to_noise=direct_inr,
            )
            no_ris_selection = expected_pd_greedy_select(
                no_ris_models, report_budget, false_alarm_rate,
                qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
            )
            methods["no_ris"].append(float(np.min(no_ris_selection.expected_pd)))
            for name in phase_sets:
                models = build_models(
                    cfg, np.random.default_rng(seed),
                    snr_gain=gain_sets[name],
                    interference_to_noise=direct_inr + reflected_sets[name],
                )
                selection = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                methods[name].append(float(np.min(selection.expected_pd)))
        cell = {
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
        "gate": "G37-quantized-null-steering",
        "seeds": seeds,
        "interference_info": interference_info,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "interference_info": interference_info,
        "summary": summary,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/quantized_null_steering_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[28, 40])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    parser.add_argument("--sources", type=float, nargs="+",
                        default=[60.0, -20.0, 0.0, -30.0, 40.0, 0.0, 80.0, 20.0, 0.0])
    parser.add_argument("--source-refs", type=float, nargs="+",
                        default=[0.1, 0.2, 0.05])
    args = parser.parse_args()
    if len(args.sources) % 3 != 0 or len(args.sources) // 3 != len(args.source_refs):
        parser.error("sources must be triples and match source_refs")
    sources = [
        tuple(args.sources[index:index + 3])
        for index in range(0, len(args.sources), 3)
    ]
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
        sources=sources,
        source_refs=args.source_refs,
    )


if __name__ == "__main__":
    main()
