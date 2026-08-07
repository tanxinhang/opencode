"""G38 gate: joint quantized null-steering and RIS placement.

For every candidate RIS position, the UPA quantized null-steering phases are
redesigned for all targets.  The exact system objective is evaluated with
those phases, and position coordinate ascent finds the joint local optimum.
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
    quantized_null_steering_phases,
    reflected_interference_inr,
)
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_control_overhead_bits,
)
from uav_otfs_isac.ris_upd import upd_physics_gain_matrix
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


BOUNDS = np.array([
    [-20.0, 20.0],
    [10.0, 50.0],
    [0.0, 12.0],
])


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
    *, output: Path, seeds: int, grid: int, qos_target: float,
    total_budget: int, ris_elements: int, aperture_scale: float,
    phase_bits: int, coherence_frames: int, direct_blockage: float,
    sources, source_refs,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    direct_inr = direct_inr_profile(
        transmitter_positions, sources, source_refs
    )
    reference_ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        aperture_shape=(16, 16),
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(
        reference_ris, coherence_frames=coherence_frames
    )
    report_budget = int(total_budget - overhead)

    def evaluate(position):
        ris = RisConfig(
            position=np.asarray(position, dtype=float),
            num_elements=ris_elements,
            aperture_shape=(16, 16),
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        phases = [
            quantized_null_steering_phases(
                ris, target, sources, lambda_=1.0
            )
            for target in targets
        ]
        gain = upd_physics_gain_matrix(
            ris, transmitter_positions, targets, receiver, aperture_scale,
            direct_blockage=direct_blockage, phase_per_target=phases,
        )
        reflected = np.mean([
            reflected_interference_inr(
                ris, phases[q], sources, transmitter_positions, inr_ref=1.0
            )
            for q in range(cfg.num_targets)
        ], axis=0)
        worsts = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                interference_to_noise=direct_inr + reflected,
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            worsts.append(float(np.min(selection.expected_pd)))
        return float(np.mean(worsts)), float(np.mean(reflected)), phases

    no_ris_worst = []
    for seed in seed_list:
        models = build_models(
            cfg, np.random.default_rng(seed),
            interference_to_noise=direct_inr,
        )
        selection = expected_pd_greedy_select(
            models, report_budget, false_alarm_rate, qos_pd=qos_pd,
            qos_weights=qos_weights, grid=grid,
        )
        no_ris_worst.append(float(np.min(selection.expected_pd)))
    fixed_value, fixed_reflected, _ = evaluate(np.array([0.0, 30.0, 6.0]))
    best_position = np.array([0.0, 30.0, 6.0])
    best_value = fixed_value
    best_reflected = fixed_reflected
    for step in (2.0, 1.0, 0.5):
        improved = True
        while improved:
            improved = False
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = best_position.copy()
                    candidate[axis] += sign * step
                    if (candidate[axis] < BOUNDS[axis, 0]
                            or candidate[axis] > BOUNDS[axis, 1]):
                        continue
                    value, reflected, _ = evaluate(candidate)
                    if value > best_value + 1e-12:
                        best_value = value
                        best_reflected = reflected
                        best_position = candidate
                        improved = True

    summary = {
        "no_ris_worst": float(np.mean(no_ris_worst)),
        "no_ris_qos": float(np.mean(no_ris_worst) >= qos_target - 1e-9),
        "fixed_position": [0.0, 30.0, 6.0],
        "fixed_value": fixed_value,
        "fixed_reflected_inr": fixed_reflected,
        "fixed_qos": float(fixed_value >= qos_target - 1e-9),
        "optimized_position": best_position.tolist(),
        "optimized_value": best_value,
        "optimized_reflected_inr": best_reflected,
        "optimized_qos": float(best_value >= qos_target - 1e-9),
    }
    payload = {
        "gate": "G38-joint-null-placement",
        "seeds": seeds,
        "report_budget_bits": report_budget,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_null_placement_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--total-budget", type=int, default=40)
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
        grid=args.grid,
        qos_target=args.qos_target,
        total_budget=args.total_budget,
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
