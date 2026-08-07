"""G33 gate: spatial interference and RIS placement.

An interference source at a fixed position generates per-UAV INR through
free-space path loss:

``INR_i = inr_ref * (d_ref / d_i)^2``,

where ``d_i`` is the distance from the source to UAV ``i``.  The gate sweeps
``inr_ref`` and compares no-RIS, fixed RIS, and RIS position optimized on the
exact system objective.
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


BOUNDS = np.array([
    [-20.0, 20.0],
    [10.0, 50.0],
    [0.0, 12.0],
])


def inr_profile(
    interference_position,
    transmitter_positions,
    inr_ref: float,
    reference_distance: float = 100.0,
) -> np.ndarray:
    distances = np.linalg.norm(
        transmitter_positions - np.asarray(interference_position, dtype=float),
        axis=1,
    )
    return inr_ref * (reference_distance / np.maximum(distances, 1e-9)) ** 2


def run_gate(
    *, output: Path, seeds: int, inr_ref_options, grid: int,
    qos_target: float, total_budget: int, ris_elements: int,
    aperture_scale: float, phase_bits: int, coherence_frames: int,
    direct_blockage: float, interference_position,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    fixed_position = np.array([0.0, 30.0, 6.0])
    fixed_ris = RisConfig(
        position=fixed_position,
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(
        fixed_ris, coherence_frames=coherence_frames
    )
    report_budget = int(total_budget - overhead)
    summary = []
    for inr_ref in inr_ref_options:
        inr_vector = inr_profile(
            interference_position, transmitter_positions, inr_ref
        )

        def evaluate(position):
            ris = RisConfig(
                position=np.asarray(position, dtype=float),
                num_elements=ris_elements,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            )
            phases = [ris_beam_phase(target, ris) for target in targets]
            gain = ris_physics_gain_matrix(
                ris, transmitter_positions, targets, receiver, aperture_scale,
                direct_blockage=direct_blockage, phase_per_target=phases,
            )
            worsts = []
            for seed in seed_list:
                models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain,
                    interference_to_noise=inr_vector,
                )
                selection = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                    qos_weights=qos_weights, grid=grid,
                )
                worsts.append(float(np.min(selection.expected_pd)))
            return float(np.mean(worsts))

        no_ris_worst = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed),
                interference_to_noise=inr_vector,
            )
            selection = expected_pd_greedy_select(
                models, report_budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            no_ris_worst.append(float(np.min(selection.expected_pd)))
        fixed_value = evaluate(fixed_position)
        best_position = fixed_position.copy()
        best_value = fixed_value
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
                        value = evaluate(candidate)
                        if value > best_value + 1e-12:
                            best_value = value
                            best_position = candidate
                            improved = True
        summary.append({
            "inr_ref": inr_ref,
            "mean_inr": float(np.mean(inr_vector)),
            "no_ris_worst": float(np.mean(no_ris_worst)),
            "no_ris_qos": float(np.mean(no_ris_worst) >= qos_target - 1e-9),
            "fixed_ris_worst": fixed_value,
            "fixed_ris_qos": float(fixed_value >= qos_target - 1e-9),
            "optimized_position": best_position.tolist(),
            "optimized_ris_worst": best_value,
            "optimized_ris_qos": float(best_value >= qos_target - 1e-9),
        })

    payload = {
        "gate": "G33-spatial-interference-placement",
        "seeds": seeds,
        "interference_position": list(interference_position),
        "report_budget_bits": report_budget,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/spatial_interference_placement_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--inr-ref", type=float, nargs="+", default=[1e-2, 1e-1, 1.0])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    parser.add_argument("--interference-position", type=float, nargs=3,
                        default=[60.0, -20.0, 0.0])
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        inr_ref_options=args.inr_ref,
        grid=args.grid,
        qos_target=args.qos_target,
        total_budget=args.total_budget,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
        interference_position=args.interference_position,
    )


if __name__ == "__main__":
    main()
