"""G18 gate: joint RIS placement and subarray allocation local optimization.

The system objective is the exact greedy-aware expected P_D

``F(s, a) = mean_seed min_q E_PD(q, S_q(s, a))``,

where ``s`` is the RIS position and ``a`` is the aperture allocation.
Alternating coordinate ascent optimizes allocation with the bounded
multi-block certificate (T<=3) and position with 2/1/0.5-meter coordinate
steps.  The final point is certified locally for both degrees of freedom.
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
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.ris_scenario import RisConfig
from uav_otfs_isac.ris_subarray import (
    bounded_multi_move_certificate,
    multi_beam_phase,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


DEFAULT_BOUNDS = np.array([
    [-20.0, 20.0],
    [15.0, 45.0],
    [0.0, 12.0],
])


def run_gate(
    *, output: Path, seeds: int, configurations, grid: int,
    qos_target: float, aperture_scale: float, direct_blockage: float,
    g17_result: Path, bounds: np.ndarray,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    with g17_result.open(encoding="utf-8") as handle:
        g17 = json.load(handle)
    upstream_seeds = g17.get("seeds")
    if upstream_seeds is not None and int(upstream_seeds) != seeds:
        raise ValueError(
            f"G18 seeds ({seeds}) must match upstream G17 seeds "
            f"({upstream_seeds})"
        )
    summary = []
    for config in configurations:
        num_elements, phase_bits, coherence_frames, total_budget = config
        overhead = num_elements * phase_bits / coherence_frames
        report_budget = int(total_budget - overhead)
        g17_cell = next(
            item for item in g17["summary"]
            if item["num_elements"] == num_elements
            and item["phase_bits"] == phase_bits
            and item["coherence_frames"] == coherence_frames
            and item["total_budget_bits"] == total_budget
        )
        position = np.array([0.0, 30.0, 6.0])
        allocation = tuple(g17_cell["multi_block_allocation"])

        def system_objective(
            candidate_position,
            candidate_allocation: tuple[int, ...],
        ) -> float:
            ris = RisConfig(
                position=np.asarray(candidate_position, dtype=float),
                num_elements=num_elements,
                weak_target_id=cfg.num_targets - 1,
                phase_bits=phase_bits,
            )
            phase = multi_beam_phase(ris, targets, candidate_allocation)
            gain = shared_phase_gain_matrix(
                ris, transmitter_positions, targets, receiver,
                aperture_scale,
                direct_blockage=direct_blockage, phase=phase,
            )
            worsts = []
            for seed in seed_list:
                models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain
                )
                selection = expected_pd_greedy_select(
                    models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                worsts.append(float(np.min(selection.expected_pd)))
            return float(np.mean(worsts))

        def allocation_objective(candidate_allocation):
            return system_objective(position, candidate_allocation)

        def position_objective(candidate_position):
            return system_objective(candidate_position, allocation)

        initial_value = system_objective(position, allocation)
        rounds = 0
        while rounds < 6:
            allocation_changed = False
            position_changed = False
            while True:
                certificate = bounded_multi_move_certificate(
                    allocation_objective, allocation, max_transfer=3
                )
                if not certificate["improved"]:
                    break
                allocation = certificate["allocation"]
                allocation_changed = True
            for step in (2.0, 1.0, 0.5):
                improved = True
                while improved:
                    improved = False
                    for axis in range(3):
                        best_value = system_objective(position, allocation)
                        best_position = position.copy()
                        for sign in (-1.0, 1.0):
                            candidate = position.copy()
                            candidate[axis] += sign * step
                            if (candidate[axis] < bounds[axis, 0]
                                    or candidate[axis] > bounds[axis, 1]):
                                continue
                            value = system_objective(candidate, allocation)
                            if value > best_value + 1e-12:
                                best_value = value
                                best_position = candidate
                        if not np.allclose(best_position, position):
                            position = best_position
                            improved = True
                            position_changed = True
            rounds += 1
            if not allocation_changed and not position_changed:
                break

        final_allocation_cert = bounded_multi_move_certificate(
            allocation_objective, allocation, max_transfer=3
        )
        position_gradients = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                candidate = position.copy()
                candidate[axis] += sign * 0.5
                if (candidate[axis] < bounds[axis, 0]
                        or candidate[axis] > bounds[axis, 1]):
                    continue
                value = system_objective(candidate, allocation)
                position_gradients.append(value - system_objective(
                    position, allocation
                ))
        summary.append({
            "total_budget_bits": total_budget,
            "num_elements": num_elements,
            "phase_bits": phase_bits,
            "coherence_frames": coherence_frames,
            "report_budget_bits": report_budget,
            "g17_value": g17_cell["multi_block_value"],
            "g17_allocation": g17_cell["multi_block_allocation"],
            "initial_position": [0.0, 30.0, 6.0],
            "final_position": position.tolist(),
            "final_allocation": list(allocation),
            "final_value": system_objective(position, allocation),
            "rounds": rounds,
            "allocation_certificate": {
                "local_optimal": final_allocation_cert["local_optimal"],
                "max_transfer": final_allocation_cert["max_transfer"],
            },
            "position_certificate": {
                "local_optimal": (
                    max(position_gradients, default=0.0) <= 1e-9
                ),
                "maximum_gradient": max(position_gradients, default=0.0),
                "step_meters": 0.5,
            },
        })

    payload = {
        "gate": "G18-joint-placement-allocation",
        "seeds": seeds,
        "qos_target": qos_target,
        "aperture_scale": aperture_scale,
        "direct_blockage": direct_blockage,
        "bounds": bounds.tolist(),
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/joint_placement_allocation_gate.json")
    parser.add_argument("--g17-result", type=Path,
                        default="results/multi_move_certificate_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument(
        "--configurations",
        nargs="+",
        type=lambda value: tuple(int(part) for part in value.split(",")),
        default=[
            (1024, 1, 64, 20),
            (1344, 3, 256, 20),
            (2048, 3, 256, 40),
        ],
    )
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        g17_result=args.g17_result,
        seeds=args.seeds,
        configurations=args.configurations,
        grid=args.grid,
        qos_target=args.qos_target,
        aperture_scale=args.aperture_scale,
        direct_blockage=args.direct_blockage,
        bounds=DEFAULT_BOUNDS,
    )


if __name__ == "__main__":
    main()
