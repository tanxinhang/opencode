"""G17 gate: bounded multi-block transfer certificate.

The G16 certificate covers single-element moves.  This gate searches the
exact system objective over all zero-sum reallocations moving at most three
elements in total, iterates to the best allocation in that bounded
multi-block neighborhood, and certifies the final point with
``bounded_multi_move_certificate``.
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


def run_gate(
    *, output: Path, seeds: int, configurations, grid: int,
    qos_target: float, aperture_scale: float, direct_blockage: float,
    g16_result: Path,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris_position = np.array([0.0, 30.0, 6.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    with g16_result.open(encoding="utf-8") as handle:
        g16 = json.load(handle)
    upstream_seeds = g16.get("seeds")
    if upstream_seeds is not None and int(upstream_seeds) != seeds:
        raise ValueError(
            f"G17 seeds ({seeds}) must match upstream G16 seeds "
            f"({upstream_seeds})"
        )
    summary = []
    for config in configurations:
        num_elements, phase_bits, coherence_frames, total_budget = config
        overhead = num_elements * phase_bits / coherence_frames
        report_budget = int(total_budget - overhead)
        ris = RisConfig(
            position=ris_position,
            num_elements=num_elements,
            weak_target_id=cfg.num_targets - 1,
            phase_bits=phase_bits,
        )
        g16_cell = next(
            item for item in g16["summary"]
            if item["num_elements"] == num_elements
            and item["phase_bits"] == phase_bits
            and item["coherence_frames"] == coherence_frames
            and item["total_budget_bits"] == total_budget
        )
        allocation = tuple(g16_cell["refined_allocation"])

        def system_objective(candidate: tuple[int, ...]) -> float:
            phase = multi_beam_phase(ris, targets, candidate)
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

        rounds = 0
        while rounds < 20:
            certificate = bounded_multi_move_certificate(
                system_objective, allocation, max_transfer=3
            )
            if not certificate["improved"]:
                break
            allocation = certificate["allocation"]
            rounds += 1
        final = bounded_multi_move_certificate(
            system_objective, allocation, max_transfer=3
        )
        summary.append({
            "total_budget_bits": total_budget,
            "num_elements": num_elements,
            "phase_bits": phase_bits,
            "coherence_frames": coherence_frames,
            "report_budget_bits": report_budget,
            "g16_allocation": g16_cell["refined_allocation"],
            "g16_value": g16_cell["refined_value"],
            "multi_block_allocation": list(allocation),
            "multi_block_value": final["value"],
            "rounds": rounds,
            "certificate": {
                "local_optimal": final["local_optimal"],
                "evaluated": final["evaluated"],
                "max_transfer": final["max_transfer"],
            },
        })

    payload = {
        "gate": "G17-bounded-multi-block-certificate",
        "seeds": seeds,
        "qos_target": qos_target,
        "aperture_scale": aperture_scale,
        "direct_blockage": direct_blockage,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/multi_move_certificate_gate.json")
    parser.add_argument("--g16-result", type=Path,
                        default="results/single_move_certificate_gate.json")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument(
        "--configurations",
        nargs="+",
        type=lambda value: tuple(int(part) for part in value.split(",")),
        default=[
            (1024, 1, 64, 20),
            (704, 3, 128, 20),
            (1344, 3, 256, 20),
            (960, 3, 128, 28),
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
        g16_result=args.g16_result,
        seeds=args.seeds,
        configurations=args.configurations,
        grid=args.grid,
        qos_target=args.qos_target,
        aperture_scale=args.aperture_scale,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
