"""G25 gate: scaled white-box G18 architecture in the scalability matrix.

The full G18 multi-block certificate is exponential in Q, so for Q>3 the
scaled architecture uses the derived max-min water-filling allocation and
exact position coordinate ascent, with the position certificate retained.
This gate compares it against no-RIS, per-target ideal phase, and peer
majority at Q in {2,4,6} and M/Q in {1,2,3}.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.architecture_objective import (
    aperture_constants,
    waterfilling_allocation,
)
from uav_otfs_isac.config import load_config
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.fusion import optimal_deflection
from uav_otfs_isac.ris_optimization import shared_phase_gain_matrix
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
    ris_quantized_gain_loss,
)
from uav_otfs_isac.ris_subarray import multi_beam_phase
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import peer_majority_fusion


BOUNDS = np.array([
    [-20.0, 20.0],
    [15.0, 45.0],
    [0.0, 12.0],
])


def spaced_owners(num_uavs: int, num_targets: int) -> tuple[int, ...]:
    if num_targets == 1:
        return (0,)
    return tuple(
        int(round(index * (num_uavs - 1) / (num_targets - 1)))
        for index in range(num_targets)
    )


def run_gate(
    *, output: Path, seeds: int, grid: int, qos_target: float,
    ris_elements: int, aperture_scale: float, phase_bits: int,
    coherence_frames: int, direct_blockage: float,
    target_counts, uav_ratios,
) -> None:
    base = load_config("config/demo.yaml")
    false_alarm_rate = base.false_alarm_rate
    summary = []
    for num_targets in target_counts:
        for ratio in uav_ratios:
            num_uavs = num_targets * ratio
            owners = spaced_owners(num_uavs, num_targets)
            report_budget = 20 * num_targets
            cfg = replace(
                base,
                num_uavs=num_uavs,
                num_targets=num_targets,
                owners=owners,
                target_present=tuple([True] * num_targets),
                qos_min_deflection=tuple([3.0] * num_targets),
                qos_weights=tuple([1.0] * num_targets),
                performance_weights=tuple([1.0] * num_targets),
                report_budget_bits=report_budget,
            )
            cfg.validate()
            qos_pd = np.full(num_targets, qos_target)
            qos_weights = np.ones(num_targets)
            transmitter_positions = uav_geometry(num_uavs)
            targets = [target_geometry(q) for q in range(num_targets)]
            receiver = np.array([0.0, 0.0, 0.0])
            seed_list = [cfg.seed + offset for offset in range(seeds)]
            reference_ris = RisConfig(
                position=np.array([0.0, 30.0, 6.0]),
                num_elements=ris_elements,
                weak_target_id=num_targets - 1,
                phase_bits=phase_bits,
            )
            overhead = ris_control_overhead_bits(
                reference_ris, coherence_frames=coherence_frames
            )
            total_budget = report_budget + int(overhead)
            constants = aperture_constants(
                reference_ris, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=direct_blockage,
            )
            base_models = [
                build_models(cfg, np.random.default_rng(seed))
                for seed in seed_list
            ]
            base_deflections = np.mean([
                [
                    optimal_deflection(
                        model.delta, model.sigma0, {model.owner}
                    )
                    for model in models
                ]
                for models in base_models
            ], axis=0)
            effective_constants = constants * ris_quantized_gain_loss(phase_bits)
            allocation = waterfilling_allocation(
                ris_elements, effective_constants, base_deflections
            )

            def evaluate(position, candidate_allocation):
                ris = RisConfig(
                    position=np.asarray(position, dtype=float),
                    num_elements=ris_elements,
                    weak_target_id=num_targets - 1,
                    phase_bits=phase_bits,
                )
                phase = multi_beam_phase(ris, targets, candidate_allocation)
                gain = shared_phase_gain_matrix(
                    ris, transmitter_positions, targets, receiver,
                    aperture_scale, direct_blockage=direct_blockage, phase=phase,
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

            position = np.array([0.0, 30.0, 6.0])
            best_value = evaluate(position, allocation)
            for step in (2.0, 1.0, 0.5):
                improved = True
                rounds = 0
                while improved and rounds < 2:
                    improved = False
                    for axis in range(3):
                        for sign in (-1.0, 1.0):
                            candidate = position.copy()
                            candidate[axis] += sign * step
                            if (candidate[axis] < BOUNDS[axis, 0]
                                    or candidate[axis] > BOUNDS[axis, 1]):
                                continue
                            value = evaluate(candidate, allocation)
                            if value > best_value + 1e-12:
                                best_value = value
                                position = candidate
                                improved = True
                    rounds += 1

            phases = [ris_beam_phase(target, reference_ris) for target in targets]
            ideal_gain = ris_physics_gain_matrix(
                reference_ris, transmitter_positions, targets, receiver,
                aperture_scale, direct_blockage=direct_blockage,
                phase_per_target=phases,
            )
            no_ris_worst = []
            ideal_worst = []
            scaled_worst = []
            peer_worst = []
            for seed in seed_list:
                no_ris_models = build_models(
                    cfg, np.random.default_rng(seed)
                )
                no_ris_selection = expected_pd_greedy_select(
                    no_ris_models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                no_ris_worst.append(float(np.min(no_ris_selection.expected_pd)))
                ideal_models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=ideal_gain
                )
                ideal_selection = expected_pd_greedy_select(
                    ideal_models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                ideal_worst.append(float(np.min(ideal_selection.expected_pd)))
                scaled_models = build_models(
                    cfg, np.random.default_rng(seed),
                    snr_gain=shared_phase_gain_matrix(
                        RisConfig(
                            position=position,
                            num_elements=ris_elements,
                            weak_target_id=num_targets - 1,
                            phase_bits=phase_bits,
                        ),
                        transmitter_positions, targets, receiver,
                        aperture_scale, direct_blockage=direct_blockage,
                        phase=multi_beam_phase(
                            RisConfig(
                                position=position,
                                num_elements=ris_elements,
                                weak_target_id=num_targets - 1,
                                phase_bits=phase_bits,
                            ),
                            targets, allocation,
                        ),
                    ),
                )
                scaled_selection = expected_pd_greedy_select(
                    scaled_models, report_budget, false_alarm_rate,
                    qos_pd=qos_pd, qos_weights=qos_weights, grid=grid,
                )
                scaled_worst.append(float(np.min(scaled_selection.expected_pd)))
                peer_worst.append(float(np.min([
                    float(peer_majority_fusion(model, false_alarm_rate)["pd"])
                    for model in scaled_models
                ])))
            summary.append({
                "num_targets": num_targets,
                "num_uavs": num_uavs,
                "uav_to_target_ratio": ratio,
                "report_budget_bits": report_budget,
                "total_budget_bits": total_budget,
                "allocation": list(allocation),
                "position": position.tolist(),
                "scaled_g18_value": best_value,
                "no_ris_worst": float(np.mean(no_ris_worst)),
                "ris_ideal_worst": float(np.mean(ideal_worst)),
                "scaled_g18_worst": float(np.mean(scaled_worst)),
                "peer_majority_worst": float(np.mean(peer_worst)),
                "scaled_g18_qos": float(np.mean([
                    value >= qos_target - 1e-9 for value in scaled_worst
                ])),
                "ris_ideal_qos": float(np.mean([
                    value >= qos_target - 1e-9 for value in ideal_worst
                ])),
                "peer_majority_qos": float(np.mean([
                    value >= qos_target - 1e-9 for value in peer_worst
                ])),
            })

    payload = {
        "gate": "G25-scaled-g18-scalability",
        "seeds": seeds,
        "qos_target": qos_target,
        "ris_elements": ris_elements,
        "aperture_scale": aperture_scale,
        "phase_bits": phase_bits,
        "coherence_frames": coherence_frames,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/scaled_g18_scalability_gate.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument("--ris-elements", type=int, default=256)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    parser.add_argument("--target-counts", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--uav-ratios", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        grid=args.grid,
        qos_target=args.qos_target,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
        target_counts=args.target_counts,
        uav_ratios=args.uav_ratios,
    )


if __name__ == "__main__":
    main()
