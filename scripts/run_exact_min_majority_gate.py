"""G43-B gate: exact minimum majority UAV count and monotonicity audit.

G43 evaluates exact Poisson-binomial feasibility on a discrete UAV-count
grid.  This gate computes the exact minimum prefix size for a voter sequence
and explicitly audits whether feasibility is monotone in the voter count.
Monotonicity is required for a binary search of `M_min`; the audit reports
whether the audited sequence satisfies it.
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

from uav_otfs_isac.config import load_config
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry
from uav_otfs_isac.sota_baselines import (
    exact_min_majority_uavs,
    hard_decision_local_probabilities,
    majority_feasibility_trace,
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
    distances = np.linalg.norm(transmitter_positions - source, axis=1)
    return inr_ref * (100.0 / np.maximum(distances, 1e-9)) ** 2


def run_gate(
    *, output: Path, uav_counts, qos_target: float, ris_elements: int,
    aperture_scale: float, phase_bits: int, coherence_frames: int,
    direct_blockage: float,
) -> None:
    base = load_config("config/demo.yaml")
    false_alarm_rate = base.false_alarm_rate
    alpha_grid = tuple(
        float(value) for value in np.geomspace(0.005, 0.5, 20)
    ) + (0.1,)
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
        transmitter_positions = uav_geometry(num_uavs)
        targets = [target_geometry(q) for q in range(base.num_targets)]
        receiver = np.array([0.0, 0.0, 0.0])
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

        best_alpha = None
        best_min = None
        best_trace = None
        for alpha in alpha_grid:
            pairs = [
                hard_decision_local_probabilities(model, uav, alpha)
                for model in reference_models
                for uav in range(num_uavs)
                if uav != model.owner
            ]
            p0 = np.array([pair[0] for pair in pairs])
            p1 = np.array([pair[1] for pair in pairs])
            exact_min = exact_min_majority_uavs(
                p0, p1, false_alarm_rate, qos_target
            )
            trace = majority_feasibility_trace(
                p0, p1, false_alarm_rate, qos_target
            )
            if exact_min is not None and (
                best_min is None or exact_min < best_min
            ):
                best_min = exact_min
                best_alpha = alpha
                best_trace = trace

        monotone = (
            best_trace is None
            or all(
                not best_trace[index] or best_trace[index + 1]
                for index in range(len(best_trace) - 1)
            )
        )
        summary.append({
            "num_uavs": num_uavs,
            "max_voters": base.num_targets * (num_uavs - 1),
            "pooled_voter_groups": base.num_targets,
            "exact_min_uavs": best_min,
            "exact_min_voters": best_min,
            "best_local_alpha": best_alpha,
            "feasibility_monotone": monotone,
            "trace_prefix": (
                None if best_trace is None else best_trace[:16]
            ),
        })

    payload = {
        "gate": "G43-B-exact-min-majority",
        "qos_target": qos_target,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/exact_min_majority_gate.json")
    parser.add_argument("--uav-counts", type=int, nargs="+",
                        default=[3, 6, 8, 12, 16])
    parser.add_argument("--qos-target", type=float, default=0.70)
    parser.add_argument("--ris-elements", type=int, default=128)
    parser.add_argument("--aperture-scale", type=float, default=1e-2)
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--coherence-frames", type=int, default=64)
    parser.add_argument("--direct-blockage", type=float, default=0.01)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        uav_counts=args.uav_counts,
        qos_target=args.qos_target,
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
