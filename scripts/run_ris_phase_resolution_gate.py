"""G5-Q gate: finite-resolution RIS phase quantization and control overhead.

The G5 channel audit used ideal continuous RIS phases.  This gate quantizes
the phase profile to 1/2/3 bits, compares the theoretical array-gain loss
``sinc^2(1/2^b)`` with simulated gains, and reports a control-plane overhead
ledger (``N * phase_bits / coherence_frames`` bits per frame) so the 6G
scenario does not silently assume free RIS control.
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
    ris_gain_matrix,
    ris_quantized_gain_loss,
)
from uav_otfs_isac.scenario import build_models, target_geometry


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, ris_elements: int,
    coherence_frames: int,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    phase_bits_options = (None, 1, 2, 3)
    rows = []
    for budget in budgets:
        for offset in range(seeds):
            seed = cfg.seed + offset
            model_rng = np.random.default_rng(seed)
            no_ris_models = build_models(cfg, model_rng)
            no_ris_selection = expected_pd_greedy_select(
                no_ris_models, budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            no_ris_vector = np.asarray(no_ris_selection.expected_pd)
            for phase_bits in phase_bits_options:
                ris = RisConfig(
                    position=np.array([55.0, 15.0, 12.0]),
                    num_elements=ris_elements,
                    weak_target_id=cfg.num_targets - 1,
                    phase_bits=phase_bits,
                )
                phases = [ris_beam_phase(target, ris) for target in targets]
                gain = ris_gain_matrix(ris, targets, cfg.num_uavs, phases)
                ris_models = build_models(
                    cfg, np.random.default_rng(seed), snr_gain=gain
                )
                selection = expected_pd_greedy_select(
                    ris_models, budget, false_alarm_rate, qos_pd=qos_pd,
                    qos_weights=qos_weights, grid=grid,
                )
                vector = np.asarray(selection.expected_pd)
                rows.append({
                    "budget_bits": budget,
                    "seed_offset": offset,
                    "phase_bits": phase_bits,
                    "mean_expected_pd": float(np.mean(vector)),
                    "worst_expected_pd": float(np.min(vector)),
                    "no_ris_mean": float(np.mean(no_ris_vector)),
                    "no_ris_worst": float(np.min(no_ris_vector)),
                })

    summary = []
    for budget in budgets:
        group = [row for row in rows if row["budget_bits"] == budget]
        entry = {"budget_bits": budget}
        for phase_bits in phase_bits_options:
            subset = [row for row in group if row["phase_bits"] == phase_bits]
            entry[("continuous" if phase_bits is None else f"bits_{phase_bits}")] = {
                "mean_expected_pd": float(np.mean([
                    row["mean_expected_pd"] for row in subset
                ])),
                "worst_expected_pd": float(np.mean([
                    row["worst_expected_pd"] for row in subset
                ])),
                "mean_gain_vs_no_ris": float(np.mean([
                    row["mean_expected_pd"] - row["no_ris_mean"]
                    for row in subset
                ])),
                "worst_gain_vs_no_ris": float(np.mean([
                    row["worst_expected_pd"] - row["no_ris_worst"]
                    for row in subset
                ])),
                "theoretical_array_gain_loss": (
                    ris_quantized_gain_loss(phase_bits)
                ),
                "control_overhead_bits_per_frame": (
                    ris_control_overhead_bits(
                        RisConfig(
                            position=np.array([55.0, 15.0, 12.0]),
                            num_elements=ris_elements,
                            phase_bits=phase_bits,
                        ),
                        coherence_frames=coherence_frames,
                    )
                ),
            }
        summary.append(entry)

    payload = {
        "gate": "G5-Q-ris-phase-resolution",
        "ris_elements": ris_elements,
        "coherence_frames": coherence_frames,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/ris_phase_resolution_gate.json"
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--ris-elements", type=int, default=16)
    parser.add_argument("--coherence-frames", type=int, default=100)
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        grid=args.grid,
        ris_elements=args.ris_elements,
        coherence_frames=args.coherence_frames,
    )


if __name__ == "__main__":
    main()
