"""G5-P gate: physics-based RIS cascaded-channel audit.

This gate replaces the controlled additive-power RIS gain with a two-way
bistatic radar law for the direct path and a three-leg cascaded loss for the
RIS path:

``gain = 1 + N^2 array_gain^2 aperture_scale (R_tx R_rx)^2 /
         (R_txris^2 R_ristarget^2 R_targetrx^2)``

with an optional direct-path blockage for the weak target.  It sweeps the
number of RIS elements and aperture scale, compares aligned versus random
phase, and reports mean/worst expected P_D and QoS feasibility under the
expected-P_D greedy selector.
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
    blocked_direct_gain_matrix,
    ris_beam_phase,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, elements_options,
    aperture_scale_options,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, 0.85)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    ris_position = np.array([55.0, 15.0, 12.0])
    rows = []
    for budget in budgets:
        for offset in range(seeds):
            seed = cfg.seed + offset
            no_ris_models = build_models(
                cfg, np.random.default_rng(seed)
            )
            no_ris_selection = expected_pd_greedy_select(
                no_ris_models, budget, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            no_ris_vector = np.asarray(no_ris_selection.expected_pd)
            rng_phase = np.random.default_rng(seed + 300000)
            for elements in elements_options:
                for aperture_scale in aperture_scale_options:
                    ris = RisConfig(
                        position=ris_position,
                        num_elements=elements,
                        weak_target_id=cfg.num_targets - 1,
                    )
                    aligned_phases = [
                        ris_beam_phase(target, ris) for target in targets
                    ]
                    aligned_gain = ris_physics_gain_matrix(
                        ris, transmitter_positions, targets, receiver,
                        aperture_scale, direct_blockage=0.01,
                        phase_per_target=aligned_phases,
                    )
                    random_phases = [
                        rng_phase.uniform(0.0, 2.0 * np.pi, elements)
                        for _ in targets
                    ]
                    random_gain = ris_physics_gain_matrix(
                        ris, transmitter_positions, targets, receiver,
                        aperture_scale, direct_blockage=0.01,
                        phase_per_target=random_phases,
                    )
                    # P1-1 control (P4 matched-control factory, advice/011
                    # section 5): the honest weak-target baseline is
                    # "blocked, NO RIS" (the direct-path floor), NOT the
                    # clean no-RIS scenario; clean no-RIS is only the
                    # unblocked reference.  Comparing aligned to the clean
                    # no-RIS conflates the blockage itself with the RIS
                    # benefit (the pre-fix inflation masked this entirely).
                    blocked_gain = blocked_direct_gain_matrix(
                        cfg.num_targets, cfg.num_uavs,
                        weak_target_id=cfg.num_targets - 1,
                        direct_blockage=0.01,
                    )
                    scenarios = {
                        "ris_aligned": aligned_gain,
                        "ris_random": random_gain,
                        "blocked_no_ris": blocked_gain,
                    }
                    for name, gain in scenarios.items():
                        models = build_models(
                            cfg, np.random.default_rng(seed), snr_gain=gain
                        )
                        selection = expected_pd_greedy_select(
                            models, budget, false_alarm_rate, qos_pd=qos_pd,
                            qos_weights=qos_weights, grid=grid,
                        )
                        vector = np.asarray(selection.expected_pd)
                        rows.append({
                            "budget_bits": budget,
                            "seed_offset": offset,
                            "elements": elements,
                            "aperture_scale": aperture_scale,
                            "scenario": name,
                            "mean_expected_pd": float(np.mean(vector)),
                            "worst_expected_pd": float(np.min(vector)),
                            "no_ris_mean": float(np.mean(no_ris_vector)),
                            "no_ris_worst": float(np.min(no_ris_vector)),
                            "qos_feasible": bool(np.all(
                                vector >= qos_pd - 1e-9
                            )),
                        })

    summary = []
    for budget in budgets:
        for elements in elements_options:
            for aperture_scale in aperture_scale_options:
                group = [
                    row for row in rows
                    if row["budget_bits"] == budget
                    and row["elements"] == elements
                    and row["aperture_scale"] == aperture_scale
                ]
                aligned = [row for row in group if row["scenario"] == "ris_aligned"]
                blocked = [row for row in group
                           if row["scenario"] == "blocked_no_ris"]
                random_ris = [row for row in group if row["scenario"] == "ris_random"]
                summary.append({
                    "budget_bits": budget,
                    "elements": elements,
                    "aperture_scale": aperture_scale,
                    "no_ris_mean": float(np.mean([
                        row["no_ris_mean"] for row in aligned
                    ])),
                    "no_ris_worst": float(np.mean([
                        row["no_ris_worst"] for row in aligned
                    ])),
                    "aligned_mean": float(np.mean([
                        row["mean_expected_pd"] for row in aligned
                    ])),
                    "aligned_worst": float(np.mean([
                        row["worst_expected_pd"] for row in aligned
                    ])),
                    "random_mean": float(np.mean([
                        row["mean_expected_pd"] for row in random_ris
                    ])),
                    "random_worst": float(np.mean([
                        row["worst_expected_pd"] for row in random_ris
                    ])),
                    "mean_gain_aligned_vs_no_ris": float(np.mean([
                        row["mean_expected_pd"] - row["no_ris_mean"]
                        for row in aligned
                    ])),
                    "worst_gain_aligned_vs_no_ris": float(np.mean([
                        row["worst_expected_pd"] - row["no_ris_worst"]
                        for row in aligned
                    ])),
                    # P1-1: the correct control for the RIS-rescue claim is
                    # the "blocked, no RIS" direct-path floor.
                    "blocked_no_ris_worst": float(np.mean([
                        row["worst_expected_pd"] for row in blocked
                    ])),
                    "worst_gain_aligned_vs_blocked_no_ris": float(np.mean([
                        af["worst_expected_pd"] - bf["worst_expected_pd"]
                        for af, bf in zip(aligned, blocked)
                    ])),
                    "mean_gain_aligned_vs_random": float(np.mean([
                        aligned_row["mean_expected_pd"]
                        - random_row["mean_expected_pd"]
                        for aligned_row, random_row in zip(aligned, random_ris)
                    ])),
                    # P4 recovery metric (advice/011 section 5):
                    # eta = (P[aligned] - P[blocked]) / (P[clean] -
                    # P[blocked]) measures how much of the blockage-induced
                    # detection loss the aligned RIS restores.
                    "recovery_eta_worst": float(np.mean([
                        (af["worst_expected_pd"] - bf["worst_expected_pd"])
                        / max(af["no_ris_worst"] - bf["worst_expected_pd"],
                              1e-12)
                        for af, bf in zip(aligned, blocked)
                    ])),
                    "qos_feasible_rate": float(np.mean([
                        row["qos_feasible"] for row in aligned
                    ])),
                })

    payload = {
        "gate": "G5-P-ris-physics",
        "ris_position": ris_position.tolist(),
        "direct_blockage": 0.01,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ris_physics_gate.json")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--elements", type=int, nargs="+", default=[256, 1024])
    parser.add_argument(
        "--aperture-scales", type=float, nargs="+", default=[1e-3, 1e-2]
    )
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        seeds=args.seeds,
        budgets=args.budgets,
        grid=args.grid,
        elements_options=args.elements,
        aperture_scale_options=args.aperture_scales,
    )


if __name__ == "__main__":
    main()
