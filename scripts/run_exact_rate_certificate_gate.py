"""G30-E gate: exact-objective rate certificate.

Gate G30 certifies a variable-rate profile as a single-rate local optimum of

``F_greedy(bits) = mean_seed min_q E_PD(q, S_q^greedy(bits))``.

Under heterogeneous report costs the greedy schedule is not exact, so that
certificate does not apply to the true system objective.  Gate G8-K evaluates
the exact schedule for the same budget, which defines

``F_exact(bits) = mean_seed min_q E_PD(q, S_q^exact(bits))``.

``F_exact`` uses the max-min exact selector so the inner objective matches
G30's worst-target value.  This gate re-checks every single-UAV quantizer-bit
change of the G30 profile under ``F_exact``, then runs exact coordinate
ascent until the profile is a single-change local optimum of the exact
objective.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.exact_quota_selection import exact_maxmin_select
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.ris_scenario import (
    RisConfig,
    ris_beam_phase,
    ris_control_overhead_bits,
    ris_physics_gain_matrix,
)
from uav_otfs_isac.scenario import build_models, target_geometry, uav_geometry


def _load_g30_profile(path: Path, total_budget: int) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload["summary"]:
        if cell["total_budget_bits"] == total_budget:
            return tuple(int(value) for value in cell["optimized_bits"])
    raise ValueError(f"G30 result has no cell for total budget {total_budget}")


def run_gate(
    *, output: Path, seeds: int, budgets, grid: int, qos_target: float,
    g30_result: Path, ris_elements: int, aperture_scale: float,
    phase_bits: int, coherence_frames: int, direct_blockage: float,
) -> None:
    cfg = load_config("config/demo.yaml")
    false_alarm_rate = cfg.false_alarm_rate
    qos_pd = np.full(cfg.num_targets, qos_target)
    qos_weights = np.asarray(cfg.qos_weights, dtype=float)
    transmitter_positions = uav_geometry(cfg.num_uavs)
    targets = [target_geometry(q) for q in range(cfg.num_targets)]
    receiver = np.array([0.0, 0.0, 0.0])
    seed_list = [cfg.seed + offset for offset in range(seeds)]
    ris = RisConfig(
        position=np.array([0.0, 30.0, 6.0]),
        num_elements=ris_elements,
        weak_target_id=cfg.num_targets - 1,
        phase_bits=phase_bits,
    )
    overhead = ris_control_overhead_bits(ris, coherence_frames=coherence_frames)
    phases = [ris_beam_phase(target, ris) for target in targets]
    gain = ris_physics_gain_matrix(
        ris, transmitter_positions, targets, receiver, aperture_scale,
        direct_blockage=direct_blockage, phase_per_target=phases,
    )

    @lru_cache(maxsize=None)
    def exact_value(bits: tuple[int, ...], budget_bits: int) -> float:
        worsts = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                quantizer_bits_per_uav=list(bits),
            )
            selection = exact_maxmin_select(
                models, budget_bits, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            worsts.append(float(np.min(selection.expected_pd)))
        return float(np.mean(worsts))

    @lru_cache(maxsize=None)
    def greedy_value(bits: tuple[int, ...], budget_bits: int) -> float:
        worsts = []
        for seed in seed_list:
            models = build_models(
                cfg, np.random.default_rng(seed), snr_gain=gain,
                quantizer_bits_per_uav=list(bits),
            )
            selection = expected_pd_greedy_select(
                models, budget_bits, false_alarm_rate, qos_pd=qos_pd,
                qos_weights=qos_weights, grid=grid,
            )
            worsts.append(float(np.min(selection.expected_pd)))
        return float(np.mean(worsts))

    def neighbors(profile: tuple[int, ...], budget_bits: int):
        rows = []
        for uav in range(cfg.num_uavs):
            for new_bits in (1, 2, 3):
                if new_bits == profile[uav]:
                    continue
                trial = list(profile)
                trial[uav] = new_bits
                trial = tuple(trial)
                rows.append({
                    "uav": uav,
                    "new_bits": new_bits,
                    "exact_value": float(exact_value(trial, budget_bits)),
                })
        return rows

    summary = []
    for total_budget in budgets:
        report_budget = int(total_budget - overhead)
        if report_budget < 0:
            continue
        start = _load_g30_profile(g30_result, total_budget)
        start_exact = exact_value(start, report_budget)
        start_greedy = greedy_value(start, report_budget)
        initial_rows = neighbors(start, report_budget)
        initial_improvements = [
            row for row in initial_rows
            if row["exact_value"] > start_exact + 1e-9
        ]

        current = start
        current_value = start_exact
        history = [{
            "bits": list(current),
            "exact_value": current_value,
        }]
        improved = True
        while improved:
            improved = False
            best_profile = None
            best_value = current_value
            for uav in range(cfg.num_uavs):
                for new_bits in (1, 2, 3):
                    if new_bits == current[uav]:
                        continue
                    trial = list(current)
                    trial[uav] = new_bits
                    trial = tuple(trial)
                    value = exact_value(trial, report_budget)
                    if value > best_value + 1e-9:
                        best_value = value
                        best_profile = trial
            if best_profile is not None:
                current = best_profile
                current_value = best_value
                improved = True
                history.append({
                    "bits": list(current),
                    "exact_value": current_value,
                })

        final_rows = neighbors(current, report_budget)
        final_improvements = [
            row for row in final_rows
            if row["exact_value"] > current_value + 1e-9
        ]
        adaptive = tuple([
            3 if index < max(1, report_budget // cfg.num_targets) else 1
            for index in range(cfg.num_uavs)
        ])
        summary.append({
            "total_budget_bits": total_budget,
            "report_budget_bits": report_budget,
            "g30_bits": list(start),
            "g30_greedy_value": start_greedy,
            "g30_exact_value": start_exact,
            "exact_optimized_bits": list(current),
            "exact_optimized_value": current_value,
            "exact_optimized_greedy_value": greedy_value(
                current, report_budget
            ),
            "exact_gain_over_g30": current_value - start_exact,
            "greedy_certificate_false_under_exact": bool(initial_improvements),
            "initial_exact_improvements": initial_improvements,
            "exact_single_change_local_optimal": bool(final_improvements) is False,
            "fixed3_exact_value": exact_value(
                tuple([1] * cfg.num_uavs), report_budget
            ),
            "fixed5_exact_value": exact_value(
                tuple([3] * cfg.num_uavs), report_budget
            ),
            "adaptive_exact_value": exact_value(adaptive, report_budget),
            "history": history,
        })

    payload = {
        "gate": "G30-E-exact-rate-certificate",
        "seeds": seeds,
        "grid": grid,
        "qos_target": qos_target,
        "g30_result": str(g30_result),
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps([{
        "total_budget_bits": cell["total_budget_bits"],
        "g30_exact_value": cell["g30_exact_value"],
        "exact_optimized_value": cell["exact_optimized_value"],
        "exact_gain_over_g30": cell["exact_gain_over_g30"],
        "greedy_certificate_false_under_exact": cell["greedy_certificate_false_under_exact"],
        "exact_single_change_local_optimal": cell["exact_single_change_local_optimal"],
    } for cell in summary], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/exact_rate_certificate_gate.json")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--budgets", type=int, nargs="+", default=[28])
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--qos-target", type=float, default=0.85)
    parser.add_argument(
        "--g30-result",
        default="results/global_rate_optimization_gate.json",
    )
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
        g30_result=Path(args.g30_result),
        ris_elements=args.ris_elements,
        aperture_scale=args.aperture_scale,
        phase_bits=args.phase_bits,
        coherence_frames=args.coherence_frames,
        direct_blockage=args.direct_blockage,
    )


if __name__ == "__main__":
    main()
