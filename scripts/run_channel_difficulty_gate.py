"""BSC channel-difficulty sweep for joint bit allocation.

The joint allocation experiments so far use a clean channel.  This gate adds
a binary-symmetric-channel flip probability and sweeps it over
0 / 0.05 / 0.10 / 0.15 for the same strong-vs-weak two-target model, so we
can see whether harder communication channels increase the value of exact
joint allocation over the greedy per-report rule.
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

from uav_otfs_isac.joint_allocation import (
    exact_joint_maxmin,
    greedy_bits,
    subset_options,
    target_options,
)


FLIPS = (0.0, 0.05, 0.10, 0.15)
BUDGETS = (14, 16, 18)


def _scenario(seed: int):
    rng = np.random.default_rng(seed)
    strong = np.concatenate(([0.4], rng.uniform(1.8, 2.2, 4)))
    weak = np.concatenate(([0.3], rng.uniform(1.2, 1.6, 4)))
    return strong, weak


def run_gate(*, output: Path, seeds: int, grid: int) -> None:
    pattern = np.array([0, 1, 2, 3, 4])
    rows = []
    for flip in FLIPS:
        for budget in BUDGETS:
            for seed in range(seeds):
                strong, weak = _scenario(seed)
                gs = np.concatenate((
                    [0], greedy_bits(strong[1:], budget, grid,
                                     bit_flip_probability=flip),
                ))
                gw = np.concatenate((
                    [0], greedy_bits(weak[1:], budget, grid,
                                     bit_flip_probability=flip),
                ))
                greedy = exact_joint_maxmin(
                    [
                        subset_options(
                            0.4, strong[1:], gs[1:], grid, flip,
                        ),
                        subset_options(
                            0.3, weak[1:], gw[1:], grid, flip,
                        ),
                    ],
                    budget,
                )
                exact = exact_joint_maxmin(
                    [
                        target_options(0.4, strong[1:], grid,
                                       bit_flip_probability=flip),
                        target_options(0.3, weak[1:], grid,
                                       bit_flip_probability=flip),
                    ],
                    budget,
                )
                fixed = exact_joint_maxmin(
                    [
                        subset_options(0.4, strong[1:], pattern[1:], grid, flip),
                        subset_options(0.3, weak[1:], pattern[1:], grid, flip),
                    ],
                    budget,
                )
                rows.append({
                    "bit_flip": flip,
                    "budget_bits": budget,
                    "seed": seed,
                    "fixed_pd": fixed,
                    "greedy_pd": greedy,
                    "exact_joint_pd": exact,
                    "gain_over_greedy_pp": float((exact - greedy) * 100.0),
                })

    summary = []
    for flip in FLIPS:
        for budget in BUDGETS:
            cell = [r for r in rows if r["bit_flip"] == flip
                    and r["budget_bits"] == budget]
            gains = [r["gain_over_greedy_pp"] for r in cell]
            summary.append({
                "bit_flip": flip,
                "budget_bits": budget,
                "n_seeds": len(cell),
                "fixed_pd_mean": float(np.mean([r["fixed_pd"] for r in cell])),
                "greedy_pd_mean": float(np.mean([r["greedy_pd"] for r in cell])),
                "exact_joint_pd_mean": float(
                    np.mean([r["exact_joint_pd"] for r in cell])
                ),
                "gain_over_greedy_mean_pp": float(np.mean(gains)),
            })

    payload = {
        "gate": "channel-difficulty-bsc-joint",
        "seeds": seeds,
        "grid": grid,
        "flips": list(FLIPS),
        "budgets": list(BUDGETS),
        "rows": rows,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/channel_difficulty_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--grid", type=int, default=32)
    args = parser.parse_args()
    run_gate(output=Path(args.output), seeds=args.seeds, grid=args.grid)


if __name__ == "__main__":
    main()
