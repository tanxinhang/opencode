"""Communication-ambiguity endpoint-reduction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.communication_ambiguity import (
    build_endpoint_scenario_groups,
    verify_endpoint_dominance,
)
from uav_otfs_isac.robust_portfolio import (
    optimize_robust_chance_constrained_portfolio,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/communication_ambiguity_gate.json")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        deltas = rng.uniform(1.0, 2.0, 4)
        bits = np.array([2, 3, 2, 3])
        row = verify_endpoint_dominance(
            0.4,
            deltas,
            bits,
            (0.0, 0.2),
            (0.5, 1.0),
            scheduled=set(range(5)),
            minimum_pd=0.2,
            false_alarm_rate=0.05,
            grid=32,
            p_steps=5,
            s_steps=5,
        )
        rows.append({"seed": seed, **row})
    dp_rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        targets = [
            (
                0.4,
                rng.uniform(1.0, 2.0, 4),
                np.array([2, 3, 2, 3]),
            ),
            (
                0.3,
                rng.uniform(0.8, 1.5, 4),
                np.array([2, 2, 3, 3]),
            ),
        ]
        full, reduced = build_endpoint_scenario_groups(
            targets, (0.0, 0.2), (0.5, 1.0)
        )
        common = dict(
            budget_bits=8,
            minimum_quality=[0.2, 0.2],
            target_weights=[1.0, 1.3],
            violation_limits=[0.1, 0.1],
            quality_mode="gaussian_pd",
            false_alarm_rate=0.05,
        )
        full_result = optimize_robust_chance_constrained_portfolio(
            full, **common
        )
        reduced_result = optimize_robust_chance_constrained_portfolio(
            reduced, **common
        )
        dp_rows.append({
            "seed": seed,
            "full_scenarios": full_result.scenario_count,
            "reduced_scenarios": reduced_result.scenario_count,
            "full_worst_excess": full_result.worst_weighted_violation_excess,
            "reduced_worst_excess": (
                reduced_result.worst_weighted_violation_excess
            ),
            "equal": bool(np.isclose(
                full_result.worst_weighted_violation_excess,
                reduced_result.worst_weighted_violation_excess,
            )),
        })
    payload = {
        "gate": "communication-ambiguity-endpoint",
        "passed": all(row["passed"] for row in rows)
        and all(row["equal"] for row in dp_rows),
        "rows": rows,
        "dp_rows": dp_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": payload["passed"],
        "rows": rows,
        "dp_rows": dp_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
