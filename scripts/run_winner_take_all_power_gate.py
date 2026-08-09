"""Winner-take-all sensing power allocation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.power_split_theory import verify_winner_take_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/winner_take_all_power_gate.json")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        deltas = rng.uniform(0.8, 2.0, 4)
        bits = np.array([2, 3, 2, 3])
        row = verify_winner_take_all(
            0.4,
            deltas,
            bits,
            flip_probability=0.1,
            success_probability=0.8,
            budget=4.0,
            power_levels=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            grid=16,
        )
        rows.append({"seed": seed, **row})
    payload = {
        "gate": "winner-take-all-power",
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": payload["passed"],
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
