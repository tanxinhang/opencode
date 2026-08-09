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

from uav_otfs_isac.communication_ambiguity import verify_endpoint_dominance


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
    payload = {
        "gate": "communication-ambiguity-endpoint",
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
