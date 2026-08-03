from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.adaptive import controlled_two_report_oracles


def main() -> None:
    rows = []
    for p in (0.3, 0.5, 0.7, 0.9):
        independent = controlled_two_report_oracles(
            [1.0], [[p, p]], [[p, p]]
        )
        rows.append({
            "case": "independent", "p": p,
            **{
                f"{name}_{metric}": getattr(result, metric)
                for name, result in independent.items()
                for metric in ("success_probability", "expected_transmissions")
            },
        })
    persistent = controlled_two_report_oracles(
        [0.25] * 4,
        [[0.25, 0.35], [0.25, 0.85], [0.75, 0.35], [0.75, 0.85]],
        [[0.20, 0.30], [0.70, 0.80], [0.20, 0.30], [0.70, 0.80]],
    )
    rows.append({
        "case": "persistent_shared_state",
        **{
            f"{name}_{metric}": getattr(result, metric)
            for name, result in persistent.items()
            for metric in ("success_probability", "expected_transmissions")
        },
        "adaptive_policy_by_ack_00_01_10_11": list(persistent["adaptive"].policy),
    })
    payload = {
        "controlled_only": True,
        "hard_worst_case_transmissions": 3,
        "rows": rows,
    }
    output = Path("results/controlled_adaptive_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
