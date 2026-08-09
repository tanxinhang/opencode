"""Velocity-bounded sensing mobility envelope gate.

Checks that bounded target displacement implies bounded UAV-target range
change and bounded free-space power change, so the stress suite's mobility
axis stays inside a physically defensible sensing envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.config import load_config
from uav_otfs_isac.mobility_envelope import (
    verify_displacement_envelope,
    verify_range_snr_envelope,
)
from uav_otfs_isac.scenario import target_geometry, uav_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/sensing_mobility_envelope_gate.json")
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--max-displacement", type=float, default=8.0)
    args = parser.parse_args()

    cfg = load_config("config/demo.yaml")
    positions = uav_geometry(cfg.num_uavs)
    rows = []
    for q in range(cfg.num_targets):
        free_space = verify_displacement_envelope(
            positions,
            target_geometry(q),
            args.max_displacement,
            samples=args.samples,
        )
        range_snr = verify_range_snr_envelope(
            cfg,
            positions,
            target_geometry(q),
            args.max_displacement,
            samples=args.samples,
        )
        rows.append({
            "target": q,
            "free_space": free_space,
            "range_snr": range_snr,
        })
    payload = {
        "gate": "sensing-mobility-envelope",
        "max_displacement": args.max_displacement,
        "samples_per_target": args.samples,
        "rows": rows,
        "passed": all(
            row["free_space"]["passed"] and row["range_snr"]["passed"]
            for row in rows
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": [
            {
                "target": row["target"],
                "free_space_bound": row["free_space"]["path_loss_relative_bound"],
                "range_snr_bound": row["range_snr"]["range_snr_relative_bound"],
                "free_space_passed": row["free_space"]["passed"],
                "range_snr_passed": row["range_snr"]["passed"],
            }
            for row in rows
        ],
        "passed": payload["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
