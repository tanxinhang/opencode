"""Resource fairness gate for the front-end integration comparison.

Reports three fairness paths:
1. Fixed sensing frame structure (same per-frame energy, report budget fixed).
2. Fixed total sensing energy (per-frame energy scales as 1/L).
3. Fixed total time-bandwidth occupation (identity + sensing + reports).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _scene_recovery(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    row = summary.get("2", {})
    mode = "front_end_covariance" if "front_end_covariance" in row else "fixed"
    return {
        "scene_exact_recovery": row[mode]["scene_exact_recovery"],
        "path_recall": row[mode]["path_recall"],
    }


def _resources(amplitude, frames, num_uavs, report_count, bits_per_report):
    sensing_energy = frames * num_uavs * amplitude ** 2
    sensing_time = frames
    identity_resources = num_uavs
    report_bits = report_count * bits_per_report
    report_time = report_count
    total_time = sensing_time + report_time
    total_occupation = sensing_energy + identity_resources + report_bits
    return {
        "per_frame_amplitude": amplitude,
        "otfs_frames": frames,
        "sensing_energy": sensing_energy,
        "sensing_time": sensing_time,
        "identity_resources": identity_resources,
        "report_bits": report_bits,
        "report_time": report_time,
        "total_time": total_time,
        "total_occupation": total_occupation,
    }


def run_gate(*, output: Path, single_frame: Path, four_frame: Path,
             equal_energy: Path) -> None:
    single = _scene_recovery(single_frame)
    four = _scene_recovery(four_frame)
    equal = _scene_recovery(equal_energy)
    num_uavs = 4
    report_count = 3
    bits_per_report = 5
    rows = [
        {
            "label": "L=1, A=2.0",
            "resources": _resources(
                2.0, 1, num_uavs, report_count, bits_per_report
            ),
            "scene_exact_recovery": single["scene_exact_recovery"],
            "path_recall": single["path_recall"],
        },
        {
            "label": "L=4, A=2.0 (fixed per-frame energy)",
            "resources": _resources(
                2.0, 4, num_uavs, report_count, bits_per_report
            ),
            "scene_exact_recovery": four["scene_exact_recovery"],
            "path_recall": four["path_recall"],
        },
        {
            "label": "L=4, A=1.0 (fixed total sensing energy)",
            "resources": _resources(
                1.0, 4, num_uavs, report_count, bits_per_report
            ),
            "scene_exact_recovery": equal["scene_exact_recovery"],
            "path_recall": equal["path_recall"],
        },
    ]
    payload = {
        "gate": "P2-resource-fairness",
        "paths": {
            "fixed_sensing_frame_structure": [
                row["label"] for row in rows[:2]
            ],
            "fixed_total_sensing_energy": [
                rows[0]["label"], rows[2]["label"],
            ],
            "fixed_total_occupation": [
                rows[0]["label"], rows[2]["label"],
            ],
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "fixed_per_frame_energy_gain": (
            four["scene_exact_recovery"] - single["scene_exact_recovery"]
        ),
        "fixed_total_energy_gain": (
            equal["scene_exact_recovery"] - single["scene_exact_recovery"]
        ),
        "rows": [{
            "label": row["label"],
            "sensing_energy": row["resources"]["sensing_energy"],
            "frames": row["resources"]["otfs_frames"],
            "total_occupation": row["resources"]["total_occupation"],
            "scene_exact_recovery": row["scene_exact_recovery"],
        } for row in rows],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/resource_fairness_gate.json"
    )
    parser.add_argument(
        "--single-frame",
        default="results/multistatic_front_end_gate.json",
    )
    parser.add_argument(
        "--four-frame",
        default="results/multistatic_front_end_integration.json",
    )
    parser.add_argument(
        "--equal-energy",
        default="results/multistatic_front_end_equal_energy_L4_A1.json",
    )
    args = parser.parse_args()
    run_gate(
        output=Path(args.output),
        single_frame=Path(args.single_frame),
        four_frame=Path(args.four_frame),
        equal_energy=Path(args.equal_energy),
    )


if __name__ == "__main__":
    main()
