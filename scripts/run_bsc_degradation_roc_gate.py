"""Exact-LRT ROC-dominance gate for BSC degradation ordering.

The gate verifies that, for quantized Gaussian sources, the exact likelihood
ratio under a cleaner BSC dominates the degraded BSC at every tested
false-alarm point.  The cascade identity

``BSC(hi) = BSC(lo) followed by BSC((hi-lo)/(1-2lo))``

is checked first, then the exact randomized-LRT P_D grid is audited.
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

from uav_otfs_isac.channel_degradation import (
    bsc_cascade_transition,
    verify_bsc_roc_dominance,
)
from uav_otfs_isac.reporting import bsc_transition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/bsc_degradation_roc_gate.json")
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--deltas", type=float, nargs="+",
                        default=[1.0, 1.5, 2.0])
    parser.add_argument("--lo", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2])
    parser.add_argument("--hi", type=float, nargs="+",
                        default=[0.3, 0.4, 0.45])
    parser.add_argument("--pfa-grid", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.2])
    args = parser.parse_args()

    result = verify_bsc_roc_dominance(
        bits_options=args.bits,
        mu1_options=args.deltas,
        lo_options=args.lo,
        hi_options=args.hi,
        false_alarm_grid=args.pfa_grid,
    )
    cascade_violations = []
    for bits in args.bits:
        for lo in args.lo:
            for hi in args.hi:
                if hi < lo:
                    continue
                cascade = bsc_cascade_transition(bits, lo, hi)
                direct = bsc_transition(bits, hi)
                if not np.allclose(cascade, direct, atol=1e-12):
                    cascade_violations.append({
                        "bits": bits,
                        "lo": lo,
                        "hi": hi,
                    })
    payload = {
        "gate": "bsc-degradation-roc",
        "bits": args.bits,
        "deltas": args.deltas,
        "lo_flips": args.lo,
        "hi_flips": args.hi,
        "pfa_grid": args.pfa_grid,
        "cascade_passed": not cascade_violations,
        "cascade_violations": cascade_violations,
        "result": result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "cells": result["cells"],
        "minimum_pd_gap_clean_minus_degraded": (
            result["minimum_pd_gap_clean_minus_degraded"]
        ),
        "cascade_passed": not cascade_violations,
        "roc_passed": result["passed"],
        "passed": result["passed"] and not cascade_violations,
    }, indent=2))


if __name__ == "__main__":
    main()
