"""P5-A same-cell multi test-seed ladder (advice/019 section 2, multi-seed
robustness of the mechanism attribution).

The registered ladder ``run_p5a_ablation_ladder.py`` runs the frozen
congested cell (geom=2, rho=1.8, scale=(16,8)) ONCE with held-out seed
400000 and attributes the gain to D_owner_bundle / D_pi / D_lambda /
D_admission.  That single draw is a point estimate of the attribution;
this wrapper re-runs the SAME ladder on the SAME cell with N independent
held-out CRN test-seed namespaces (500000 + k*... + mc, disjoint from
400000) and aggregates how STABLE the mechanism attribution is across
seeds:

  - per seed: the full A/B00/B0/B1/C ladder + D_* consecutive deltas with
    pooled-estimand block bootstrap CIs + dominant-mechanism verdict;
  - cross-seed: the dominant mechanism chosen per seed, the fraction of
    seeds on which each mechanism is the largest CERTIFIED positive
    delta, the per-delta mean/CI over seeds, and whether the direction
    (D_pi>0, D_owner_bundle>0, D_lambda<0) is sign-stable across seeds.

The registered runner is invoked as a subprocess per seed (same frozen
code path, same calibration seed 100, same geometry/scale/rho), so the
aggregate inherits every audit fix of advice/018/019.  It is a
credibility gate for the paper claim "detection-deficit task pricing is
the dominant algorithm-level mechanism" -- if the dominant verdict flips
across seeds, the claim must be weakened to a per-seed statement.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNNER = str(PROJECT_ROOT / "scripts" / "run_p5a_ablation_ladder.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/p5a_multiseed_ladder.json")
    parser.add_argument("--test-seeds", type=int, nargs="+",
                        default=[400000, 400001, 400002, 400003],
                        help="independent held-out ladder seed namespaces "
                             "(must be disjoint from one another AND from "
                             "the registered 400000 if used, so blocks "
                             "never repeat across seeds)")
    parser.add_argument("--test-cell-runs", type=int, default=500)
    parser.add_argument("--test-mc", type=int, default=24)
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter to run the ladder with")
    parser.add_argument("--ladder-extra", nargs="*", default=[],
                        help="extra args forwarded to the ladder runner "
                             "(e.g. --norm-free, --pi-bits 12)")
    args = parser.parse_args()
    t0 = time.time()

    seeds = sorted(set(args.test_seeds))
    per_seed = {}
    for k, seed in enumerate(seeds):
        out_path = PROJECT_ROOT / "results" / "_multiseed_tmp" / \
            f"ladder_seed_{seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [args.python, RUNNER, "--output", str(out_path),
               "--test-seed", str(seed),
               "--test-cell-runs", str(args.test_cell_runs),
               "--test-mc", str(args.test_mc), *args.ladder_extra]
        print(f"[{k+1}/{len(seeds)}] running ladder test_seed={seed} "
              f"({time.time()-t0:.0f}s)", flush=True)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        with open(out_path, encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload["metrics"]
        per_seed[seed] = {
            "ladder_J": metrics["ladder"],
            "deltas": {
                name: {"point": d["point"], "ci95": d["ci95"],
                       "state": d["state"],
                       "is_certified_gain": d["is_certified_gain"]}
                for name, d in metrics["deltas"].items()},
            "dominant": metrics["dominant_mechanism"],
            "qos": {a: metrics["arms"][a]["qos"]
                    for a in ("A", "B00", "B0", "B1", "C")},
        }

    # ---- cross-seed aggregation -------------------------------------------
    dom_keys = [per_seed[s]["dominant"]["key"] for s in seeds]
    dom_counts = {key: sum(1 for d in dom_keys if d == key)
                  for key in sorted(set(dom_keys))}
    # sign-stability of each consecutive delta across seeds
    sign_stability = {}
    for name in ("D_owner_bundle", "D_pi", "D_lambda", "D_admission"):
        pts = [per_seed[s]["deltas"][name]["point"] for s in seeds]
        gains = [1 for p in pts if p > 0]
        losses = [1 for p in pts if p < 0]
        cert_gains = [1 for s in seeds
                      if per_seed[s]["deltas"][name]["is_certified_gain"]]
        sign_stability[name] = {
            "point_mean": float(np_mean(pts)),
            "point_min": float(min(pts)),
            "point_max": float(max(pts)),
            "frac_positive": float(sum(gains)) / max(len(pts), 1),
            "frac_negative": float(sum(losses)) / max(len(pts), 1),
            "frac_certified_gain": float(sum(cert_gains))
            / max(len(seeds), 1),
            "sign_stable": bool(all(p > 0 for p in pts)
                                or all(p < 0 for p in pts)),
        }
    # dominant verdict stability
    if len(seeds) > 0:
        dominant_point = max(dom_counts, key=dom_counts.get)
        dominant_stable = dom_counts[dominant_point] == len(seeds)
    else:
        dominant_point = None
        dominant_stable = False

    payload = {
        "gate_id": "p5a-multiseed-ladder",
        "params": {
            "cell": "geom=2 congested rho=1.8 scale=(16,8)",
            "test_seeds": seeds,
            "test_episodes_per_seed": args.test_cell_runs * args.test_mc,
            "ladder_extra": args.ladder_extra,
            "seed_scheme": (
                "each seed is a disjoint held-out CRN namespace (same cell, "
                "same calibration 100, same frozen policy-B thresholds); the "
                "registered run used 400000, so 400000 may be included as "
                "one of the seeds to reproduce it"),
        },
        "metrics": {
            "per_seed": per_seed,
            "cross_seed": {
                "dominant_counts": dom_counts,
                "dominant_majority": dominant_point,
                "dominant_stable": dominant_stable,
                "sign_stability": sign_stability,
                "interpretation": (
                    "dominant_stable=True means EVERY test seed selected "
                    "the same dominant mechanism (the attribution is not "
                    "a single-seed accident).  sign_stability tells, per "
                    "consecutive delta, whether the DIRECTION (gain/loss) "
                    "reproduces across seeds and how often it is CERTIFIED "
                    "positive.  If D_pi is the stable dominant and always "
                    "certified positive, the paper claim 'detection-deficit "
                    "task pricing is the dominant algorithm-level mechanism' "
                    "is cross-seed supported; if D_owner_bundle dominates "
                    "instead, the paper must lead with the owner-directed "
                    "architecture bundle (advice/019 section 2)."),
            },
        },
        "runtime_s": round(time.time() - t0, 1),
        "provenance": {
            "git_commit": _git_sha(),
            "git_dirty": _git_dirty(),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("cross_seed:", json.dumps(payload["metrics"]["cross_seed"],
                                    indent=1))
    print("done", round(time.time() - t0, 1), "s")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
            text=True)
        return bool(out.strip())
    except Exception:
        return True


def np_mean(xs) -> float:
    return float(sum(xs)) / max(len(xs), 1)


if __name__ == "__main__":
    main()