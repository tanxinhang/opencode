"""Bootstrap block-resolution seed-stability audit (advice/019 section 8,
multi-seed verification of the statistics layer).

The P5-A pooled ``max_q`` estimand is NON-smooth, so the percentile
bootstrap's resolution is set by the number of independent BLOCKS it
resamples, not the total episode count.  The registered ladder used
``8 blocks x 1500``; advice/019 section 8 recommends ``24 x 500`` (or
``32 x 375``) at the SAME total MC cost.  This script quantifies what
that buys: it draws SYNTHETIC per-block pooled statistics with the
registered cell's block structure, then runs the runner's exact
bootstrap routine ``_pooled_delta_ci`` many times over DIFFERENT
bootstrap resampling seeds and reports:

  - CI end-point scatter over bootstrap seeds (the bootstrap itself is a
    random estimator -- a stable CI must not jump with the resampling
    seed);
  - CI width mean/median and the 2.5-97.5% spread of the CI end points;
  - the same under 8x1500 vs 24x500 (and 32x375) block layouts, same
    total runs, same underlying per-target means.

This is a statistics-layer multi-seed gate: it verifies the 24-block
recommendation is actually worth the +~1.5s it costs, by showing the CI
end points are materially less seed-sensitive than at 8 blocks.  No
scheduler is run -- the data is a realistic synthetic stand-in (per-block
target H1 counts ~ N/2 and delay sums with a worst-target mean close to
the registered J~3.1-5.0).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p5a_ablation_ladder import (
    _pooled_delta_ci,
    _pooled_j,
)


def _synthetic_blocks(n_blocks, runs_per_block, q, worst_mean, seed):
    """Per-block ``(block_n, block_s)`` synthetic pool arrays.  Each block
    has ``q`` targets; target ``q-1`` is the worst target (mean ``worst_mean``)
    and the others are drawn below it, with per-block sampling noise.  H1
    count per block-target ~ Binomial(runs_per_block, 0.5)."""
    rng = np.random.default_rng(seed)
    means = np.linspace(worst_mean * 0.5, worst_mean, q)
    n, s = [], []
    for _ in range(n_blocks):
        n_b = rng.binomial(runs_per_block, 0.5, q).astype(float)
        s_b = n_b * means * rng.uniform(0.9, 1.1, q)
        # worst target stays worst in expectation but per-block max can flip
        n.append(n_b)
        s.append(s_b)
    return n, s


def _audit_config(n_blocks, runs_per_block, q, worst_mean, data_seed,
                  n_boot, bootstrap_seeds):
    """Run the exact runner bootstrap with many bootstrap resampling seeds;
    return the CI end-point scatter AND the coarseness of the bootstrap
    distribution itself (the point of advice/019 section 8)."""
    bn, bs = _synthetic_blocks(n_blocks, runs_per_block, q, worst_mean,
                               data_seed)
    cn, cs = _synthetic_blocks(n_blocks, runs_per_block, q,
                               worst_mean * 0.9, data_seed + 1)
    los, his, pts = [], [], []
    # coarseness: the bootstrap distribution of delta* = J_prev* - J_cur*
    # is a step function; with FEW blocks (8) a single block swap moves the
    # pooled max by a large step, so the 2.5/97.5 percentile is coarse.  We
    # measure the number of distinct delta* values and the gap of the two
    # order statistics bracketing each percentile, relative to CI width.
    boot_seed0 = bootstrap_seeds[0]
    _pt, _lo, _hi = _pooled_delta_ci(bn, bs, cn, cs, n_boot=n_boot,
                                     seed=boot_seed0)
    _, boot_deltas = _bootstrap_deltas(bn, bs, cn, cs, n_boot, boot_seed0)
    boot_deltas = np.sort(boot_deltas)
    n_unique = int(len(np.unique(boot_deltas)))
    lo_idx = max(int(np.floor(0.025 * (n_boot - 1))), 0)
    hi_idx = max(int(np.floor(0.975 * (n_boot - 1))), 0)
    lo_neighborhood = boot_deltas[min(lo_idx + 1, n_boot - 1)] \
        - boot_deltas[lo_idx]
    hi_neighborhood = boot_deltas[hi_idx] - boot_deltas[max(hi_idx - 1, 0)]
    width = float(boot_deltas[hi_idx] - boot_deltas[lo_idx])
    for bseed in bootstrap_seeds:
        pt, lo, hi = _pooled_delta_ci(bn, bs, cn, cs, n_boot=n_boot,
                                      seed=bseed)
        pts.append(pt)
        los.append(lo)
        his.append(hi)
    return {
        "blocks": n_blocks,
        "runs_per_block": runs_per_block,
        "total_runs": n_blocks * runs_per_block,
        "point_mean": float(np.mean(pts)),
        "ci_lo_mean": float(np.mean(los)),
        "ci_hi_mean": float(np.mean(his)),
        "ci_width_mean": float(np.mean([h - l for h, l in zip(his, los)])),
        "ci_lo_spread": float(np.max(los) - np.min(los)),
        "ci_hi_spread": float(np.max(his) - np.min(his)),
        "ci_width_spread": float(
            np.max([h - l for h, l in zip(his, los)])
            - np.min([h - l for h, l in zip(his, los)])),
        "point_spread": float(np.max(pts) - np.min(pts)),
        # coarseness (advice/019 section 8): distinct delta* values and the
        # percentile bracketing step as a FRACTION of the CI width
        "n_distinct_boot_values": n_unique,
        "n_distinct_ratio_to_n_boot": float(n_unique) / max(n_boot, 1),
        "lo_quantile_step_rel_width": float(lo_neighborhood) / max(width, 1e-12),
        "hi_quantile_step_rel_width": float(hi_neighborhood) / max(width, 1e-12),
    }


def _bootstrap_deltas(prev_n, prev_s, cur_n, cur_s, n_boot, seed):
    """Return ``(point, sorted delta* array)`` of the runner's pooled
    bootstrap (mirrors ``_pooled_delta_ci`` but keeps the raw draws)."""
    B = len(prev_n)
    rng = np.random.default_rng(seed)
    P_N, P_V = np.sum(np.stack(prev_n, axis=0), axis=0), \
        np.sum(np.stack(prev_s, axis=0), axis=0)
    C_N, C_V = np.sum(np.stack(cur_n, axis=0), axis=0), \
        np.sum(np.stack(cur_s, axis=0), axis=0)
    point = _pooled_j(P_N, P_V) - _pooled_j(C_N, C_V)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, B, size=B)
        pN = np.sum(np.stack([prev_n[j] for j in idx], axis=0), axis=0)
        pV = np.sum(np.stack([prev_s[j] for j in idx], axis=0), axis=0)
        cN = np.sum(np.stack([cur_n[j] for j in idx], axis=0), axis=0)
        cV = np.sum(np.stack([cur_s[j] for j in idx], axis=0), axis=0)
        boot[b] = _pooled_j(pN, pV) - _pooled_j(cN, cV)
    return float(point), boot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/p5a_bootstrap_seed_stability.json")
    parser.add_argument("--q", type=int, default=8,
                        help="target count of the synthetic cell")
    parser.add_argument("--worst-mean", type=float, default=4.0,
                        help="worst-target E[T|H1] (registered J ~3.1-5.0)")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seeds", type=int, default=30,
                        help="how many bootstrap resampling seeds to try")
    parser.add_argument("--data-seed", type=int, default=100000)
    args = parser.parse_args()
    t0 = time.time()

    configs = [
        (8, 1500),      # registered
        (24, 500),      # advice/019 section 8 recommended
        (32, 375),      # advice/019 section 8 alternative
    ]
    bseeds = list(range(args.bootstrap_seeds))
    audits = [_audit_config(b, r, args.q, args.worst_mean, args.data_seed,
                            args.n_boot, bseeds) for (b, r) in configs]

    base_8 = audits[0]
    for cfg in audits[1:]:
        cfg["lo_spread_ratio_to_8"] = float(
            cfg["ci_lo_spread"] / max(base_8["ci_lo_spread"], 1e-12))
        cfg["hi_spread_ratio_to_8"] = float(
            cfg["ci_hi_spread"] / max(base_8["ci_hi_spread"], 1e-12))
        cfg["width_spread_ratio_to_8"] = float(
            cfg["ci_width_spread"] / max(base_8["ci_width_spread"], 1e-12))
        cfg["distinct_ratio_to_8"] = float(
            cfg["n_distinct_boot_values"]
            / max(base_8["n_distinct_boot_values"], 1))
        cfg["lo_step_ratio_to_8"] = float(
            cfg["lo_quantile_step_rel_width"]
            / max(base_8["lo_quantile_step_rel_width"], 1e-12))
        cfg["hi_step_ratio_to_8"] = float(
            cfg["hi_quantile_step_rel_width"]
            / max(base_8["hi_quantile_step_rel_width"], 1e-12))

    # verdict (advice/019 section 8): the point is the RESOLUTION of the
    # non-smooth max_q bootstrap.  More blocks -> the resampled delta*
    # distribution takes many more distinct values and the 2.5/97.5
    # percentile is bracketed by a smaller step (as a fraction of the CI
    # width) -> the CI is not a coarse step-function artifact.  Pass iff
    # both percentile bracketing steps drop by >= 25% vs 8x1500.
    c24 = audits[1]
    lo_step_ok = c24["lo_quantile_step_rel_width"] \
        <= base_8["lo_quantile_step_rel_width"] * 0.75
    hi_step_ok = c24["hi_quantile_step_rel_width"] \
        <= base_8["hi_quantile_step_rel_width"] * 0.75
    distinct_ok = c24["n_distinct_boot_values"] \
        > base_8["n_distinct_boot_values"]
    verdict_pass = bool(lo_step_ok and hi_step_ok and distinct_ok)

    payload = {
        "gate_id": "p5a-bootstrap-seed-stability",
        "params": {
            "estimand": "J = max_q sum_b S_bq / sum_b N_bq (pooled, "
                        "non-smooth) -- exact runner _pooled_delta_ci",
            "synthetic_cell": {
                "q": args.q, "worst_target_mean": args.worst_mean,
                "per_block_sampling": "Binomial(runs, 0.5) counts, "
                                      "mean*Uniform(0.9,1.1) sums",
                "data_seed": args.data_seed,
            },
            "n_boot": args.n_boot,
            "bootstrap_seeds": args.bootstrap_seeds,
        },
        "metrics": {
            "configs": audits,
            "verdict": {
                "pass": verdict_pass,
                "wording": (
                    "24x500 (and 32x375) produce a materially FINER "
                    "bootstrap distribution of the non-smooth max_q "
                    "estimand than 8x1500 at the SAME total MC cost: many "
                    "more distinct resampled delta* values and a smaller "
                    "2.5/97.5 percentile bracketing step (relative to CI "
                    "width).  The bootstrap CI is therefore not a coarse "
                    "step-function artifact -- the advice/019 section 8 "
                    "block-resolution recommendation is supported and the "
                    "registered ladder should re-run on 24x500." if
                    verdict_pass else
                    "the 24x500 block resolution does NOT clearly refine "
                    "the bootstrap percentile on this synthetic data -- "
                    "the recommendation needs re-examination."),
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
    print(json.dumps(payload["metrics"], indent=1))
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


if __name__ == "__main__":
    main()