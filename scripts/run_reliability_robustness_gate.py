"""Gate F0-G5: reliability robustness (advice/011).

The scheduler believes ``s_hat = clip(s_true / kappa_s, 0, 1)``
(kappa_s = s_true / s_assumed; kappa_s < 1 is the dangerous direction:
overestimating reliability).  Three variants, all else frozen:

- nominal : g uses s_hat;
- robust  : g uses s^- = max(0, s_hat * (1 - eps)) -- the worst point of
  the interval uncertainty set, exact because g(s) = s * I+ is monotone
  linear in s (no sampling, no inner minimization);
- oracle  : g uses the true s.

The delivery draws always use the TRUE matrix.  Life-or-death gate:
robust must improve over nominal by >= 5% at kappa_s <= 0.8 while
keeping P_MD <= beta + tol, and degrade by no more than 2% at kappa_s =
1.  If not met: keep FRIDS-v2, no robust layer.

Writes ``results/reliability_robustness_gate.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.frids import simulate_frids_v2

SCALES = ((12, 6), (16, 8))
KAPPAS = (1.0, 0.9, 0.8, 0.6)
ROBUST_EPS = 0.2


def eval_v2(sc, bounds, n_runs, seeds, max_steps, **kw):
    J, md, fa = [], [], []
    for seed in range(seeds):
        out = simulate_frids_v2(sc, bounds, n_runs=n_runs,
                                seed=seed * 1000 + 7,
                                max_steps=max_steps, **kw)
        J.append(out["worst_target_delay"])
        md.append(max(out["p_md"]))
        fa.append(max(out["p_fa"]))
    return {"J": float(np.mean(J)), "p_md_max": float(np.max(md)),
            "p_fa_max": float(np.max(fa))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/reliability_robustness_gate.json")
    parser.add_argument("--n-runs", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--n-scenarios", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    parser.add_argument("--b-delta", type=float, default=1.0,
                        help="policy-matched B offset (frozen)")
    args = parser.parse_args()

    t0 = time.time()
    rows = {}
    for (k, q) in SCALES:
        per = {}
        for s in range(args.n_scenarios):
            sc = build_distributed_scenario(np.random.default_rng(s),
                                            k_uavs=k, q_targets=q)
            bt = calibrate_target_bounds(
                sc, args.alpha, args.beta, n_runs=300,
                seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                verify_runs=args.calib_verify)
            bounds = [[bt[qq][0], bt[qq][1] - args.b_delta]
                      for qq in range(q)]
            u2u = sc["u2u_success"]
            cell = {}
            for kappa in KAPPAS:
                s_hat = np.clip(u2u / kappa, 0.0, 1.0)
                s_minus = np.maximum(s_hat * (1.0 - ROBUST_EPS), 0.0)
                nom = eval_v2(sc, bounds, args.n_runs, args.seeds,
                              args.max_steps, delivery_matrix=u2u,
                              s_for_g=s_hat)
                rob = eval_v2(sc, bounds, args.n_runs, args.seeds,
                              args.max_steps, delivery_matrix=u2u,
                              s_for_g=s_minus)
                orc = eval_v2(sc, bounds, args.n_runs, args.seeds,
                              args.max_steps, delivery_matrix=u2u,
                              s_for_g=u2u)
                cell[str(kappa)] = {"nominal": nom, "robust": rob,
                                    "oracle": orc}
                print(f"({k},{q}) s{s} kappa {kappa}: nom "
                      f"{nom['J']:.2f} rob {rob['J']:.2f} "
                      f"orc {orc['J']:.2f} "
                      f"(P_MD {rob['p_md_max']:.3f}) "
                      f"({time.time()-t0:.0f}s)", flush=True)
            per[str(s)] = cell
        rows[f"{k}_{q}"] = {"k": k, "q": q, "per_scenario": per}

    # aggregate + life-or-death gate on (16,8)
    r168 = rows["16_8"]

    def agg(scale_rows, kappa, algo):
        vals = [v[str(kappa)][algo]["J"] for v in scale_rows.values()]
        return {"mean_J": float(np.mean(vals)),
                "p_md_max": float(max(v[str(kappa)][algo]["p_md_max"]
                                      for v in scale_rows.values()))}

    for key, rr in rows.items():
        rows[key]["agg"] = {
            str(k): {"nominal": agg(rr["per_scenario"], k, "nominal"),
                     "robust": agg(rr["per_scenario"], k, "robust"),
                     "oracle": agg(rr["per_scenario"], k, "oracle")}
            for k in KAPPAS
        }
    nom1 = r168["agg"]["1.0"]["nominal"]["mean_J"]
    gate = {}
    for kappa in (0.8, 0.6):
        n = r168["agg"][str(kappa)]["nominal"]
        r_ = r168["agg"][str(kappa)]["robust"]
        gate[f"kappa_{str(kappa).replace('.', '_')}"] = {
            "nominal_J": n["mean_J"],
            "robust_J": r_["mean_J"],
            "robust_gain": float((n["mean_J"] - r_["mean_J"])
                                 / max(n["mean_J"], 1e-12)),
            "p_md_max": r_["p_md_max"],
            "meets_5pct": bool(r_["mean_J"] <= n["mean_J"] * 0.95),
            "error_ok": bool(r_["p_md_max"] <= args.beta + 0.02),
        }
    deg1 = r168["agg"]["1.0"]["robust"]["mean_J"] - nom1
    gate["kappa_1_0"] = {
        "nominal_J": nom1,
        "robust_J": r168["agg"]["1.0"]["robust"]["mean_J"],
        "degradation": float(deg1 / max(nom1, 1e-12)),
        "within_2pct": bool(deg1 <= nom1 * 0.02 + 1e-9),
    }
    # oracle recovery ratio at kappa=0.6
    n06 = r168["agg"]["0.6"]["nominal"]["mean_J"]
    r06 = r168["agg"]["0.6"]["robust"]["mean_J"]
    o06 = r168["agg"]["0.6"]["oracle"]["mean_J"]
    gate["oracle_recovery_ratio_kappa_0_6"] = float(
        (n06 - r06) / max(n06 - o06, 1e-12))
    gate["adopt_robust"] = bool(
        gate["kappa_0_8"]["meets_5pct"] and gate["kappa_0_8"]["error_ok"]
        and gate["kappa_0_6"]["meets_5pct"]
        and gate["kappa_0_6"]["error_ok"]
        and gate["kappa_1_0"]["within_2pct"])
    gate["verdict"] = ("Robust-FRIDS adopted"
                       if gate["adopt_robust"]
                       else "keep FRIDS-v2; no robust layer")
    # per-link mismatch probe: a uniform global kappa scales every g by
    # the same factor, which preserves the argmax ranking (FRIDS is
    # scale-invariant to the reliability factor); the harmful case is
    # PER-LINK misestimation.  Probe with random per-link kappa_ij.
    probe = {}
    for kappa_lo in (0.6, 0.8):
        pn, pr, po = [], [], []
        for s in range(min(args.n_scenarios, 5)):
            sc = build_distributed_scenario(np.random.default_rng(s),
                                            k_uavs=16, q_targets=8)
            bt = calibrate_target_bounds(
                sc, args.alpha, args.beta, n_runs=300,
                seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                verify_runs=args.calib_verify)
            bounds = [[bt[qq][0], bt[qq][1] - args.b_delta]
                      for qq in range(8)]
            u2u = sc["u2u_success"]
            rng = np.random.default_rng(1000 + s)
            kappa_ij = rng.uniform(kappa_lo, 1.0, u2u.shape)
            np.fill_diagonal(kappa_ij, 1.0)
            s_hat = np.clip(u2u / kappa_ij, 0.0, 1.0)
            s_minus = np.maximum(s_hat * (1.0 - ROBUST_EPS), 0.0)
            pn.append(eval_v2(sc, bounds, args.n_runs, args.seeds,
                              args.max_steps, delivery_matrix=u2u,
                              s_for_g=s_hat)["J"])
            pr.append(eval_v2(sc, bounds, args.n_runs, args.seeds,
                              args.max_steps, delivery_matrix=u2u,
                              s_for_g=s_minus)["J"])
            po.append(eval_v2(sc, bounds, args.n_runs, args.seeds,
                              args.max_steps, delivery_matrix=u2u,
                              s_for_g=u2u)["J"])
        probe[f"per_link_{kappa_lo}"] = {
            "nominal_J": float(np.mean(pn)),
            "robust_J": float(np.mean(pr)),
            "oracle_J": float(np.mean(po)),
            "robust_gain": float((np.mean(pn) - np.mean(pr))
                                 / max(np.mean(pn), 1e-12)),
        }
        print(f"per-link probe kappa_lo {kappa_lo}: "
              f"nom {np.mean(pn):.3f} rob {np.mean(pr):.3f} "
              f"orc {np.mean(po):.3f}", flush=True)
    gate["per_link_probe"] = probe
    payload = {
        "gate": "f0g5-reliability-robustness",
        "params": {
            "scales": [list(s) for s in SCALES],
            "n_scenarios": args.n_scenarios,
            "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "calib_seed": args.calib_seed,
            "calib_verify": args.calib_verify,
            "b_delta": args.b_delta,
            "kappas": list(KAPPAS),
            "robust_eps": ROBUST_EPS,
            "frozen": ["FRIDS-v2", "fixed owner", "full mesh",
                       "17/19-bit token", "static geometry",
                       "current kernels", "current thresholds"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
