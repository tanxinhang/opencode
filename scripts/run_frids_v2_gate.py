"""Gate F0-G4: FRIDS-v2 (advice/010 step 2).

FRIDS-v2 keeps the FRIDS idea and fixes the dual consistency and the
provenance: demand-normalized primal (local score ``y_q * g_iq / (D_q +
eps)``, dual on the ordinary simplex), exponentiated-gradient mirror
descent on the normalized service gap (no nu_floor), and STRICTLY LOCAL
state (each UAV uses its own belief for the deficit, its own received
info for the service, and its own price vector -- no owner-belief
access).  Compared against FRIDS-v1 at all scales with 10 scenario seeds
at the critical scales; the life-or-death gate:

- Delta J at (16,8) >= 3% (v2 better);
- P_FA <= alpha + tol, P_MD <= beta + tol (policy-matched B per
  algorithm/scenario, adaptive offset);
- no small-scale regression (v2 J(12,6) <= v1 J(12,6) + 2%);
- stable cross-scenario win rate (>= 0.6).

If v2 gains less than ~2%, it is NOT adopted (v1 stays, the line moves
to feasibility theory).  The information-load cut ``rho(S)`` is computed
per scenario as the theory-grounded feasibility certificate.
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
from uav_otfs_isac.frids import (
    load_cut,
    simulate_frids,
    simulate_frids_v2,
)

SCALES_ALL = ((6, 3), (8, 4), (12, 6), (16, 8))
N_SCEN = {6: 5, 8: 5, 12: 10, 16: 10}   # scenarios per scale


def eval_algo(sim, sc, bounds, n_runs, seeds, max_steps, **kw):
    J, md, fa = [], [], []
    for seed in range(seeds):
        out = sim(sc, bounds, n_runs=n_runs, seed=seed * 1000 + 7,
                  max_steps=max_steps, **kw)
        J.append(out["worst_target_delay"])
        md.append(max(out["p_md"]))
        fa.append(max(out["p_fa"]))
    return {"J": float(np.mean(J)), "p_md_max": float(np.max(md)),
            "p_fa_max": float(np.max(fa))}


def matched_b(sim, sc, bt, n_runs, seeds, max_steps, alpha, beta,
              scan_runs=120, scan_seeds=1, **sim_kw):
    """Adaptive policy-matched B: lower B by delta until P_MD enters the
    constraint (escalate 1.0 -> 2.0 -> 3.0), verify at the scan MC."""
    q = sc["q"]
    for delta in (1.0, 2.0, 3.0):
        cand = [[bt[qq][0], bt[qq][1] - delta] for qq in range(q)]
        row = eval_algo(sim, sc, cand, scan_runs, scan_seeds,
                        max_steps, **sim_kw)
        if row["p_md_max"] <= beta + 0.005:
            return cand
    # fallback: deepest delta
    return [[bt[qq][0], bt[qq][1] - 3.0] for qq in range(q)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/frids_v2_gate.json")
    parser.add_argument("--n-runs", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    parser.add_argument("--mu-v1", type=float, default=0.2)
    parser.add_argument("--mu-v2", type=float, default=0.5)
    args = parser.parse_args()

    t0 = time.time()
    rows = {}
    for (k, q) in SCALES_ALL:
        n_scen = N_SCEN[k]
        per = {}
        for s in range(n_scen):
            sc = build_distributed_scenario(np.random.default_rng(s),
                                            k_uavs=k, q_targets=q)
            bt = calibrate_target_bounds(
                sc, args.alpha, args.beta, n_runs=300,
                seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                verify_runs=args.calib_verify)
            b1 = matched_b(simulate_frids, sc, bt, args.n_runs,
                           args.seeds, args.max_steps, args.alpha,
                           args.beta, mu=args.mu_v1)
            b2 = matched_b(simulate_frids_v2, sc, bt, args.n_runs,
                           args.seeds, args.max_steps, args.alpha,
                           args.beta, mu=args.mu_v2)
            r1 = eval_algo(simulate_frids, sc, b1, args.n_runs,
                           args.seeds, args.max_steps, mu=args.mu_v1)
            r2 = eval_algo(simulate_frids_v2, sc, b2, args.n_runs,
                           args.seeds, args.max_steps, mu=args.mu_v2)
            # information-load cuts (t=0): full set + worst singleton
            owner = sc["owner_of"]
            rho_full = load_cut(sc, owner, list(range(q)),
                                args.max_steps, args.beta, args.alpha)
            per[str(s)] = {
                "v1": r1, "v2": r2,
                "b1_min": float(min(b[1] for b in b1)),
                "b2_min": float(min(b[1] for b in b2)),
                "rho_full": rho_full,
            }
            print(f"({k},{q}) s{s}: v1 {r1['J']:.2f} (P_MD "
                  f"{r1['p_md_max']:.3f}) v2 {r2['J']:.2f} (P_MD "
                  f"{r2['p_md_max']:.3f}) rho_full {rho_full:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        def agg(name):
            vals = [v[name]["J"] for v in per.values()]
            return {"mean_J": float(np.mean(vals)),
                    "sd_J": float(np.std(vals, ddof=1)),
                    "p_md_max": float(max(v[name]["p_md_max"]
                                          for v in per.values())),
                    "p_fa_max": float(max(v[name]["p_fa_max"]
                                          for v in per.values()))}
        rows[f"{k}_{q}"] = {
            "k": k, "q": q, "n_scenarios": n_scen,
            "v1": agg("v1"), "v2": agg("v2"),
            "per_scenario": per,
        }
    # life-or-death gate on (16,8), no-regression on (12,6)
    r168 = rows["16_8"]
    r126 = rows["12_6"]
    j1 = [v["v1"]["J"] for v in r168["per_scenario"].values()]
    j2 = [v["v2"]["J"] for v in r168["per_scenario"].values()]
    gain = (r168["v1"]["mean_J"] - r168["v2"]["mean_J"]) \
        / max(r168["v1"]["mean_J"], 1e-12)
    reg12 = (r126["v2"]["mean_J"] - r126["v1"]["mean_J"]) \
        / max(r126["v1"]["mean_J"], 1e-12)
    win = float(np.mean([a < b for a, b in zip(j2, j1)]))
    err_ok = (r168["v2"]["p_md_max"] <= args.beta + 0.02
              and r168["v2"]["p_fa_max"] <= args.alpha + 0.02
              and r126["v2"]["p_md_max"] <= args.beta + 0.02)
    gate = {
        "gain_16_8": float(gain),
        "gain_meets_3pct": bool(gain >= 0.03),
        "regression_12_6": float(reg12),
        "regression_within_2pct": bool(reg12 <= 0.02),
        "error_ok": bool(err_ok),
        "win_rate_16_8": float(win),
        "win_rate_stable": bool(win >= 0.6),
        "adopt_v2": bool(gain >= 0.03 and reg12 <= 0.02 and err_ok
                        and win >= 0.6),
        "verdict": ("FRIDS-v2 adopted"
                    if gain >= 0.03 and reg12 <= 0.02 and err_ok
                    and win >= 0.6
                    else ("FRIDS-v1 stays; move to feasibility theory"
                          if gain < 0.03 or not err_ok
                          else "v2 adopted with caution")),
    }
    payload = {
        "gate": "f0g4-frids-v2",
        "params": {
            "scales": [list(s) for s in SCALES_ALL],
            "n_scenarios_per_scale": N_SCEN,
            "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "calib_seed": args.calib_seed,
            "calib_verify": args.calib_verify,
            "mu_v1": args.mu_v1, "mu_v2": args.mu_v2,
            "frozen": ["fixed owner", "full mesh", "19-bit token "
                       "(scale-aware q/intent fields)", "communication-"
                       "domain beliefs", "policy-matched B", "current "
                       "scenario gen"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    for key in rows:
        r = rows[key]
        print(f"{key}: v1 {r['v1']['mean_J']:.3f} +/- {r['v1']['sd_J']:.3f}"
              f"  v2 {r['v2']['mean_J']:.3f} +/- {r['v2']['sd_J']:.3f}")
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
