"""Gate F0-G6: Q/K x U2U-reliability feasibility envelope (advice/012).

FRIDS-v2 is frozen.  Only two physical axes vary: Q/K (K = 16 fixed,
Q in {4,8,12,16,24,32} => Q/K in {0.25,...,2}) and the U2U delivery
reliability s in {0.95,0.8,0.6,0.4,0.2} (uniform; it scales every g by
s -- the natural sensing-communication coupling).  Each grid cell:

- rho_I*  : the strongest information-load cut (submodular minimization
  + binary search, verified against brute force);
- rho_C   : the communication receive load b_tok(K-1)/Bbar_rx;
- FRIDS-v2 QoS: realized P_MD / P_FA / worst E1[T] at the policy-matched
  B protocol.

Classification: Red (rho_I* > 1: information-infeasible, no scheduler
can fix it), Yellow (rho_I* < 1 and rho_C < 1 but FRIDS violates QoS:
the true coordination gap), Green (feasible).  The feasibility
utilization Gamma(s) = lambda_FRIDS(s) / lambda_cut(s) tells whether the
scheduler sits at the achievable boundary (Gamma ~ 1: stop optimizing)
or has a real gap (Gamma << 1).

Writes ``results/feasibility_envelope_gate.json``.
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
    token_bits,
)
from uav_otfs_isac.feasibility import (
    communication_load,
    strongest_load_cut,
)
from uav_otfs_isac.frids import simulate_frids_v2

K_FIXED = 16
Q_GRID = (4, 8, 12, 16, 24, 32)
S_GRID = (0.95, 0.8, 0.6, 0.4, 0.2)
RX_BUDGET = 400.0   # per-UAV receive/decode budget (bits/cycle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/feasibility_envelope_gate.json")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=500)
    parser.add_argument("--b-delta", type=float, default=1.0)
    args = parser.parse_args()

    t0 = time.time()
    cells = {}
    rho_star_by_q = {}
    for q in Q_GRID:
        sc = build_distributed_scenario(np.random.default_rng(0),
                                        k_uavs=K_FIXED, q_targets=q)
        bt = calibrate_target_bounds(
            sc, args.alpha, args.beta, n_runs=300,
            seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
            verify_runs=args.calib_verify)
        bounds = [[bt[qq][0], bt[qq][1] - args.b_delta]
                  for qq in range(q)]
        owner = sc["owner_of"]
        b_tok = float(token_bits(q)["total"])
        rho_c = communication_load(K_FIXED, b_tok, RX_BUDGET)
        for s in S_GRID:
            u2u = np.full((K_FIXED, K_FIXED), float(s))
            np.fill_diagonal(u2u, 1.0)
            # advice/004 P0.5-7: rho_I* must be recomputed on the SAME
            # swept reliability, not on the raw scenario draw -- the cut
            # reads ``scenario["u2u_success"]``, so the swept matrix must
            # enter the scenario before ``strongest_load_cut`` (otherwise
            # ``rho_I*`` stays constant across the s sweep and the whole
            # information/communication phase split is dead).
            sc_s = dict(sc)
            sc_s["u2u_success"] = u2u
            rho_star = strongest_load_cut(sc_s, owner,
                                          horizon=args.max_steps,
                                          beta=args.beta,
                                          alpha=args.alpha)
            rho_star_by_q[(q, s)] = rho_star
            J, md, fa = [], [], []
            for seed in range(args.seeds):
                out = simulate_frids_v2(
                    sc, bounds, n_runs=args.n_runs,
                    seed=seed * 1000 + 7,
                    max_steps=args.max_steps,
                    delivery_matrix=u2u, s_for_g=u2u)
                J.append(out["worst_target_delay"])
                md.append(max(out["p_md"]))
                fa.append(max(out["p_fa"]))
            qos_ok = (max(md) <= args.beta + 0.02
                      and max(fa) <= args.alpha + 0.02)
            rho_i = rho_star["rho_star"]
            if rho_i > 1.0:
                zone = "red"
            elif qos_ok:
                zone = "green"
            else:
                zone = "yellow"
            cells[f"Q{q}_s{int(s * 100)}"] = {
                "Q": q, "K": K_FIXED, "Q_over_K": round(q / K_FIXED, 3),
                "s": s, "rho_I_star": round(rho_i, 4),
                "rho_C": round(rho_c, 3),
                "J": float(np.mean(J)),
                "p_md_max": float(max(md)),
                "p_fa_max": float(max(fa)),
                "qos_ok": bool(qos_ok),
                "zone": zone,
            }
            print(f"Q={q} (Q/K={q / K_FIXED:.2f}) s={s}: "
                  f"rho_I* {rho_i:.3f} rho_C {rho_c:.3f} "
                  f"J {np.mean(J):.2f} P_MD {max(md):.3f} "
                  f"-> {zone} ({time.time()-t0:.0f}s)", flush=True)

    # feasibility utilization Gamma(s): the largest Q/K with green QoS
    # vs the largest Q/K with rho_I* <= 1 (the cut never binds in this
    # family, so lambda_cut = the grid max; the QoS pattern is
    # non-monotone near the boundary, reported honestly)
    gamma = {}
    for s in S_GRID:
        green_qk = []
        cut_qk = []
        for q in Q_GRID:
            key = f"Q{q}_s{int(s * 100)}"
            c = cells[key]
            if c["zone"] == "green":
                green_qk.append(q / K_FIXED)
            if c["rho_I_star"] <= 1.0:
                cut_qk.append(q / K_FIXED)
        lam_f = max(green_qk) if green_qk else 0.0
        lam_c = max(cut_qk) if cut_qk else 0.0
        gamma[str(s)] = {
            "lambda_frids": lam_f,
            "lambda_cut": lam_c,
            "Gamma": round(lam_f / max(lam_c, 1e-12), 3),
            "green_cells": [f"Q{q}_s{int(s * 100)}" for q in Q_GRID
                            if cells[f"Q{q}_s{int(s * 100)}"]["zone"]
                            == "green"],
        }
    yellow = [k for k, c in cells.items() if c["zone"] == "yellow"]
    payload = {
        "gate": "f0g6-feasibility-envelope",
        "params": {
            "K": K_FIXED, "Q_grid": list(Q_GRID), "s_grid": list(S_GRID),
            "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "calib_seed": args.calib_seed,
            "calib_verify": args.calib_verify,
            "b_delta": args.b_delta,
            "rx_budget": RX_BUDGET,
            "frozen": ["FRIDS-v2", "fixed owner", "full mesh",
                       "scale-aware token", "policy-matched B",
                       "current kernels", "scenario seed 0"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "cells": cells,
        "rho_star_by_Q_s": {str((q, s)): rho_star_by_q[(q, s)]
                         for q in Q_GRID for s in S_GRID},
        "gamma": gamma,
        "summary": {
            "counts": {
                "green": sum(1 for c in cells.values()
                             if c["zone"] == "green"),
                "yellow": len(yellow),
                "red": sum(1 for c in cells.values()
                           if c["zone"] == "red"),
            },
            "yellow_cells": yellow,
            "coord_gap_exists": bool(yellow),
            "communication_limited": bool(
                any(c["rho_C"] > 1.0 for c in cells.values())),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("summary:", json.dumps(payload["summary"], indent=1))
    print("gamma:", json.dumps(gamma, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
