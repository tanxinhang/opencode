"""Gate F0-E: token fidelity audit (2026-08-17).

F0-D attributed the 16/8 decentralization gap partly to evidence
quantization.  This gate tests the token-fidelity lever with three
principled designs -- all inside the SAME 19-bit token (communication
principle: no protocol change, no extra traffic):

1. codebook redesign: Lloyd-Max (centroid condition = per-bin unbiased,
   H1-mass weighted), mu-law companding, and range matching, all at
   5-bit L_hat;
2. L_hat bit reallocation: the F0-A diagnostics proved the token fields
   u/r/chi (6 bits) are dead payload (the algorithm never reads them), so
   L_hat goes 5 -> 10 bits within the same budget (q2 + Lhat10 + intent2
   + stamp4 = 18 <= 19).

The decisive comparison is PAIRED at fixed thresholds (the F0-D style):
the codebook/rate effect on the belief path alone, without the
calibration operating-point noise.  Everything else is frozen
(normalized dual-G, eta = 1, fixed owner, full mesh, current kernels,
stopping rule, scenario gen).

Writes ``results/token_fidelity_gate.json``.  The expected verdict from
the F0-D methodology correction: the dec gap at 16/8 is delivery /
coordination-bound, NOT evidence-fidelity-bound -- the token-fidelity
designs recover at most ~1% of the worst-target delay.
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

from uav_otfs_isac.competition_audit import simulate_competition_audit
from uav_otfs_isac.distributed_audit import (
    build_distributed_scenario,
    build_target_values,
    build_token_quantizer,
    calibrate_target_bounds,
    mu_law_quantizer,
    uniform_quantizer,
)

SCALES = ((12, 6), (16, 8))


def eval_token(sc, q, bounds, singles, nu, quantizer, n_runs, seeds,
               max_steps, eta):
    J, md, fa, e1 = [], [], [], []
    for seed in range(seeds):
        out = simulate_competition_audit(
            sc, bounds, singles, nu, n_runs=n_runs,
            seed=seed * 1000 + 7, max_steps=max_steps,
            eta=eta, normalize_gains=True, quantizer=quantizer,
        )
        J.append(out["worst_target_delay"])
        md.append(max(out["p_md"]))
        fa.append(max(out["p_fa"]))
        e1.append(out["e1_delays"])
    return {"J": float(np.mean(J)), "p_md_max": float(np.max(md)),
            "p_fa_max": float(np.max(fa)),
            "e1": [float(np.mean([r[i] for r in e1])) for i in range(q)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/token_fidelity_gate.json")
    parser.add_argument("--n-runs", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    args = parser.parse_args()

    t0 = time.time()
    rows = {}
    for (k, q) in SCALES:
        rng = np.random.default_rng(args.scenario_seed)
        sc = build_distributed_scenario(rng, k_uavs=k, q_targets=q)
        nu = tuple([1.0 / q] * q)
        q5 = uniform_quantizer(bits=5, llr_range=6.0)
        q4 = uniform_quantizer(bits=5, llr_range=4.0)
        qm = build_token_quantizer(sc, weight="h1",
                                   per_target_equal=True,
                                   llr_range=4.0)
        qmulaw = mu_law_quantizer(mu=50.0, llr_range=4.0)
        q10 = uniform_quantizer(bits=10, llr_range=6.0)
        # frozen-threshold calibration (5-bit uniform token)
        b5 = calibrate_target_bounds(sc, args.alpha, args.beta,
                                     n_runs=300, seed=args.calib_seed,
                                     llr_bits=5, verify_runs=2000)
        s5 = build_target_values(sc, b5, horizon=args.max_steps, nu=nu)
        # 10-bit L_hat reallocation calibration
        b10 = calibrate_target_bounds(sc, args.alpha, args.beta,
                                      n_runs=300, seed=args.calib_seed,
                                      llr_bits=5, verify_runs=2000,
                                      quantizer=q10)
        s10 = build_target_values(sc, b10, horizon=args.max_steps, nu=nu)
        ev = lambda qm_: eval_token(sc, q, b5, s5, nu, qm_,
                                    args.n_runs, args.seeds,
                                    args.max_steps, args.eta)
        rows[f"{k}_{q}"] = {
            "k": k, "q": q,
            "same_thresholds": {
                "uniform5_r6": ev(None),
                "uniform5_r4": ev(q4),
                "lloyd5_r4": ev(qm),
                "mulaw5_r4": ev(qmulaw),
                "uniform10_r6": ev(q10),
            },
            "recalibrated": {
                "uniform5_r6": ev(None),
                "uniform10_r6": eval_token(
                    sc, q, b10, s10, nu, q10, args.n_runs, args.seeds,
                    args.max_steps, args.eta),
            },
            "token_layout": {
                "frozen": {"q": 2, "Lhat": 5, "u": 2, "r": 2, "chi": 2,
                           "intent": 2, "stamp": 4, "total": 19},
                "reallocated": {"q": 2, "Lhat": 10, "intent": 2,
                                "stamp": 4, "total": 18,
                                "dropped": ["u", "r", "chi"]},
            },
        }
        print(f"scale {k}_{q} done ({time.time()-t0:.0f}s)", flush=True)
    payload = {
        "gate": "f0e-token-fidelity",
        "params": {
            "scales": [list(s) for s in SCALES],
            "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "scenario_seed": args.scenario_seed,
            "calib_seed": args.calib_seed, "eta": args.eta,
            "frozen": ["fixed owner", "full mesh", "19-bit budget",
                       "dual-G + normalized coordination (eta=1)",
                       "calibrated two-threshold stopping", "current "
                       "scenario gen"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for key, row in payload["rows"].items():
        st = row["same_thresholds"]
        base = st["uniform5_r6"]["J"]
        print(f"scale {key} (same thresholds, base J {base:.3f}):")
        for name, v in st.items():
            print(f"  {name:<14} J {v['J']:.3f} (P_MD {v['p_md_max']:.3f})"
                  f"  delta {v['J']-base:+.3f}")
        r = row["recalibrated"]
        print(f"  recalibrated: uniform5 {r['uniform5_r6']['J']:.3f} -> "
              f"Lhat10 {r['uniform10_r6']['J']:.3f} "
              f"({r['uniform10_r6']['J']-r['uniform5_r6']['J']:+.3f})")


if __name__ == "__main__":
    main()
