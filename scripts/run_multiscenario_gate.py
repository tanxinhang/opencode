"""Gate F0-G2: multi-scenario statistics (advice/009).

The scaling claim rests on a single scenario draw per scale.  This gate
runs the current best config (normalized dual-G + scale-adaptive price,
reference calibrated thresholds) over N scenario seeds at the critical
scales and decomposes the J variance into the scenario component and the
Monte-Carlo component:

    Var(J) = Var_scenario(E[J | scenario]) + E_scenario[Var(J | scenario]).

It answers whether the 16/8 degradation survives across scenario draws.
Writes ``results/multiscenario_gate.json``.
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
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
)

SCALES = ((12, 6), (16, 8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/multiscenario_gate.json")
    parser.add_argument("--n-scenarios", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    args = parser.parse_args()

    t0 = time.time()
    rows = {}
    for (k, q) in SCALES:
        eta_use = 1.0 if k <= 12 else 0.0
        scenario_J = []          # mean J per scenario
        scenario_pmd = []
        within_var = []          # MC variance per scenario
        per_scenario = {}
        for s in range(args.n_scenarios):
            sc = build_distributed_scenario(
                np.random.default_rng(s), k_uavs=k, q_targets=q)
            nu = tuple([1.0 / q] * q)
            bt = calibrate_target_bounds(
                sc, args.alpha, args.beta, n_runs=300,
                seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                verify_runs=args.calib_verify)
            singles = build_target_values(sc, bt,
                                          horizon=args.max_steps, nu=nu)
            J, md = [], []
            for seed in range(args.seeds):
                out = simulate_competition_audit(
                    sc, bt, singles, nu, n_runs=args.n_runs,
                    seed=seed * 1000 + 7, max_steps=args.max_steps,
                    eta=eta_use, normalize_gains=True)
                J.append(out["worst_target_delay"])
                md.append(max(out["p_md"]))
            scenario_J.append(float(np.mean(J)))
            scenario_pmd.append(float(np.max(md)))
            within_var.append(float(np.var(J, ddof=1)))
            per_scenario[str(s)] = {
                "J": float(np.mean(J)),
                "J_per_seed": [round(v, 3) for v in J],
                "p_md_max": float(np.max(md)),
            }
            print(f"({k},{q}) scenario {s}: J {np.mean(J):.3f} "
                  f"P_MD {np.max(md):.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        sc_mean = float(np.mean(scenario_J))
        var_scenario = float(np.var(scenario_J, ddof=1))
        var_mc = float(np.mean(within_var))
        rows[f"{k}_{q}"] = {
            "k": k, "q": q, "eta_used": eta_use,
            "n_scenarios": args.n_scenarios,
            "mean_J": sc_mean,
            "sd_J": float(np.std(scenario_J, ddof=1)),
            "min_J": float(np.min(scenario_J)),
            "max_J": float(np.max(scenario_J)),
            "var_scenario": var_scenario,
            "var_mc": var_mc,
            "var_total_share_scenario": float(
                var_scenario / max(var_scenario + var_mc, 1e-12)),
            "p_md_max_across": float(np.max(scenario_pmd)),
            "per_scenario": per_scenario,
        }
    # cross-scale comparison: is 16/8 worse than 12/6 across scenarios?
    a = rows["12_6"]
    b = rows["16_8"]
    j12 = [v["J"] for v in a["per_scenario"].values()]
    j16 = [v["J"] for v in b["per_scenario"].values()]
    win_rate = float(np.mean([x < y for x, y in zip(j16, j12)]))
    growth = (b["mean_J"] - a["mean_J"]) / max(a["mean_J"], 1e-12)
    rows["_cross_scale"] = {
        "mean_J_12_6": a["mean_J"],
        "mean_J_16_8": b["mean_J"],
        "relative_growth": float(growth),
        "scenario_win_rate_16_better": win_rate,
        "growth_is_robust": bool(
            growth > 0 and all(x > y for x, y in zip(j16, j12))),
    }
    payload = {
        "gate": "f0g2-multiscenario",
        "params": {
            "scales": [list(s) for s in SCALES],
            "n_scenarios": args.n_scenarios,
            "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "calib_seed": args.calib_seed,
            "calib_verify": args.calib_verify,
            "frozen": ["fixed owner", "full mesh", "19-bit token",
                       "normalized dual-G + scale-adaptive price",
                       "reference calibrated thresholds"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("cross-scale:", json.dumps(rows["_cross_scale"], indent=1))
    for key in ("12_6", "16_8"):
        r = rows[key]
        print(f"{key}: mean J {r['mean_J']:.3f} +/- {r['sd_J']:.3f} "
              f"[{r['min_J']:.2f}, {r['max_J']:.2f}]  "
              f"var_scenario {r['var_scenario']:.4f} "
              f"var_mc {r['var_mc']:.4f} "
              f"(scenario share {r['var_total_share_scenario']:.0%})")
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
