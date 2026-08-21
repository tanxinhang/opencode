"""Deep audit A (v2): F0-G7 physical-airtime gate robustness.

The F0-G7 verdict (adopted with caution) was measured on scenario 0 only
and its headline numbers (52.8% cut at Delta J <= 2% non-congested;
congested info improvement) were the SELECTED lambda_base for that draw.
This audit runs the FULL gate selection rule per scenario draw over
lambda_base and reports, for each decisive regime, whether the life-gate
conditions hold OUT-OF-SAMPLE under the gate's own selection:

- non-congested (rho ~ 0.5): exists a lambda_base with Delta J <= 2%
  AND airtime reduction >= 30%?
- congested (rho ~ 1.5): does the best-J value config improve J >= 5%
  over always-report and the volume-matched random-drop?

A claim that requires per-scenario lambda_base re-selection is a real,
if fragile, mechanism; a claim that fails even with per-scenario
selection is not supported.  Writes ``results/deep_audit_g7.json``.
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

from uav_otfs_isac.airtime import build_airtime_model, simulate_frids_v2_air
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)

RHO_GRID = (0.5, 1.5)
LAMBDA_BASE_GRID = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6)
VALUE_MODES = ("deficit", "info")
MU_C = 0.05
EMA_RHO = 0.8


def eval_row(sc, bounds, airtime, n_runs, seeds, max_steps, mode,
             value_mode, lb, p=1.0):
    kw = {"mu_c": MU_C, "ema_rho": EMA_RHO, "value_mode": value_mode}
    if mode == "value":
        kw["lambda_base"] = lb
    if mode == "random":
        kw["p"] = p
    J, md = [], []
    comm = {key: [] for key in ("airtime_per_cycle", "tx_reports_per_uav")}
    for seed in range(seeds):
        out = simulate_frids_v2_air(sc, bounds, airtime, n_runs=n_runs,
                                    seed=seed * 1000 + 7, max_steps=max_steps,
                                    report_mode=mode, **kw)
        J.append(out["worst_target_delay"])
        md.append(max(out["p_md"]))
        for key in comm:
            comm[key].append(out["comm"][key])
    return {"J": float(np.mean(J)), "p_md_max": float(max(md)),
            "comm": {key: float(np.mean(v)) for key, v in comm.items()}}


def run_regime(sc, bounds, rho_target, args):
    am = build_airtime_model(sc, rho_target=rho_target)
    always = eval_row(sc, bounds, am, args.n_runs, args.seeds,
                      args.max_steps, "always", "deficit", 0.0)
    out = {}
    for value_mode in VALUE_MODES:
        best = None
        for lb in LAMBDA_BASE_GRID:
            row = eval_row(sc, bounds, am, args.n_runs, args.seeds,
                           args.max_steps, "value", value_mode, lb)
            if best is None or row["J"] < best["J"]:
                best = {"lb": lb, **row}
        out[value_mode] = best
    return always, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/deep_audit_g7.json")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=400)
    parser.add_argument("--b-delta", type=float, default=1.0)
    parser.add_argument("--scenario-seeds", type=int, default=3)
    args = parser.parse_args()

    t0 = time.time()
    k, q = args.k, args.q
    per_scenario = {}
    for s in range(args.scenario_seeds):
        sc = build_distributed_scenario(np.random.default_rng(s),
                                        k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(sc, args.alpha, args.beta, n_runs=300,
                                     seed=args.calib_seed,
                                     llr_bits=TOKEN_LLR_BITS,
                                     verify_runs=args.calib_verify)
        bounds = [[bt[qq][0], bt[qq][1] - args.b_delta] for qq in range(q)]

        # non-congested: per value mode, the max reduction subject to
        # Delta J <= 2%
        am05 = build_airtime_model(sc, rho_target=0.5)
        always05 = eval_row(sc, bounds, am05, args.n_runs, args.seeds,
                            args.max_steps, "always", "deficit", 0.0)
        nc = {}
        for value_mode in VALUE_MODES:
            best = None
            for lb in LAMBDA_BASE_GRID:
                row = eval_row(sc, bounds, am05, args.n_runs, args.seeds,
                               args.max_steps, "value", value_mode, lb)
                dJ = (row["J"] - always05["J"]) / max(always05["J"], 1e-12)
                red = 1.0 - row["comm"]["airtime_per_cycle"] / max(
                    always05["comm"]["airtime_per_cycle"], 1e-12)
                if dJ <= 0.02:
                    if best is None or red > best["red"]:
                        best = {"lb": lb, "dJ": float(dJ), "red": float(red),
                                "p_md": row["p_md_max"]}
            nc[value_mode] = best
        nc_ok = any(b is not None and b["red"] >= 0.30
                    and b["p_md"] <= args.beta + 0.02
                    for b in nc.values())

        # congested: best-J value vs always-report and matched random
        am15 = build_airtime_model(sc, rho_target=1.5)
        always15 = eval_row(sc, bounds, am15, args.n_runs, args.seeds,
                            args.max_steps, "always", "deficit", 0.0)
        cong = {}
        for value_mode in VALUE_MODES:
            best = None
            for lb in LAMBDA_BASE_GRID:
                row = eval_row(sc, bounds, am15, args.n_runs, args.seeds,
                               args.max_steps, "value", value_mode, lb)
                if best is None or row["J"] < best["J"]:
                    best = {"lb": lb, **row}
            # volume-matched random
            p = best["comm"]["tx_reports_per_uav"]
            rnd = eval_row(sc, bounds, am15, args.n_runs, args.seeds,
                           args.max_steps, "random", value_mode, 0.0,
                           p=float(np.clip(p, 1e-6, 1.0)))
            cong[value_mode] = {
                "lb": best["lb"], "J": best["J"],
                "always_J": always15["J"], "random_J": rnd["J"],
                "dJ_vs_always": float((always15["J"] - best["J"])
                                      / max(always15["J"], 1e-12)),
                "dJ_vs_random": float((rnd["J"] - best["J"])
                                      / max(rnd["J"], 1e-12)),
                "p_md": best["p_md_max"],
            }
        cong_ok = any(c["dJ_vs_always"] >= 0.05 and c["dJ_vs_random"] >= 0.05
                      and c["p_md"] <= args.beta + 0.02 for c in cong.values())

        per_scenario[str(s)] = {
            "non_congested": nc,
            "non_congested_life_gate": bool(nc_ok),
            "congested": cong,
            "congested_life_gate": bool(cong_ok),
        }
        nc_str = ", ".join(
            f"{vm}: dJ {nc[vm]['dJ']:+.1%} red {nc[vm]['red']:.1%}"
            if nc[vm] else f"{vm}: none"
            for vm in VALUE_MODES)
        print(f"  scenario {s}: non-cong OK {nc_ok} ({nc_str}) | "
              f"congested OK {cong_ok}", flush=True)

    summary = {
        "non_congested_passes": {s: per_scenario[s]["non_congested_life_gate"]
                                 for s in per_scenario},
        "congested_passes": {s: per_scenario[s]["congested_life_gate"]
                             for s in per_scenario},
        "non_congested_fraction": float(np.mean(
            [per_scenario[s]["non_congested_life_gate"]
             for s in per_scenario])),
        "congested_fraction": float(np.mean(
            [per_scenario[s]["congested_life_gate"] for s in per_scenario])),
    }
    payload = {
        "audit": "deep-audit-A-v2-g7-robustness",
        "params": {"K": k, "Q": q, "n_runs": args.n_runs,
                   "seeds": args.seeds, "max_steps": args.max_steps,
                   "alpha": args.alpha, "beta": args.beta,
                   "calib_seed": args.calib_seed,
                   "calib_verify": args.calib_verify,
                   "b_delta": args.b_delta,
                   "scenario_seeds": args.scenario_seeds,
                   "rho_grid": list(RHO_GRID),
                   "lambda_base_grid": list(LAMBDA_BASE_GRID),
                   "value_modes": list(VALUE_MODES),
                   "mu_c": MU_C, "ema_rho": EMA_RHO},
        "runtime_s": round(time.time() - t0, 1),
        "per_scenario": per_scenario,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("summary:", json.dumps(summary, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()