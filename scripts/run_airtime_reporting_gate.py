"""Gate F0-G7: Physical-airtime reporting for FRIDS-v2 (advice/013).

The U2U ledger is upgraded from a bit-count abstraction to a waveform-
derived airtime constraint and exactly ONE decision variable is added:
the report/no-report gate ``z_i(t)`` (always-report -> report/no-report).

Three communication regimes by the always-report full-mesh receive-load
ratio ``rho_full`` (advice/013 section 4):

    rho_full ~ 0.5   non-congested
    rho_full ~ 1.0   critical
    rho_full > 1     congested (always-report is airtime-infeasible)

x five methods:

    always-report FRIDS-v2    the frozen mainline (overload thinned)
    random-drop               same communication volume, uniform
    periodic reporting        fixed low-overhead baseline
    value-triggered FRIDS     new method (report iff value > airtime price)
    central admission oracle  offline greedy reference (global values)

Life-or-death gate (advice/013 section 4):

- non-congested: the value-triggered config with the largest
  communication reduction subject to ``Delta J <= 2%`` must cut
  communication >= 30%;
- congested: value-triggered must improve ``J`` >= 5% over BOTH
  always-report and the volume-matched random-drop, with errors within
  ``beta + 2pp``.

The value-triggered price is ``lambda = lambda_base + lambda_dual`` with
``lambda_base`` the task-opportunity baseline (small grid swept
honestly) and ``lambda_dual`` the dual ascent on the locally-observed
receive load (cold-started from the local scarcity forecast).  Two value
definitions are evaluated: ``deficit`` (the strict joint-LP dual
``y*g/(D+eps)`` vs ``lambda*c_air`` of advice/013) and ``info`` (the
deficit-normalized price, which cancels ``D`` and compares the
min-max-weighted information ``y*g`` to the price -- Lemma 4.101).
Everything else is frozen (FRIDS-v2 index, mirror-descent prices,
policy-matched thresholds, compact token, fixed owner, full mesh,
scenario seed 0).

Writes ``results/airtime_reporting_gate.json``.
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
    token_bits,
)

RHO_GRID = (0.5, 1.0, 1.5)
LAMBDA_BASE_GRID = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6)
VALUE_MODES = ("deficit", "info")
MU_C = 0.05
EMA_RHO = 0.8


def eval_method(sim, sc, bounds, airtime, n_runs, seeds, max_steps, **kw):
    """Run a reporting method over the seed grid; aggregate the standard
    and communication metrics."""
    J, sd, md, fa = [], [], [], []
    comm = {key: [] for key in (
        "airtime_per_cycle", "tx_reports_per_uav", "rx_load_per_uav",
        "max_load_ratio", "budget_feasible_fraction",
        "thinned_tokens_per_cycle")}
    for seed in range(seeds):
        out = sim(sc, bounds, airtime, n_runs=n_runs, seed=seed * 1000 + 7,
                  max_steps=max_steps, **kw)
        J.append(out["worst_target_delay"])
        md.append(max(out["p_md"]))
        fa.append(max(out["p_fa"]))
        for key in comm:
            comm[key].append(out["comm"][key])
    return {
        "J": float(np.mean(J)),
        "J_sd": float(np.std(J, ddof=1)) if len(J) > 1 else 0.0,
        "p_md_max": float(max(md)),
        "p_fa_max": float(max(fa)),
        "comm": {key: float(np.mean(vals)) for key, vals in comm.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/airtime_reporting_gate.json")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=500)
    parser.add_argument("--b-delta", type=float, default=1.0)
    parser.add_argument("--bandwidth", type=float, default=1e6)
    parser.add_argument("--mu-c", type=float, default=MU_C)
    parser.add_argument("--ema-rho", type=float, default=EMA_RHO)
    args = parser.parse_args()

    t0 = time.time()
    k, q = args.k, args.q
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=k, q_targets=q)
    bt = calibrate_target_bounds(
        sc, args.alpha, args.beta, n_runs=300,
        seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
        verify_runs=args.calib_verify)
    bounds = [[bt[qq][0], bt[qq][1] - args.b_delta] for qq in range(q)]
    print(f"[G7] calibration done ({time.time() - t0:.0f}s)", flush=True)

    b_tok = float(token_bits(q)["total"])
    regimes = {}
    for rho_target in RHO_GRID:
        airtime = build_airtime_model(
            sc, bandwidth=args.bandwidth, rho_target=rho_target)
        kw = {"mu_c": args.mu_c, "ema_rho": args.ema_rho}
        cells = {}

        always = eval_method(simulate_frids_v2_air, sc, bounds, airtime,
                             args.n_runs, args.seeds, args.max_steps,
                             report_mode="always", **kw)
        cells["always_report"] = always

        value_modes = {}
        for value_mode in VALUE_MODES:
            value_rows = {}
            for lb in LAMBDA_BASE_GRID:
                row = eval_method(simulate_frids_v2_air, sc, bounds, airtime,
                                  args.n_runs, args.seeds, args.max_steps,
                                  report_mode="value", lambda_base=lb,
                                  value_mode=value_mode, **kw)
                value_rows[str(lb)] = row
            dJ = {lb: (value_rows[lb]["J"] - always["J"])
                  / max(always["J"], 1e-12) for lb in value_rows}
            reduction = {lb: 1.0 - value_rows[lb]["comm"]["airtime_per_cycle"]
                         / max(always["comm"]["airtime_per_cycle"], 1e-12)
                         for lb in value_rows}
            if rho_target <= 1.0:
                # non-congested / critical: max communication reduction
                # subject to Delta J <= 2%
                candidates = [lb for lb in value_rows if dJ[lb] <= 0.02]
                lb_sel = (max(candidates, key=lambda lb: reduction[lb])
                          if candidates
                          else min(value_rows, key=lambda lb: dJ[lb]))
            else:
                # congested: best worst-target delay
                lb_sel = min(value_rows, key=lambda lb: value_rows[lb]["J"])
            selected = value_rows[lb_sel]
            value_modes[value_mode] = {
                "selected_lambda_base": float(lb_sel),
                "configs": value_rows,
                "delta_J": float(dJ[lb_sel]),
                "airtime_reduction": float(reduction[lb_sel]),
            }
            print(f"    value_mode={value_mode} lb={lb_sel}: "
                  f"J {selected['J']:.2f} red {reduction[lb_sel]:.1%} "
                  f"dJ {dJ[lb_sel]:+.1%}", flush=True)
        cells["value_triggered"] = value_modes

        # volume-matched fair baselines use the best-J value config
        # (the congested-regime selection criterion)
        best_mode = min(
            VALUE_MODES,
            key=lambda vm: value_modes[vm]["configs"][
                str(value_modes[vm]["selected_lambda_base"])]["J"])
        selected = value_modes[best_mode]["configs"][
            str(value_modes[best_mode]["selected_lambda_base"])]
        p_vt = selected["comm"]["tx_reports_per_uav"]
        random = eval_method(simulate_frids_v2_air, sc, bounds, airtime,
                             args.n_runs, args.seeds, args.max_steps,
                             report_mode="random", p=float(np.clip(p_vt, 1e-6, 1.0)),
                             **kw)
        period = max(1, int(round(1.0 / max(p_vt, 1e-6))))
        periodic = eval_method(simulate_frids_v2_air, sc, bounds, airtime,
                               args.n_runs, args.seeds, args.max_steps,
                               report_mode="periodic", period=period, **kw)
        oracle = eval_method(simulate_frids_v2_air, sc, bounds, airtime,
                             args.n_runs, args.seeds, args.max_steps,
                             report_mode="oracle", **kw)

        cells["random_drop"] = {"p": float(p_vt), **random}
        cells["periodic"] = {"period": period, **periodic}
        cells["oracle"] = oracle
        lb_sel = value_modes[best_mode]["selected_lambda_base"]
        selected = value_modes[best_mode]["configs"][str(lb_sel)]
        reduction = value_modes[best_mode]["airtime_reduction"]

        regimes[str(rho_target)] = {
            "rho_target": float(rho_target),
            "rho_full": float(airtime["rho_full"]),
            "T_air_s": float(airtime["t_air"]),
            "b_tok": b_tok,
            "cells": cells,
        }
        print(f"[G7] rho={rho_target}: rho_full "
              f"{airtime['rho_full']:.3f} | always J {always['J']:.2f} "
              f"(P_MD {always['p_md_max']:.3f}) | value[{best_mode}] "
              f"(lb={lb_sel}) J {selected['J']:.2f} (P_MD "
              f"{selected['p_md_max']:.3f}) red {reduction:.1%} | "
              f"random J {random['J']:.2f} | periodic J {periodic['J']:.2f} "
              f"| oracle J {oracle['J']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- life-or-death gate (evaluated per value mode) -----------------
    non_cong = regimes["0.5"]
    cong = regimes["1.5"]
    crit = regimes["1.0"]

    gate_by_mode = {}
    for value_mode in VALUE_MODES:
        # non-congested: max reduction subject to dJ <= 2%, then >= 30%
        vt_nc = non_cong["cells"]["value_triggered"][value_mode]
        red_nc = vt_nc["airtime_reduction"]
        dj_nc = vt_nc["delta_J"]
        errors_nc = (
            vt_nc["configs"][str(vt_nc["selected_lambda_base"])]["p_md_max"]
            <= args.beta + 0.02)
        gate_nc = {
            "delta_J": float(dj_nc),
            "airtime_reduction": float(red_nc),
            "delta_J_ok": bool(dj_nc <= 0.02),
            "reduction_ok": bool(red_nc >= 0.30),
            "errors_ok": bool(errors_nc),
            "pass": bool(dj_nc <= 0.02 and red_nc >= 0.30 and errors_nc),
        }

        # congested: J improvement >= 5% over always-report AND matched
        # random-drop
        vt_c = cong["cells"]["value_triggered"][value_mode]
        vj = vt_c["configs"][str(vt_c["selected_lambda_base"])]["J"]
        aj = cong["cells"]["always_report"]["J"]
        rj = cong["cells"]["random_drop"]["J"]
        dj_always = (aj - vj) / max(aj, 1e-12)
        dj_random = (rj - vj) / max(rj, 1e-12)
        errors_cong = (
            vt_c["configs"][str(vt_c["selected_lambda_base"])]["p_md_max"]
            <= args.beta + 0.02)
        gate_cong = {
            "value_J": float(vj),
            "always_J": float(aj),
            "random_J": float(rj),
            "delta_J_vs_always": float(dj_always),
            "delta_J_vs_random": float(dj_random),
            "delta_J_ok": bool(dj_always >= 0.05 and dj_random >= 0.05),
            "errors_ok": bool(errors_cong),
            "pass": bool(dj_always >= 0.05 and dj_random >= 0.05
                         and errors_cong),
        }

        # critical regime (report only)
        vt_k = crit["cells"]["value_triggered"][value_mode]
        vk = vt_k["configs"][str(vt_k["selected_lambda_base"])]["J"]
        ak = crit["cells"]["always_report"]["J"]
        gate_k = {
            "value_J": float(vk),
            "always_J": float(ak),
            "delta_J": float((ak - vk) / max(ak, 1e-12)),
            "airtime_reduction": float(vt_k["airtime_reduction"]),
        }
        gate_by_mode[value_mode] = {
            "non_congested": gate_nc,
            "critical": gate_k,
            "congested": gate_cong,
        }

    best_mode = max(VALUE_MODES,
                    key=lambda vm: int(gate_by_mode[vm]["non_congested"]["pass"])
                    + int(gate_by_mode[vm]["congested"]["pass"]))
    best = gate_by_mode[best_mode]
    adopt = bool(best["non_congested"]["pass"] and best["congested"]["pass"])
    gate = {
        "best_value_mode": best_mode,
        "per_mode": gate_by_mode,
        "adopt": adopt,
        "verdict": (
            "value-triggered airtime reporting adopted (both gates pass)"
            if adopt
            else (
                "adopted with caution (one gate passed)"
                if best["non_congested"]["pass"] or best["congested"]["pass"]
                else "rejected; close the communication-admission direction")),
    }

    payload = {
        "gate": "f0g7-physical-airtime-reporting",
        "params": {
            "K": k, "Q": q, "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps, "alpha": args.alpha,
            "beta": args.beta, "calib_seed": args.calib_seed,
            "calib_verify": args.calib_verify, "b_delta": args.b_delta,
            "bandwidth_hz": args.bandwidth,
            "rho_grid": list(RHO_GRID),
            "lambda_base_grid": list(LAMBDA_BASE_GRID),
            "value_modes": list(VALUE_MODES),
            "mu_c": args.mu_c, "ema_rho": args.ema_rho,
            "frozen": ["FRIDS-v2 index/mirror-descent", "policy-matched B",
                       "compact scale-aware token", "fixed owner",
                       "full mesh", "communication-domain beliefs",
                       "current scenario gen (seed 0)"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "airtime_model": {
            "rate_model": "W_c log2(1 + gamma)  (Shannon capacity UPPER "
                          "BOUND, not claimed throughput)",
            "snr_model": "inverse log-normal outage: "
                         "SNR = threshold + sigma * Phi^-1(s_ij)",
            "tau": "b_tok / R_ij",
            "budget": "T_air s/cycle; overflow thinning "
                      "survival min(1, T_air / L_i)",
        },
        "regimes": regimes,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()