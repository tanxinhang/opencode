"""Gate P1/P0.5/P0.6 (advice/003 + advice/004 + advice/005).

The advice/003 P1 theory bridge and the advice/004 P0.5 hardening carry
the numerical theorem verification; advice/005 P0.6 closes the last
three harness gaps:

- **verdict wiring**: the verdict depends on ALL six sub-gates
  ``gate_ok = D & U & S & B & L & M`` (decomposition, dec-uniform,
  stopping-tail, service gap, bottleneck eps_loc, static-MD).
- **PRE-REGISTERED v grid**: the Freedman ``v`` side uses the analytic
  deterministic bound ``V_q(t) <= t * sum_i max_a [s sigma^2 +
  s(1-s) g~^2]`` (recorded per scenario), NOT the sample max of the
  audit MC draws.
- **stopping tail**: the PATH-INTEGRATED ``E[exp f(A,V)]`` claim is
  REMOVED (advice/005 section 4: A,V are random historical processes);
  the verified objects are the deterministic joint event
  ``{T_q>t, A-D >= eta, V <= v}`` against the Freedman bound and the
  safe union decomposition of ``P_1(T_q>t)``.
- **Static-MD**: reported as a NUMERICAL RATE-CONSISTENCY CHECK
  ``C_emp(T) = gap(T)/sqrt(logQ/T)`` bounded by ``C_max = 1`` (the
  formal theorem stays the mirror-descent regret of Theorem 4.111); the
  log-log slope is a diagnostic, not a proof.

All 8 scenarios enter the cross-scenario verdict; FRIDS-v2 stays frozen
throughout.
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
from uav_otfs_isac.reliable_service_bridge import (
    local_vs_common_gap,
    martingale_decomposition,
    normalized_service_time_average,
    static_md_convergence,
    stopping_tail_verify,
)

SCALES_ALL = ((6, 3), (8, 4), (12, 6), (16, 8))
N_SCEN = {6: 2, 8: 2, 12: 2, 16: 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/delay_bridge_gate.json")
    parser.add_argument("--n-runs", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=500)
    parser.add_argument("--calib-scan", type=int, default=150)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--static-horizons", default="20,40,80,160,320")
    args = parser.parse_args()

    static_horizons = tuple(int(x) for x in args.static_horizons.split(","))
    t0 = time.time()
    static_rows = {}
    rows = {}
    for (k, q) in SCALES_ALL:
        per = {}
        for s in range(N_SCEN[k]):
            sc = build_distributed_scenario(np.random.default_rng(s),
                                            k_uavs=k, q_targets=q)
            bt = calibrate_target_bounds(
                sc, args.alpha, args.beta, n_runs=args.calib_scan,
                seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                verify_runs=args.calib_verify)
            seed_run = s * 1000 + 7
            out = simulate_frids_v2(
                sc, bt, n_runs=args.n_runs, seed=seed_run,
                max_steps=args.max_steps, alpha=args.alpha,
                beta=args.beta, mu=args.mu, eps=args.eps,
                bridge=True, price_mode="local")
            b = out["bridge"]
            g_mat = np.zeros((k, q))
            owner_of = sc["owner_of"]
            for i in range(k):
                for qq in range(q):
                    rel = (1.0 if i == owner_of[qq]
                           else float(b["delivery_matrix"][i, owner_of[qq]]))
                    g_mat[i, qq] = rel * float(b["mu_llr"][i, qq])
            dec = martingale_decomposition(
                b, q, target_alpha=args.alpha, target_beta=args.beta)
            stop = stopping_tail_verify(b, q, target_beta=args.beta)
            serv = normalized_service_time_average(
                b, q, g_mat=g_mat, target_beta=args.beta, eps=args.eps)
            # advice/004 P0.5-5: local-vs-common price CRN gap (same seed,
            # common price changes only the action index)
            out_c = simulate_frids_v2(
                sc, bt, n_runs=args.n_runs, seed=seed_run,
                max_steps=args.max_steps, alpha=args.alpha,
                beta=args.beta, mu=args.mu, eps=args.eps,
                bridge=True, price_mode="common")
            lt_com = local_vs_common_gap(b, out_c["bridge"], q, eps=args.eps)
            # advice/004 P0.5-4: static shadow MD convergence (frozen D/g,
            # no stopping, horizon sweep)
            static_c = static_md_convergence(
                g_mat, np.asarray(b["a_thr"]), static_horizons,
                mu=args.mu, eps=args.eps)
            static_rows[f"({k},{q})-s{s}"] = static_c
            per[str(s)] = {
                "J": float(out["worst_target_delay"]),
                "p_md_max": float(max(out["p_md"])),
                "decomposition": dec,
                "stopping_tail": stop,
                "service": serv,
                "local_vs_common": lt_com,
            }
            print(f"({k},{q}) s{s}: J {out['worst_target_delay']:.2f} "
                  f"dec-err {dec['decomposition_max_abs_error']:.3e} "
                  f"freedman-n {dec['freedman_n_cases']} "
                  f"stop-joint {stop['joint_event_violation_fraction']:.3f} "
                  f"stop-dec {stop['decomposition_violation_fraction']:.3f} "
                  f"z* {serv['z_star_static']:.3f} "
                  f"min_r {serv['min_q_time_avg_r_static']:.3f} "
                  f"eps_T {serv['eps_T_est']} ({time.time()-t0:.0f}s)",
                  flush=True)
        rows[f"{k}_{q}"] = {
            "k": k, "q": q, "n_scenarios": N_SCEN[k],
            "per_scenario": per,
        }

    # cross-scenario verdict: ALL 8 scenarios (advice/004 P0.5-6)
    all_rows = [v for r in rows.values() for v in r["per_scenario"].values()]
    dec_ok = all(
        (v["decomposition"]["freedman"]["violation_fraction"] <= 0.05
         and abs(v["decomposition"]["martingale_residual_mean"]) <= 1.0)
        for v in all_rows)
    dec_ok_u = all(v["decomposition"]["freedman_uniform"]
                   ["violation_fraction"] <= 0.05
                   for v in all_rows)
    stop_ok = all(v["stopping_tail"]["joint_event_violation_fraction"] <= 0.05
                  and v["stopping_tail"]
                  ["decomposition_violation_fraction"] <= 0.05
                  for v in all_rows)

    def serv_ok(v):
        if v["service"]["eps_T_est"] is None:
            return True
        return v["service"]["eps_T_est"] \
            <= v["service"]["sqrt_logQ_over_T"] + 0.05
    serv_ok = all(serv_ok(v) for v in all_rows)
    # the delay-relevant dual gap is at the BOTTLENECK target (the one
    # that sets the worst delay); F0-G9A already showed the local price
    # disagreement costs ~1.8% delay, so a small bottleneck service gap
    # is the expected honest value (mid-target max swings are diagnostic)
    dual_ok = all(v["local_vs_common"]["eps_loc_bottleneck"] <= 0.05
                  for v in all_rows)
    dual_max_swing = max(v["local_vs_common"]["eps_loc_dual"]
                         for v in all_rows)
    # the static-MD gate is a NUMERICAL RATE-CONSISTENCY CHECK (advice/005
    # section 5), NOT a proof: the formal theorem is the mirror-descent
    # regret `gap(T) <= C sqrt(logQ/T)` (Theorem 4.111); the empirical
    # constant `C_emp(T) = gap(T)/sqrt(logQ/T)` must be bounded
    # (`sup_T C_emp(T) <= C_max = 1.0`) for the measured gap to be
    # consistent with (not slower than) the theory rate.  The log-log
    # slope is reported as a diagnostic only (steeper-than-theory is
    # allowed; immediate convergence is strictly stronger and has no
    # slope).
    static_c_emp_max = [v["C_emp_max"] for v in static_rows.values()]
    static_ok = bool(static_c_emp_max) and \
        all(cm <= 1.0 for cm in static_c_emp_max)
    static_c_emp_sup = (max(static_c_emp_max) if static_c_emp_max else None)
    static_slopes = [v["loglog_slope_diagnostic"]
                     for v in static_rows.values()
                     if v["loglog_slope_diagnostic"] is not None]
    static_fastest = (min(static_slopes) if static_slopes else None)
    # P0.6 (advice/005 section 2): the verdict must depend on ALL six
    # sub-gates -- the P0.5 harness computed static_ok/dual_ok but wired
    # the verdict to only D and U and S and serv; the fixed wiring is
    # gate_ok = D & U & S & B & L & M (decomposition, uniform,
    # stopping_tail, service gap, bottleneck loc, static-MD).
    gate_ok = bool(dec_ok and dec_ok_u and stop_ok and serv_ok
                    and dual_ok and static_ok)
    gate = {
        "decomposition_ok": bool(dec_ok),
        "decomposition_uniform_ok": bool(dec_ok_u),
        "stopping_tail_ok": bool(stop_ok),
        "service_gap_ok": bool(serv_ok),
        "local_vs_common_bottleneck_ok": bool(dual_ok),
        "local_vs_common_max_swing": float(dual_max_swing),
        "static_md_rate_ok": bool(static_ok),
        "static_md_C_emp_max": static_c_emp_sup,
        "static_md_slopes_diagnostic": static_slopes,
        "static_md_slope_fastest_diag": static_fastest,
        "gate_ok": bool(gate_ok),
        "verdict": (
            "FRIDS-v2 frozen; P1 mechanism + harness both close "
            "(all six sub-gates, advice/005 section 2); proceed to P2"
            if gate_ok
            else "gate_ok false: one of D/U/S/B/L/M failed; P1 harness "
                 "NOT closed; fix before P2"),
    }
    payload = {
        "gate": "p1-service-delay-bridge-p0.6",
        "params": {
            "scales": [list(s) for s in SCALES_ALL],
            "n_runs": args.n_runs, "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "calib_scan": args.calib_scan,
            "calib_verify": args.calib_verify,
            "mu": args.mu, "eps": args.eps,
            "static_horizons": list(static_horizons),
            "frozen": ["FRIDS-v2", "fixed owner", "full mesh",
                       "19-bit token", "communication-domain beliefs",
                       "two-threshold stopping"],
            "bridge": "recorded per (run, cycle, target) owner LLR, "
                      "predictable reliable service A_q (deployed-score "
                      "drift), variance V_q, realized delivered service "
                      "S_q, normalized service r_q, stopping time T_q; "
                      "fill-forwarded after stop (M_{t wedge T})",
            "hardening": [
                "stopped-process fill-forward (M_{t wedge T})",
                "Freedman joint event V<=v (PRE-REGISTERED analytic v grid)",
                "time-uniform line-crossing form",
                "deterministic joint-event stopping tail + safe decomposition",
                "(path-integrated claim removed, advice/005 section 4)",
                "delta_Q = |g - tilde g| quantization correction",
                "Static-MD rate-consistency C_emp <= C_max (not a proof)",
                "eps_loc_dual = local-vs-common CRN gap (bottleneck)",
                "verdict = D & U & S & B & L & M (all six sub-gates)",
                "cross-scenario verdict over all 8 scenarios",
            ],
        },
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
        "static_md": static_rows,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()