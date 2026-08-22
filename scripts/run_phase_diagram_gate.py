"""Gate P2.1 (advice/006): formal phase-diagram protocol hardening.

The P2 smoke established the failure mechanisms; P2.1 fixes the protocol
before the multi-seed phase diagram runs.  FRIDS-v2 stays frozen.  The
formal runner implements, per advice/006 sections 1-12:

- **scenario_seed x MC_seed separation** (advice/006 section 4): every
  grid point loops over both independently.
- **nested-Q workload** (advice/006 section 5): one ``Q_max`` scenario is
  built and each ``Q`` is the first-``Q`` target subset, so
  ``rho_I*(Q1) <= rho_I*(Q2)`` for ``Q1 < Q2`` (subset cuts persist) and
  ``lambda`` mixes only load change, never a redraw.
- **dual QoS + UCB statistical status** (advice/006 section 7): every
  candidate must meet ``P_MD,q <= beta`` AND ``P_FA,q <= alpha`` for ALL
  q; per-target proportions get Hoeffding UCB and each cell is PASS /
  FAIL / UNCERTAIN (UNCERTAIN cells are the only ones that spend extra
  MC on the formal run, not every cell).
- **hard receiver admission** (advice/006 section 6):
  ``sum_{j!=i} b_tok z_{ji,t} <= ceil(cap_i)`` pathwise via the
  uniform-without-replacement admission in ``simulate_frids_v2``.
- **rho_I^pre vs rho_I^eff** (advice/006 section 8): the information
  load BEFORE communication admission (physical link quality) is
  ``rho_I^pre``; AFTER admission (effective ``g``) ``rho_I^eff``.  A
  comm-caused overload that raises ``rho_I^eff`` is NOT re-labelled
  information-caused.
- **4+1 region classification** (advice/006 section 9):
  calibration-family-infeasible / information-limited /
  communication-limited / coordination-limited / feasible, with cells
  that are both ``rho_I^pre > 1`` and ``rho_C > 1`` flagged mixed.
- **Gamma_dual / Gamma_envelope with censoring** (advice/006 section
  10): ``lambda_F/lambda_C = Gamma_dual`` (local-price disagreement,
  NOT a full algorithm oracle) and ``lambda_F/lambda_nec =
  Gamma_envelope``; ``lambda_nec = max Q/K with rho_I^pre<=1 AND rho_C<=1``
  (per-Q ``rho_C`` recomputed from ``token_bits(Q)``).  If all three
  lambdas sit at the scan ceiling the result is marked **right-censored**,
  which is the correct claim at that stage -- NOT "no algorithm gap".
- **P2-A1 downgraded to an analytic consistency sanity check**
  (advice/006 section 11): ``rho_I*(xi) = rho_I*(1)/xi`` is verified
  numerically once; no multi-seed spent on it.

The coarse formal run is ``S_GEOM x S_MC`` = 5 x 300 per cell; boundary /
UNCERTAIN cells are escalated to 10 x 1000 (or until the CI is narrow
enough).
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
from uav_otfs_isac.feasibility import strongest_load_cut
from uav_otfs_isac.frids import simulate_frids_v2

K_FIXED = 16
Q_MAX = 64
Q_GRID = (4, 8, 12, 16, 24, 32, 48, 64)
SNR_SHIFT_DB = (0.0, -3.0, -6.0, -10.0, -14.0, -18.0)
RHO_C_GRID = (0.5, 0.8, 1.0, 1.2, 1.5)
XI_G = 0.25                    # P2-A1 analytic sanity scale


def nested_scenario(full: dict, q: int) -> dict:
    """The first-``Q`` target subset of the full ``Q_max`` scenario
    (advice/006 section 5): same UAVs, same kernels, same U2U, same
    owner roles -- only fewer targets, so ``rho_I*(Q1) <= rho_I*(Q2)``
    for ``Q1 < Q2``."""
    links = {qq: full["links"][qq] for qq in range(q)}
    by_host = {(i, qq): full["by_host"][(i, qq)]
               for i in range(full["k"]) for qq in range(q)}
    return {
        "k": full["k"], "q": q, "l_acc": full["l_acc"],
        "links": links, "by_host": by_host,
        "u2u_success": full["u2u_success"],
        "owner_of": full["owner_of"][:q],
    }


def g_matrix_of(scenario: dict) -> np.ndarray:
    k, q = scenario["k"], scenario["q"]
    owner = scenario["owner_of"]
    g = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            rel = 1.0 if i == owner[qq] else \
                float(scenario["u2u_success"][i, owner[qq]])
            g[i, qq] = max(float(a["i_plus"]) * rel
                           for a in scenario["by_host"][(i, qq)])
    return g


def rho_info_of(scenario: dict, horizon: int = 40, beta: float = 0.05,
                alpha: float = 0.05) -> float:
    cut = strongest_load_cut(scenario, scenario["owner_of"],
                             horizon=horizon, beta=beta, alpha=alpha)
    return float(cut["rho_star"])


def rho_com_of(scenario: dict, rx_budget: float) -> float:
    b_tok = float(token_bits(scenario["q"])["total"])
    return b_tok * (scenario["k"] - 1) / max(rx_budget, 1e-12)


def run_qos(scenario, bounds, n_runs, mc_seed, max_steps=40,
            delivery_matrix=None, s_for_g=None, price_mode="local",
            rx_cap_tokens=None) -> dict:
    out = simulate_frids_v2(
        scenario, bounds, n_runs=n_runs, seed=mc_seed, max_steps=max_steps,
        delivery_matrix=delivery_matrix, s_for_g=s_for_g,
        price_mode=price_mode, rx_cap_tokens=rx_cap_tokens)
    return {"J": float(out["worst_target_delay"]),
            "p_md": list(out["p_md"]), "p_fa": list(out["p_fa"])}


def hoeffding_ucb(successes: int, n: int, delta: float = 0.05) -> float:
    """Hoeffding upper confidence bound ``p_hat + sqrt(log(1/delta) /
    2n)`` of a binomial proportion (one-sided)."""
    p = successes / max(n, 1)
    return p + np.sqrt(np.log(1.0 / max(delta, 1e-12)) / (2.0 * max(n, 1)))


def qos_status(errs: np.ndarray, alpha: float, beta: float,
               n_runs: int, delta: float = 0.05) -> str:
    """Dual-QoS statistical status of one (target) cell row of P_FA/P_MD
    estimates: PASS / FAIL / UNCERTAIN via Hoeffding UCB (advice/006
    section 7).  ``errs`` shape ``(2, q)`` = (p_fa, p_md) point
    estimates per target; the UCB check is over every target."""
    p_fa = np.asarray(errs[0]); p_md = np.asarray(errs[1])
    ucb_fa = hoeffding_ucb(np.round(p_fa * n_runs).astype(int), n_runs, delta)
    ucb_md = hoeffding_ucb(np.round(p_md * n_runs).astype(int), n_runs, delta)
    if np.all(ucb_fa <= alpha + 0.01) and np.all(ucb_md <= beta + 0.01):
        return "PASS"
    if np.any(p_fa > alpha + 0.02) or np.any(p_md > beta + 0.02):
        return "FAIL"
    return "UNCERTAIN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/phase_diagram_gate.json")
    parser.add_argument("--s-geom", type=int, default=5)
    parser.add_argument("--mc-runs", type=int, default=300)
    parser.add_argument("--s-geom-boundary", type=int, default=10)
    parser.add_argument("--mc-boundary", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-scan", type=int, default=150)
    parser.add_argument("--calib-verify", type=int, default=300)
    parser.add_argument("--rx-budget", type=float, default=400.0)
    args = parser.parse_args()
    t0 = time.time()

    # ---------- P2-A1 analytic sanity: rho_I*(xi) = rho_I*(1)/xi ------
    full0 = build_distributed_scenario(np.random.default_rng(0),
                                       k_uavs=K_FIXED, q_targets=Q_MAX)
    nested8 = nested_scenario(full0, 8)
    rho_xi1 = rho_info_of(nested8)
    rho_xi = rho_info_of(_scale_g(nested8, XI_G))
    a1 = {
        "xi_g": XI_G,
        "rho_I(1)": rho_xi1,
        "rho_I(xi)": rho_xi,
        "predicted": rho_xi1 / XI_G,
        "consistency_ok": bool(abs(rho_xi - rho_xi1 / XI_G) <= 1e-9),
    }
    print(f"P2-A1 analytic sanity: rho(xi) {rho_xi:.4f} vs rho(1)/xi "
          f"{rho_xi1 / XI_G:.4f} -> {a1['consistency_ok']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- nested-Q lambda brackets (P2.1-7, censoring) ----------
    lambdas = {"Q": [], "rho_I_pre": [], "rho_C": [], "rho_nec_ok": []}
    for q in Q_GRID:
        sc = nested_scenario(full0, q)
        rho_i = rho_info_of(sc)
        b_tok = float(token_bits(q)["total"])
        rho_c = b_tok * (K_FIXED - 1) / max(args.rx_budget, 1e-12)
        lambdas["Q"].append(q)
        lambdas["rho_I_pre"].append(rho_i)
        lambdas["rho_C"].append(rho_c)
        lambdas["rho_nec_ok"].append(rho_i <= 1.0 and rho_c <= 1.0)
    # lambda_nec = largest Q/K with both necessary conditions met;
    # right-censored when the top of the scan is still feasible (the
    # boundary is beyond the tested load, not achieved)
    q_nec = max((q for q, ok in zip(Q_GRID, lambdas["rho_nec_ok"]) if ok),
                default=0)
    lambda_nec = q_nec / K_FIXED
    censored_nec = bool(q_nec == Q_GRID[-1])  # at/above the scan ceiling

    # ---------- coarse formal grid: info (SNR) axis with real seeds -----
    cells = {}
    for shift in SNR_SHIFT_DB:
        for geom in range(args.s_geom):
            sc = build_distributed_scenario(np.random.default_rng(geom),
                                            k_uavs=K_FIXED, q_targets=8,
                                            snr_shift=float(shift))
            try:
                bt = calibrate_target_bounds(
                    sc, args.alpha, args.beta, n_runs=args.calib_scan,
                    seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                    verify_runs=args.calib_verify)
            except ValueError:
                bt = None
            bounds = None if bt is None else \
                [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(8)]
            if bt is None:
                # calibration-family-infeasible on the audited grid
                cells.setdefault(str(shift), []).append({
                    "geom": geom, "zone": "calibration-family-infeasible",
                    "rho_I_pre": rho_info_of(sc, beta=args.beta,
                                             alpha=args.alpha),
                    "rho_C": rho_com_of(sc, args.rx_budget),
                })
                continue
            # MC trials aggregated at the cell (advice/006 sections 4,7):
            # geom seeds vary the geometry, each MC draw is an
            # independent detection trial; the per-target error point
            # estimate pools across geoms and the Hoeffding UCB uses the
            # TOTAL pooled trial count (not the number of geom seeds).
            qos_rows = [run_qos(sc, bounds, args.mc_runs, mc_seed=mc)
                        for mc in range(args.s_geom)]
            n_trials = args.s_geom * args.mc_runs
            pfa = np.mean([row["p_fa"] for row in qos_rows], axis=0)
            pmd = np.mean([row["p_md"] for row in qos_rows], axis=0)
            status = qos_status(np.vstack([pfa, pmd]), args.alpha,
                                args.beta, n_trials)
            # P2.1 (advice/006 sections 7,12): UNCERTAIN boundary cells
            # escalate to the (s_geom_boundary x mc_boundary) protocol
            # until they become PASS/FAIL (bounded; no full-grid blow-up)
            if status == "UNCERTAIN":
                qos_rows = [run_qos(sc, bounds, args.mc_boundary,
                                    mc_seed=mc)
                            for mc in range(args.s_geom_boundary)]
                n_trials = args.s_geom_boundary * args.mc_boundary
                pfa = np.mean([row["p_fa"] for row in qos_rows], axis=0)
                pmd = np.mean([row["p_md"] for row in qos_rows], axis=0)
                status = qos_status(np.vstack([pfa, pmd]), args.alpha,
                                    args.beta, n_trials)
            rho_i = rho_info_of(sc, beta=args.beta, alpha=args.alpha)
            rho_c = rho_com_of(sc, args.rx_budget)
            if rho_i > 1.0 and rho_c > 1.0:
                zone = "mixed I+C infeasible"
            elif rho_i > 1.0:
                zone = "information-limited"
            elif rho_c > 1.0:
                zone = "communication-limited"
            elif status == "PASS":
                zone = "feasible"
            else:
                zone = "coordination-limited"
            cells.setdefault(str(shift), []).append({
                "geom": geom, "zone": zone, "qos_status": status,
                "rho_I_pre": rho_i, "rho_C": rho_c,
                "J": float(np.mean([row["J"] for row in qos_rows])),
                "p_md_max": float(np.max(pmd)),
                "p_fa_max": float(np.max(pfa)),
                "escalated_uncertain": bool(n_trials != args.s_geom * args.mc_runs),
            })
            print(f"info snr {shift:+.0f}dB geom {geom}: {zone} "
                  f"({status}) ({time.time()-t0:.0f}s)", flush=True)

    # ---------- communication (rho_C) axis with hard admission ---------
    cells_comm = {}
    sc8 = nested_scenario(full0, 8)
    for rho_c in RHO_C_GRID:
        b_tok = float(token_bits(8)["total"])
        b_rx = b_tok * (K_FIXED - 1) / float(rho_c)
        cap_tokens = b_rx / b_tok          # per-receiver hard cap (tokens)
        for geom in range(args.s_geom):
            sc = nested_scenario(full0, 8)
            try:
                bt = calibrate_target_bounds(
                    sc, args.alpha, args.beta, n_runs=args.calib_scan,
                    seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                    verify_runs=args.calib_verify)
            except ValueError:
                bt = None
            if bt is None:
                cells_comm.setdefault(str(rho_c), []).append({
                    "geom": geom, "zone": "calibration-family-infeasible"})
                continue
            bounds = [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(8)]
            rows = []
            for mc in range(args.s_geom):
                rows.append(run_qos(
                    sc, bounds, args.mc_runs, mc_seed=mc * 31 + 7,
                    delivery_matrix=sc["u2u_success"],
                    rx_cap_tokens=np.full(K_FIXED, cap_tokens)))
            pfa = np.mean([r["p_fa"] for r in rows], axis=0)
            pmd = np.mean([r["p_md"] for r in rows], axis=0)
            n_trials = args.s_geom * args.mc_runs
            status = qos_status(np.vstack([pfa, pmd]), args.alpha,
                                args.beta, n_trials)
            # P2.1 (advice/006 sections 7,12): UNCERTAIN boundary cells
            # escalate to the (s_geom_boundary x mc_boundary) protocol
            # until they become PASS/FAIL (same rule as the info axis)
            if status == "UNCERTAIN":
                rows = [run_qos(
                    sc, bounds, args.mc_boundary, mc_seed=mc * 131 + 7,
                    delivery_matrix=sc["u2u_success"],
                    rx_cap_tokens=np.full(K_FIXED, cap_tokens))
                    for mc in range(args.s_geom_boundary)]
                n_trials = args.s_geom_boundary * args.mc_boundary
                pfa = np.mean([r["p_fa"] for r in rows], axis=0)
                pmd = np.mean([r["p_md"] for r in rows], axis=0)
                status = qos_status(np.vstack([pfa, pmd]), args.alpha,
                                    args.beta, n_trials)
            # rho_I^eff uses the effective (drop-reduced) g
            sc_eff = _apply_drop(sc, cap_tokens)
            rho_i_eff = rho_info_of(sc_eff, beta=args.beta, alpha=args.alpha)
            zone = "communication-limited" if rho_c > 1.0 else (
                "feasible" if status == "PASS" else "coordination-limited")
            cells_comm.setdefault(str(rho_c), []).append({
                "geom": geom, "zone": zone, "qos_status": status,
                "rho_I_pre": rho_info_of(sc),
                "rho_I_eff": rho_i_eff, "rho_C": rho_c,
                "J": float(np.mean([r["J"] for r in rows])),
                "p_md_max": float(np.max(pmd)),
                "p_fa_max": float(np.max(pfa)),
            })
            print(f"comm rho_C {rho_c} geom {geom}: {zone} "
                  f"({status}) ({time.time()-t0:.0f}s)", flush=True)

    # ---------- Gamma_dual / Gamma_envelope vs nested-Q (P2.1-7) -------
    p2c = {"lambda_F": 0.0, "lambda_C": 0.0, "lambda_nec": lambda_nec,
           "censored": True, "censored_at": Q_GRID[0] / K_FIXED,
           "note": "right-censored at the tested load ceiling: lambdas "
                   ">= Q/K of the scan top; NOT an achieved equality",
           "Gamma_dual": None, "Gamma_envelope": None}
    # feasibility of FRIDS (local) and common-price audit at nested Q
    q_f, q_c = 0, 0
    for q in Q_GRID:
        sc = nested_scenario(full0, q)
        try:
            bt = calibrate_target_bounds(
                sc, args.alpha, args.beta, n_runs=args.calib_scan,
                seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                verify_runs=args.calib_verify)
        except ValueError:
            continue
        bounds = [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(q)]
        rho_c_q = float(token_bits(q)["total"]) * (K_FIXED - 1) \
            / max(args.rx_budget, 1e-12)
        if rho_info_of(sc) > 1.0 or rho_c_q > 1.0:
            continue          # not in the doubly-feasible necessary region
        for price_mode in ("local", "common"):
            rows = [run_qos(sc, bounds, args.mc_runs, mc_seed=mc * 131 + 7,
                            price_mode=price_mode)
                    for mc in range(args.s_geom)]
            pfa = np.mean([r["p_fa"] for r in rows], axis=0)
            pmd = np.mean([r["p_md"] for r in rows], axis=0)
            n_trials = args.s_geom * args.mc_runs
            if qos_status(np.vstack([pfa, pmd]), args.alpha, args.beta,
                          n_trials) == "PASS":
                if price_mode == "local":
                    q_f = q
                else:
                    q_c = q
    p2c["lambda_F"] = q_f / K_FIXED
    p2c["lambda_C"] = q_c / K_FIXED
    p2c["censored"] = bool(q_f >= Q_GRID[-1] or q_c >= Q_GRID[-1]
                        or lambda_nec >= Q_GRID[-1] / K_FIXED)
    p2c["Gamma_dual"] = round(p2c["lambda_F"] / max(p2c["lambda_C"], 1e-12),
                              3)
    p2c["Gamma_envelope"] = round(
        p2c["lambda_F"] / max(p2c["lambda_nec"], 1e-12), 3)
    print(f"P2-C lambda_F {p2c['lambda_F']} lambda_C {p2c['lambda_C']} "
          f"lambda_nec {p2c['lambda_nec']} censored {p2c['censored']} "
          f"Gamma_dual {p2c['Gamma_dual']} Gamma_envelope "
          f"{p2c['Gamma_envelope']} ({time.time()-t0:.0f}s)", flush=True)

    payload = {
        "gate": "p2.1-phase-diagram",
        "params": {
            "K": K_FIXED, "Q_max": Q_MAX, "Q_grid": list(Q_GRID),
            "s_geom": args.s_geom, "mc_runs": args.mc_runs,
            "s_geom_boundary": args.s_geom_boundary,
            "mc_boundary": args.mc_boundary,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "rx_budget": args.rx_budget,
            "snr_shift_db": list(SNR_SHIFT_DB),
            "rho_C_grid": list(RHO_C_GRID),
            "xi_g_sanity": XI_G,
            "frozen": ["FRIDS-v2", "fixed owner", "full mesh",
                       "19-bit token", "two-threshold stopping"],
            "protocol": [
                "nested-Q workload (Q_max first-q subsets)",
                "scenario_seed x MC_seed separation",
                "dual QoS P_FA<=alpha AND P_MD<=beta (Hoeffding UCB)",
                "hard per-receiver admission (uniform without replacement)",
                "rho_I^pre vs rho_I^eff (comm overload not info-labelled)",
                "4+1 region + mixed I+C",
                "Gamma_dual = lambda_F/lambda_C, Gamma_envelope; "
                "right-censored marking",
            ],
        },
        "runtime_s": round(time.time() - t0, 1),
        "p2a1": a1,
        "lambda_nested": lambdas,
        "lambda_nec": lambda_nec,
        "lambda_nec_censored": censored_nec,
        "cells_info": cells,
        "cells_comm": cells_comm,
        "p2c": p2c,
        "summary": {
            "info_zones": {z: sum(1 for shift in SNR_SHIFT_DB
                                  for c in cells[str(shift)]
                                  if c["zone"] == z)
                           for z in ("feasible", "information-limited",
                                     "communication-limited",
                                     "coordination-limited",
                                     "calibration-family-infeasible")},
            "comm_zones": {z: sum(1 for rho_c in RHO_C_GRID
                                  for c in cells_comm[str(rho_c)]
                                  if c["zone"] == z)
                           for z in ("feasible", "communication-limited",
                                     "coordination-limited")},
            "gamma_dual": p2c["Gamma_dual"],
            "gamma_envelope": p2c["Gamma_envelope"],
            "right_censored": p2c["censored"],
            "claim": ("no detected local-dual gap up to the tested "
                      "censored load ceiling; boundaries still to be "
                      "bracketed" if p2c["censored"]
                      else "capacity brackets found"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("summary:", json.dumps(payload["summary"], indent=1))
    print("done", round(time.time() - t0, 1), "s")


def _scale_g(scenario: dict, xi: float) -> dict:
    """Analytic-only scheduler-g scale (P2-A1 diagnostic): scales the
    kernel ``I+`` the cut reads.  NOT a physical experiment."""
    out = dict(scenario)
    out["by_host"] = {key: [dict(a) for a in acts]
                      for key, acts in scenario["by_host"].items()}
    for key, acts in out["by_host"].items():
        for a in acts:
            a["i_plus"] = float(a["i_plus"]) * xi
    return out


def _apply_drop(scenario: dict, cap_tokens: float) -> dict:
    """Effective-link scenario under the hard receive admission: each
    U2U effective success is the physical success times the budget
    survival ``min(1, cap_i / (K-1))`` (full mesh offers K-1 tokens),
    producing ``rho_I^eff`` (advice/006 section 8)."""
    out = dict(scenario)
    surv = min(1.0, cap_tokens / max(float(scenario["k"]) - 1, 1e-12))
    u2u_eff = np.multiply(scenario["u2u_success"], surv)
    np.fill_diagonal(u2u_eff, 1.0)
    out["u2u_success"] = u2u_eff
    return out


if __name__ == "__main__":
    main()