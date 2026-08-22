"""Gate P2 smoke (advice/005 sections 8-12): three orthogonal physical
failure boundaries -- information / communication / coordination.

P0.6 harness is closed; P2 smokes are deliberately SMALL (no big
multi-seed matrices yet) to first confirm that each boundary really
exists with the CORRECT mechanism before the formal phase diagram runs:

- P2-A1 (analytical diagnostic): ``g -> xi_g g`` monotonicity of
  ``rho_I*(xi_g)`` (cut-level check ONLY -- NOT a physical experiment).
- P2-A2 (physical information boundary): rebuild the sensing kernels by
  scaling the physical noncentrality (SNR), so ``I+, p0, p1, LLR`` are
  REGENERATED and ``g`` changes through the real evidence chain; ``xi_phy``
  drives ``rho_I*`` across 1 (advice/005 section 8: never fake the bound by
  only shrinking the scheduler's ``g``).
- P2-B (communication boundary): sweep ``rho_C = b_tok (K-1) / B_rx`` and
  actually apply the admission/drop survival ``s_eff = s_phy *
  min(1, B_rx / (b_tok * offered))`` inside the delivered matrix, so
  budget overload becomes a real packet-survival loss, not a table entry
  (advice/005 section 9).
- P2-C (coordination boundary): in the doubly-feasible region compare
  ``lambda_FRIDS / lambda_common = Gamma_alg`` (algorithm utilization) and
  ``lambda_common / lambda_nec = Gamma_law`` (feasibility-law tightness),
  composing ``Gamma = Gamma_alg * Gamma_law`` (advice/005 section 10).

FRIDS-v2 is frozen throughout.
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
Q_BASE = 8
XI_G_GRID = (1.0, 0.5, 0.25, 0.15, 0.10, 0.075)
SNR_SHIFT_DB = (0.0, -3.0, -6.0, -10.0, -14.0, -18.0)
RHO_C_GRID = (0.5, 0.8, 1.0, 1.2, 1.5)


def g_matrix_of(scenario: dict) -> np.ndarray:
    """(K, Q) scheduled reliable-information matrix (scheduler g, which
    drives the cuts; the deployed atom drift is the quantized analogue)."""
    k = scenario["k"]
    q = scenario["q"]
    owner = scenario["owner_of"]
    g = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            rel = 1.0 if i == owner[qq] else \
                float(scenario["u2u_success"][i, owner[qq]])
            best = max(float(a["i_plus"]) * rel
                       for a in scenario["by_host"][(i, qq)])
            g[i, qq] = best
    return g


def rho_info(scenario: dict, horizon: int = 40, beta: float = 0.05,
             alpha: float = 0.05) -> float:
    cut = strongest_load_cut(scenario, scenario["owner_of"],
                             horizon=horizon, beta=beta, alpha=alpha)
    return float(cut["rho_star"])


def run_frids_qos(scenario, bounds, n_runs, seed, max_steps=40,
                  delivery_matrix=None, s_for_g=None, price_mode="local"):
    out = simulate_frids_v2(
        scenario, bounds, n_runs=n_runs, seed=seed, max_steps=max_steps,
        delivery_matrix=delivery_matrix, s_for_g=s_for_g,
        price_mode=price_mode)
    return {"J": float(out["worst_target_delay"]),
            "p_md_max": float(max(out["p_md"])),
            "p_fa_max": float(max(out["p_fa"]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/phase_boundary_smoke.json")
    parser.add_argument("--n-runs", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-scan", type=int, default=150)
    parser.add_argument("--calib-verify", type=int, default=300)
    args = parser.parse_args()
    t0 = time.time()

    base = build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=K_FIXED, q_targets=Q_BASE)
    bt = calibrate_target_bounds(
        base, args.alpha, args.beta, n_runs=args.calib_scan,
        seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
        verify_runs=args.calib_verify)
    bounds = [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(Q_BASE)]
    b_tok = float(token_bits(Q_BASE)["total"])

    # ---------- P2-A1: analytical rho_I*(xi_g) monotonicity ----------
    g0 = g_matrix_of(base)
    p2a1 = {}
    for xi in XI_G_GRID:
        sc_xi = dict(base)
        sc_xi["by_host"] = {
            key: [dict(a) for a in acts] for key, acts in base["by_host"].items()}
        # scheduler-g diagnostic only: the cut reads u2u_success and
        # i_plus from the scenario kernels -- we scale the kernel I+
        # (the scheduler's view) so rho_I* moves; this is the ANALYTICAL
        # cut, NOT the physical system
        for key, acts in sc_xi["by_host"].items():
            for a in acts:
                a["i_plus"] = float(a["i_plus"]) * xi
        p2a1[str(xi)] = {"xi_g": xi, "rho_I": rho_info(sc_xi)}
        print(f"P2-A1 g*{xi}: rho_I {p2a1[str(xi)]['rho_I']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- P2-A2: physical information boundary (SNR scaling) ----
    p2a2 = {}
    for shift in SNR_SHIFT_DB:
        sc_p = build_distributed_scenario(np.random.default_rng(0),
                                          k_uavs=K_FIXED, q_targets=Q_BASE,
                                          snr_shift=float(shift))
        # recouple the U2U matrix to the base draw so only the sensing
        # evidence is scaled
        sc_p["u2u_success"] = base["u2u_success"]
        rho = rho_info(sc_p)
        # the physical evidence degradation can make the two-threshold
        # calibration itself infeasible -- that IS a genuine detector
        # infeasibility, recorded honestly (the same boundary that makes
        # 4-bit tokens infeasible), not a harness error
        try:
            bt_p = calibrate_target_bounds(
                sc_p, args.alpha, args.beta, n_runs=args.calib_scan,
                seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                verify_runs=args.calib_verify)
            bounds_p = [[bt_p[qq][0], bt_p[qq][1] - 1.0]
                        for qq in range(Q_BASE)]
            qos = run_frids_qos(sc_p, bounds_p, args.n_runs,
                                seed=0 * 1000 + 7,
                                max_steps=args.max_steps)
        except ValueError:
            qos = {"J": float(args.max_steps), "p_md_max": 1.0,
                   "p_fa_max": 0.0, "calibration_infeasible": True}
        zone = ("info-infeasible" if rho > 1.0
                else ("detector-infeasible"
                      if "calibration_infeasible" in qos
                      else ("green" if qos["p_md_max"] <= args.beta + 0.02
                            else "yellow")))
        p2a2[str(shift)] = {
            "snr_shift_db": shift, "rho_I": rho, "zone": zone,
            "J": qos["J"], "p_md_max": qos["p_md_max"],
            "calibration_infeasible": bool("calibration_infeasible" in qos),
        }
        print(f"P2-A2 snr {shift:+.0f}dB: rho_I {rho:.3f} J {qos['J']:.2f} "
              f"P_MD {qos['p_md_max']:.3f} -> {zone} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- P2-B: communication boundary with real drop ----------
    p2b = {}
    for rho_c in RHO_C_GRID:
        b_rx = float(b_tok) * (K_FIXED - 1.0) / float(rho_c)
        # effective delivery survival: physical success x admission
        # survival (offered tokens per receiver in full mesh = K-1)
        offered = float(K_FIXED - 1)
        surv = min(1.0, b_rx / (b_tok * offered))
        s_eff = np.multiply(base["u2u_success"], surv)
        np.fill_diagonal(s_eff, 1.0)
        qos = run_frids_qos(base, bounds, args.n_runs,
                            seed=1 * 1000 + 7, max_steps=args.max_steps,
                            delivery_matrix=s_eff, s_for_g=s_eff)
        zone = ("comm-load>1" if rho_c > 1.0 else
                ("green" if qos["p_md_max"] <= args.beta + 0.02
                 else "yellow"))
        p2b[str(rho_c)] = {
            "rho_C": rho_c, "b_rx": b_rx, "drop_survival": surv,
            "zone": zone, "J": qos["J"], "p_md_max": qos["p_md_max"],
        }
        print(f"P2-B rho_C {rho_c}: surv {surv:.3f} J {qos['J']:.2f} "
              f"P_MD {qos['p_md_max']:.3f} -> {zone} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- P2-C: coordination utilization in the doubly-feasible region
    # (rho_I* < 1, rho_C < 1): FRIDS vs common-price oracle vs the
    # necessary-law capacity, Gamma = Gamma_alg * Gamma_law.
    rho_i_base = rho_info(base)
    rho_c_base = float(b_tok) * (K_FIXED - 1.0) / 400.0
    p2c = {"rho_I_base": rho_i_base, "rho_C_base": rho_c_base,
           "doubly_feasible": bool(rho_i_base < 1.0 and rho_c_base < 1.0)}
    if p2c["doubly_feasible"]:
        # lambda as the largest Q/K (one axis here: Q) that stays
        # QoS-feasible; smoke uses a coarse Q sweep for FRIDS and common.
        def qos_feasible_q(sc, bt_q, q, price_mode):
            out = run_frids_qos(sc, bt_q, args.n_runs, seed=2 * 1000 + 7,
                                max_steps=args.max_steps,
                                price_mode=price_mode)
            return out["p_md_max"] <= args.beta + 0.02

        lam_f, lam_c, lam_nec = 0.0, 0.0, 0.0
        for q in (4, 8, 12, 16):
            sq = build_distributed_scenario(np.random.default_rng(0),
                                            k_uavs=K_FIXED, q_targets=q)
            sq["u2u_success"] = base["u2u_success"]
            bt_q = calibrate_target_bounds(
                sq, args.alpha, args.beta, n_runs=args.calib_scan,
                seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                verify_runs=args.calib_verify)
            bq = [[bt_q[qq][0], bt_q[qq][1] - 1.0] for qq in range(q)]
            rho_i_q = rho_info(sq)
            if rho_i_q <= 1.0:
                lam_nec = max(lam_nec, float(q) / K_FIXED)
            if qos_feasible_q(sq, bq, q, "common"):
                lam_c = max(lam_c, float(q) / K_FIXED)
            if qos_feasible_q(sq, bq, q, "local"):
                lam_f = max(lam_f, float(q) / K_FIXED)
        p2c.update({
            "lambda_FRIDS": lam_f, "lambda_common": lam_c,
            "lambda_nec": lam_nec,
            "Gamma_alg": round(lam_f / max(lam_c, 1e-12), 3),
            "Gamma_law": round(lam_c / max(lam_nec, 1e-12), 3),
        })
        p2c["Gamma"] = round(p2c["Gamma_alg"] * p2c["Gamma_law"], 3)
        print(f"P2-C double-feasible: lam_F {lam_f} lam_C {lam_c} "
              f"lam_nec {lam_nec} Gamma_alg {p2c['Gamma_alg']} "
              f"Gamma_law {p2c['Gamma_law']} Gamma {p2c['Gamma']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    payload = {
        "gate": "p2-phase-boundary-smoke",
        "params": {
            "K": K_FIXED, "Q": Q_BASE, "n_runs": args.n_runs,
            "seeds": args.seeds, "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "xi_g_grid": list(XI_G_GRID),
            "snr_shift_db": list(SNR_SHIFT_DB),
            "rho_C_grid": list(RHO_C_GRID),
            "frozen": ["FRIDS-v2", "fixed owner", "full mesh",
                       "19-bit token", "two-threshold stopping"],
            "principles": [
                "P2-A1: analytical cut only (scheduler-g scale diagnostic)",
                "P2-A2: physical via SNR/noncentrality (evidence regenerated)",
                "P2-B: real admission/drop survival, not a table label",
                "P2-C: Gamma = Gamma_alg * Gamma_law in doubly-feasible region",
            ],
        },
        "runtime_s": round(time.time() - t0, 1),
        "p2a1": p2a1, "p2a2": p2a2, "p2b": p2b, "p2c": p2c,
        "verdict": {
            "a1_rho_monotone": bool(
                list(p2a1.values())[-1]["rho_I"] >=
                list(p2a1.values())[0]["rho_I"]),
            "a2_info_boundary_crossed": bool(
                min(v["rho_I"] for v in p2a2.values()) <= 1.0
                and max(v["rho_I"] for v in p2a2.values()) >= 1.0),
            "a2_detector_infeasible_before_info": bool(
                max(v["snr_shift_db"] for v in p2a2.values()
                    if v["zone"] == "detector-infeasible") >
                max(v["snr_shift_db"] for v in p2a2.values()
                    if v["zone"] == "info-infeasible")),
            "b_survival_degrades": bool(
                p2b["1.5"]["drop_survival"] < p2b["0.5"]["drop_survival"]),
            "c_gamma_alg_near_one": bool(
                p2c.get("Gamma_alg", 0.0) >= 0.9),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("verdict:", json.dumps(payload["verdict"], indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()