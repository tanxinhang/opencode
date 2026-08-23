"""CAPACITY CAUSAL ablation gate (advice/013 P4.1a): four arms under ONE
shared airtime-admission primitive (``sum tau/T_air <= 1`` per receiver,
``offer -> admission -> link``) and ONE shared policy tape
``exog.U_policy[r, t, receiver, src]`` for admission ties.

Arms:
- A   = FRIDS-v2 + shared airtime + NEUTRAL admission     (capacity-matched v2)
- B0  = CA architecture (owner-only, pi_q) + lambda=0 + NEUTRAL admission
- B1  = CA architecture + lambda_j + NEUTRAL admission
- C   = CA architecture + lambda_j + DENSITY admission    (production CA)

Causal decomposition on the geometry-pooled worst-target E[T|H1]
(J LOWER is better):
- D_architecture = J_B0 - J_A     (task-routing / CA-architecture effect)
- D_lambda       = J_B1 - J_B0    (congestion-PRICE steering)
- D_admission    = J_C - J_B1     (density packing)

Because A/B0/B1/C share the same physical tape AND the same admission
tie-key tape, the three deltas differ only in exactly one mechanism each
(advice/013 sections 3-4).  The old two-arm ``d_steering = J_CA_neutral -
J_v2_neutral`` conflated architecture + price + idle + owner-only routing;
it is no longer reported as ``steering``.
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

from uav_otfs_isac.airtime import build_airtime_model
from uav_otfs_isac.ca_frids import simulate_ca_frids
from uav_otfs_isac.crn_tape import build_exogenous_tape
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.frids import simulate_frids_v2

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_ca_frids_gate import matched_qos, j_pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/capacity_ablation_gate.json")
    parser.add_argument("--geoms", type=int, default=3)
    parser.add_argument("--mc-seeds", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=300)
    parser.add_argument("--pi-bits", type=int, default=10)
    parser.add_argument("--lam-bits", type=int, default=10)
    parser.add_argument("--uncongested-rho", type=float, default=0.5)
    parser.add_argument("--congested-rho", type=float, default=1.8)
    parser.add_argument("--price-mode", default="global_simplex",
                        choices=("global_simplex", "owner_local"))
    args = parser.parse_args()
    t0 = time.time()

    k, q = (16, 8)   # the gate's critical scale
    am = None

    rows_cells = []
    for geom in range(args.geoms):
        sc = build_distributed_scenario(np.random.default_rng(geom),
                                        k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(
            sc, args.alpha, args.beta, n_runs=300, seed=args.calib_seed,
            llr_bits=TOKEN_LLR_BITS, verify_runs=args.calib_verify)
        am = build_airtime_model(sc, rho_target=args.congested_rho)
        # every arm shares the same exogenous tape (and therefore the same
        # admission tie-key policy tape U_policy[r, t, :, :])
        _, rows_a = matched_qos(
            simulate_frids_v2, sc, bt, args.n_runs, 7,
            args.max_steps, args.alpha, args.beta,
            mc_seeds=args.mc_seeds, crn=True,
            price_mode="local", airtime=am)
        _, rows_b0 = matched_qos(
            simulate_ca_frids, sc, bt, args.n_runs, 7,
            args.max_steps, args.alpha, args.beta,
            mc_seeds=args.mc_seeds, crn=True,
            price_mode=args.price_mode, airtime=am,
            admission_policy="neutral", airtime_price=False,
            pi_bits=args.pi_bits, lam_bits=args.lam_bits)
        _, rows_b1 = matched_qos(
            simulate_ca_frids, sc, bt, args.n_runs, 7,
            args.max_steps, args.alpha, args.beta,
            mc_seeds=args.mc_seeds, crn=True,
            price_mode=args.price_mode, airtime=am,
            admission_policy="neutral", airtime_price=True,
            pi_bits=args.pi_bits, lam_bits=args.lam_bits)
        _, rows_c = matched_qos(
            simulate_ca_frids, sc, bt, args.n_runs, 7,
            args.max_steps, args.alpha, args.beta,
            mc_seeds=args.mc_seeds, crn=True,
            price_mode=args.price_mode, airtime=am,
            admission_policy="density", airtime_price=True,
            pi_bits=args.pi_bits, lam_bits=args.lam_bits)
        rows_cells.append({
            "geom": geom,
            "j_a": j_pooled(rows_a),
            "j_b0": j_pooled(rows_b0),
            "j_b1": j_pooled(rows_b1),
            "j_c": j_pooled(rows_c),
            "offers_v2": float(np.mean([
                r["comm"]["offer_attempts_per_uav"] for r in rows_a])),
            "admitted_v2": float(np.mean([
                r["comm"]["admitted_tx_per_uav"] for r in rows_a])),
            "offers_ca": float(np.mean([
                r["comm"]["offer_attempts_per_uav"] for r in rows_c])),
            "admitted_ca": float(np.mean([
                r["comm"]["admitted_tx_per_uav"] for r in rows_c])),
        })
        print(f"geom {geom} done ({time.time()-t0:.0f}s)", flush=True)

    a = np.array([c["j_a"] for c in rows_cells])
    b0 = np.array([c["j_b0"] for c in rows_cells])
    b1 = np.array([c["j_b1"] for c in rows_cells])
    c = np.array([c["j_c"] for c in rows_cells])
    d_arch = float(np.mean(b0 - a))
    d_lambda = float(np.mean(b1 - b0))
    d_admission = float(np.mean(c - b1))
    gate = {
        # J is the worst-target delay: LOWER is better, so a NEGATIVE
        # delta means the later arm is BETTER.
        "d_architecture_congested": d_arch,      # B0 - A   (task routing)
        "d_lambda_congested": d_lambda,          # B1 - B0  (price steering)
        "d_admission_congested": d_admission,    # C - B1   (density packing)
        "j_v2_neutral_mean": float(np.mean(a)),
        "j_ca_neutral_no_price_mean": float(np.mean(b0)),
        "j_ca_neutral_price_mean": float(np.mean(b1)),
        "j_ca_density_mean": float(np.mean(c)),
        "offer_vs_admitted_v2": float(np.mean([
            c["offers_v2"] - c["admitted_v2"] for c in rows_cells])),
        "offer_vs_admitted_ca": float(np.mean([
            c["offers_ca"] - c["admitted_ca"] for c in rows_cells])),
    }
    payload = {
        "gate_id": "p4.1a-capacity-causal-ablation",
        "params": {"geoms": args.geoms, "mc_seeds": args.mc_seeds,
                   "n_runs": args.n_runs, "alpha": args.alpha,
                   "beta": args.beta, "price_mode": args.price_mode,
                   "admission_primitive": "airtime_admit (sum tau/T_air <= 1)",
                   "policy_tape": "shared exog.U_policy[r,t,receiver,src]",
                   "arms": ["A=v2+neutral", "B0=CA+lambda0+neutral",
                            "B1=CA+lambda+neutral", "C=CA+lambda+density"]},
        "cells": rows_cells,
        "metrics": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()