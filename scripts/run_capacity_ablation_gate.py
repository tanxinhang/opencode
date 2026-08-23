"""CAPACITY ablation gate (advice/012 section 4): decompose where the CA
congested gain comes from under the P4.1 shared-capacity closure.

Two schedulers, ONE shared ``airtime_admit`` primitive (``sum tau/T_air
<= 1`` per receiver, ``offer -> admission -> link``):

- A = FRIDS-v2 + exchangeable NEUTRAL admission  (capacity-matched v2)
- B = CA steering + NEUTRAL admission            (prices steer, admission neutral)
- C = CA steering + DENSITY admission            (prices steer, admission packed)

so ``D_steering = B - A`` isolates the congestion-price effect and
``D_admission = C - B`` isolates the density-packing effect, both on the
geometry-pooled worst-target E[T|H1] under the same physical capacity.
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
from run_ca_frids_gate import matched_bounds, matched_qos, j_pooled


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
    rho_to_air = {args.uncongested_rho: "uncongested",
                  args.congested_rho: "congested"}

    rows_cells = []
    for geom in range(args.geoms):
        sc = build_distributed_scenario(np.random.default_rng(geom),
                                        k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(
            sc, args.alpha, args.beta, n_runs=300, seed=args.calib_seed,
            llr_bits=TOKEN_LLR_BITS, verify_runs=args.calib_verify)
        am = build_airtime_model(sc, rho_target=args.congested_rho)
        # A: capacity-matched v2 (neutral admission, shared airtime)
        _, rows_a = matched_qos(simulate_frids_v2, sc, bt, args.n_runs, 7,
                                args.max_steps, args.alpha, args.beta,
                                mc_seeds=args.mc_seeds, crn=True,
                                price_mode="local", airtime=am)
        # B: CA steering, neutral admission
        _, rows_b = matched_qos(simulate_ca_frids, sc, bt, args.n_runs, 7,
                                args.max_steps, args.alpha, args.beta,
                                mc_seeds=args.mc_seeds, crn=True,
                                price_mode=args.price_mode, airtime=am,
                                admission_policy="neutral",
                                pi_bits=args.pi_bits, lam_bits=args.lam_bits)
        # C: CA steering, density admission (the production variant)
        _, rows_c = matched_qos(simulate_ca_frids, sc, bt, args.n_runs, 7,
                                args.max_steps, args.alpha, args.beta,
                                mc_seeds=args.mc_seeds, crn=True,
                                price_mode=args.price_mode, airtime=am,
                                admission_policy="density",
                                pi_bits=args.pi_bits, lam_bits=args.lam_bits)
        rows_cells.append({
            "geom": geom,
            "j_v2_neutral": j_pooled(rows_a),
            "j_ca_neutral": j_pooled(rows_b),
            "j_ca_density": j_pooled(rows_c),
            "offers_v2": float(np.mean([
                r["comm"]["offer_attempts_per_uav"] for r in rows_a])),
            "admitted_v2": float(np.mean([
                r["comm"]["admitted_tx_per_uav"] for r in rows_a])),
            "offers_ca_d": float(np.mean([
                r["comm"]["offer_attempts_per_uav"] for r in rows_c])),
            "admitted_ca_d": float(np.mean([
                r["comm"]["admitted_tx_per_uav"] for r in rows_c])),
        })
        print(f"geom {geom} done ({time.time()-t0:.0f}s)", flush=True)

    # congested-arm pooled worst J per arm (the geometry is the unit)
    a_vals = np.array([c["j_v2_neutral"] for c in rows_cells])
    b_vals = np.array([c["j_ca_neutral"] for c in rows_cells])
    c_vals = np.array([c["j_ca_density"] for c in rows_cells])
    d_steering = float(np.mean(b_vals - a_vals))
    d_admission = float(np.mean(c_vals - b_vals))
    gate = {
        # J is the worst-target delay: LOWER is better, so a negative
        # delta means the later arm is BETTER.
        "d_steering_congested": d_steering,        # B - A (price effect)
        "d_admission_congested": d_admission,      # C - B (density effect)
        "j_v2_neutral_mean": float(np.mean(a_vals)),
        "j_ca_neutral_mean": float(np.mean(b_vals)),
        "j_ca_density_mean": float(np.mean(c_vals)),
        "offer_vs_admitted_v2": float(np.mean([
            c["offers_v2"] - c["admitted_v2"] for c in rows_cells])),
        "offer_vs_admitted_ca_density": float(np.mean([
            c["offers_ca_d"] - c["admitted_ca_d"] for c in rows_cells])),
    }
    payload = {
        "gate": "p4.1-capacity-ablation",
        "params": {"geoms": args.geoms, "mc_seeds": args.mc_seeds,
                   "n_runs": args.n_runs, "alpha": args.alpha,
                   "beta": args.beta, "price_mode": args.price_mode,
                   "admission_primitive": "airtime_admit (sum tau/T_air <= 1)",
                   "arms": ["v2+neutral", "CA+neutral", "CA+density"]},
        "cells": rows_cells,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()