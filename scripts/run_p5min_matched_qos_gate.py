"""P5-MIN matched-QoS gate: B0-core vs C (advice/020 section 9, item 3).

The P5-MIN minimality gate certifies the frozen-point delay minimality of
B0-lite but, as advice/020 section 9 notes, it does NOT certify the
matched-QoS claim: its ``_verdict`` checks ``J_lite <= J_C + delta_J``,
lower control bits and hard airtime feasibility, but NOT the
``P_FA <= alpha, P_MD <= beta`` matched-QoS PASS as a freeze condition.
So the paper may currently write "B0-lite is faster than full CA on the
pooled (16,8) grid" but NOT "B0-lite is certified no slower than full CA
under matched QoS".

This gate closes that gap (the minimal B0-core vs C matched-QoS gate): it
reuses the P4.2b matched-QoS frontier machinery (advice/017 section 12.1)
but compares the two CA schedulers

    B0-core : task price on, lambda OFF, NEUTRAL admission, NORM_FREE
              (the minimal Normalization-Free Distributed Deficit Pricing
              core, advice/020 section 15-16);
    C       : task price on, lambda ON,  DENSITY admission   (full CA).

For each scheduler it sweeps the per-target upper-threshold multiplier
``A_q = A0_q * m`` and finds a CERTIFIED FEASIBLE matched operating point
(all targets certified FA/MD within spec).  On the held-out CRN stream it
then compares the matched-QoS stopping delay.  Only when BOTH schedulers
are CERTIFIED FEASIBLE (CASE B) is the matched-QoS comparison reported;
this supports the claim "the minimal B0-core is not certified slower than
full CA at matched QoS" -- and, since B0-core carries strictly fewer
control bits (no lambda bus), that B0-core is the minimal algorithm to
freeze.

Cells: the registered (16,8,rho=1.8) cell plus a capacity-binding stress
cell (``--stress-rho``) -- exactly the two cells advice/020 section 9
asks for before writing the matched-QoS statement.

Run (full-scale cells, heavy):
    python scripts/run_p5min_matched_qos_gate.py

Run (smoke):
    python scripts/run_p5min_matched_qos_gate.py --cal-cell-runs 40 \
        --cal-mc 1 --test-cell-runs 60 --test-mc 1 \
        --geom 2 --rho 1.8 --stress-rho 2.5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

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
from uav_otfs_isac.qos import anytime_qos_status

from scripts.run_p42b_qos_frontier_gate import (
    SPEC,
    N_STREAMS,
    _acc,
    _certified_bounds,
    _empty_acc,
    _matched_bounds,
    classify_frontier_state,
    paired_reduction_bootstrap,
)

DEFAULT_SCALE = (16, 8)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
            text=True).strip())
    except Exception:  # noqa: BLE001
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/p5min_matched_qos_gate.json")
    parser.add_argument("--geom", type=int, default=2)
    parser.add_argument("--rho", type=float, default=1.8,
                        help="registered (16,8) cell congestion")
    parser.add_argument("--stress-rho", type=float, default=None,
                        help="capacity-binding stress cell rho; if None "
                             "only the registered cell is run")
    parser.add_argument("--scale", default="16,8", help="K,Q")
    parser.add_argument("--cal-seed", type=int, default=200000)
    parser.add_argument("--test-seed", type=int, default=300000)
    parser.add_argument("--cal-cell-runs", type=int, default=300)
    parser.add_argument("--cal-mc", type=int, default=2)
    parser.add_argument("--test-cell-runs", type=int, default=1500)
    parser.add_argument("--test-mc", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=SPEC)
    parser.add_argument("--beta", type=float, default=SPEC)
    parser.add_argument("--cell-delta", type=float, default=0.05)
    parser.add_argument("--grid", default="0.6,0.8,1.0,1.5,2.0,3.0,5.0")
    parser.add_argument("--pi-bits", type=int, default=10)
    parser.add_argument("--lam-bits", type=int, default=10)
    parser.add_argument("--theta-bits", type=int, default=10)
    parser.add_argument("--price-mode", default="global_simplex",
                        choices=("global_simplex", "owner_local"))
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    parser.add_argument("--calib-n-runs", type=int, default=300)
    args = parser.parse_args()
    t0 = time.time()

    kk, qq = args.scale.split(",")
    k, q = int(kk), int(qq)
    grid = sorted({float(m) for m in args.grid.split(",")})
    delta_q = args.cell_delta / (2.0 * q)

    cells = []
    rhos = [args.rho]
    if args.stress_rho is not None:
        rhos.append(args.stress_rho)
    for rho in rhos:
        sc = build_distributed_scenario(np.random.default_rng(args.geom),
                                        k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(sc, args.alpha, args.beta,
                                     n_runs=args.calib_n_runs,
                                     seed=args.calib_seed,
                                     llr_bits=TOKEN_LLR_BITS,
                                     verify_runs=args.calib_verify)
        am = build_airtime_model(sc, rho_target=rho)

        def run_core(bounds_, n_runs, seed_, tape):
            return simulate_ca_frids(
                sc, bounds_, am, n_runs, seed=seed_,
                max_steps=args.max_steps, raw_counts=True,
                price_mode=args.price_mode, pi_bits=args.pi_bits,
                lam_bits=args.lam_bits, theta_bits=args.theta_bits,
                task_price=True, airtime_price=False, norm_free=True,
                admission_policy="neutral", exog=tape)

        def run_c(bounds_, n_runs, seed_, tape):
            return simulate_ca_frids(
                sc, bounds_, am, n_runs, seed=seed_,
                max_steps=args.max_steps, raw_counts=True,
                price_mode=args.price_mode, pi_bits=args.pi_bits,
                lam_bits=args.lam_bits,
                task_price=True, airtime_price=True,
                admission_policy="density", exog=tape)

        def calibrate_scheduler(runner) -> dict:
            frontier = {"m_star": [None] * q, "A_star": [None] * q,
                        "cert_u_fa": [0.0] * q, "cert_u_md": [0.0] * q,
                        "target_state": ["UNRESOLVED"] * q,
                        "lc_fa_at_max": [0.0] * q, "lc_md_at_min": [0.0] * q,
                        "feasible": True, "scheduler_state": "UNRESOLVED"}
            lc_fa_max = [0.0] * q
            lc_md_min = [1.0] * q
            for q_t in range(q):
                for m in grid:
                    acc = _empty_acc(q)
                    for mc in range(args.cal_mc):
                        tape = build_exogenous_tape(
                            args.cal_seed + mc, args.cal_cell_runs, q, k,
                            args.max_steps)
                        mq = [1.0] * q
                        mq[q_t] = m
                        out = runner(_matched_bounds(bt, mq, 1.0),
                                     args.cal_cell_runs,
                                     args.cal_seed + mc, tape)
                        _acc(out, acc)
                    l_fa, u_fa, l_md, u_md = _certified_bounds(
                        acc, q, delta_q)
                    if m == max(grid):
                        lc_fa_max[q_t] = l_fa[q_t]
                    if m == min(grid):
                        lc_md_min[q_t] = l_md[q_t]
                    if (u_fa[q_t] <= SPEC and u_md[q_t] <= SPEC
                            and frontier["m_star"][q_t] is None):
                        frontier["m_star"][q_t] = m
                        frontier["A_star"][q_t] = bt[q_t][0] * m
                        frontier["cert_u_fa"][q_t] = u_fa[q_t]
                        frontier["cert_u_md"][q_t] = u_md[q_t]
                frontier["lc_fa_at_max"][q_t] = lc_fa_max[q_t]
                frontier["lc_md_at_min"][q_t] = lc_md_min[q_t]
            target_state, scheduler_state = classify_frontier_state(
                frontier["m_star"], frontier["lc_fa_at_max"],
                frontier["lc_md_at_min"], SPEC)
            frontier["target_state"] = target_state
            frontier["scheduler_state"] = scheduler_state
            frontier["feasible"] = bool(
                scheduler_state == "CERTIFIED FEASIBLE")
            return frontier

        core_frontier = calibrate_scheduler(run_core)
        c_frontier = calibrate_scheduler(run_c)

        def held_out(runner, m_star):
            acc = _empty_acc(q)
            pool_s, pool_n = np.zeros(q), np.zeros(q)
            block_J = []
            for mc in range(args.test_mc):
                tape = build_exogenous_tape(args.test_seed + mc,
                                            args.test_cell_runs, q, k,
                                            args.max_steps)
                out = runner(_matched_bounds(bt, m_star, 1.0),
                             args.test_cell_runs, args.test_seed + mc, tape)
                _acc(out, acc)
                pool_s += np.asarray(out["pool"]["sum_h1_delay"], dtype=float)
                pool_n += np.asarray(out["pool"]["n_h1"], dtype=float)
                b_s = np.asarray(out["pool"]["sum_h1_delay"], dtype=float)
                b_n = np.asarray(out["pool"]["n_h1"], dtype=float)
                block_J.append(float(np.max(b_s / np.maximum(b_n, 1.0))))
            J = float(np.max(pool_s / np.maximum(pool_n, 1.0)))
            status, _ = anytime_qos_status(
                acc["n_H0"], acc["n_H1"], acc["n_FA"], acc["n_MD"],
                args.alpha, args.beta, delta_fam=0.05,
                n_streams=N_STREAMS)
            return {"J": J, "qos": status, "m_star": list(m_star),
                    "block_J": block_J}

        core_held = (held_out(run_core, core_frontier["m_star"])
                     if core_frontier["feasible"] else None)
        c_held = (held_out(run_c, c_frontier["m_star"])
                  if c_frontier["feasible"] else None)

        core_state = core_frontier["scheduler_state"]
        c_state = c_frontier["scheduler_state"]
        if core_state == "CERTIFIED FEASIBLE" and c_state == "CERTIFIED FEASIBLE":
            case = "B"
        elif "CERTIFIED INFEASIBLE" in (core_state, c_state):
            case = "A-CERTIFIED-INFEASIBLE"
        else:
            case = "A-UNRESOLVED"

        matched = None
        if case == "B":
            # paired per-block bootstrap on the matched-QoS delay DIFFERENCE
            # J_C - J_core (CERTIFIED_GAIN if the core is faster; the gate
            # PASSES as long as the core is not CERTIFIED slower).
            delta = np.asarray(c_held["block_J"]) - np.asarray(
                core_held["block_J"])
            mean_delta = float(np.mean(delta))
            rng = np.random.default_rng(12345)
            boot = np.empty(2000)
            n = len(delta)
            for b in range(2000):
                idx = rng.integers(0, n, size=n)
                boot[b] = float(np.mean(delta[idx]))
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            if ci_lo > 0:
                state = "CERTIFIED_GAIN"          # core strictly faster
            elif ci_hi < 0:
                state = "CERTIFIED_LOSS"          # core certified slower
            else:
                state = "UNRESOLVED"              # not certified slower
            matched = {
                "J_core": core_held["J"], "J_C": c_held["J"],
                "D_C_minus_core_point": mean_delta,
                "D_C_minus_core_ci95": [float(ci_lo), float(ci_hi)],
                "state": state,
                "core_qos": core_held["qos"], "C_qos": c_held["qos"],
                "core_m_star": core_held["m_star"],
                "C_m_star": c_held["m_star"],
            }

        cells.append({
            "rho": rho, "scale": [k, q],
            "core_state": core_state, "C_state": c_state, "case": case,
            "core_frontier": core_frontier, "C_frontier": c_frontier,
            "matched": matched,
            "wording": _wording(core_state, c_state, matched),
        })

    ok = all(c["case"] == "B" for c in cells) and all(
        c["matched"]["state"] != "CERTIFIED_LOSS" for c in cells)
    payload = {
        "pass": bool(ok),
        "cells": cells,
        "spec": SPEC,
        "runtime_s": round(time.time() - t0, 1),
        "provenance": {"git_commit": _git_sha(), "git_dirty": _git_dirty()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("overall:", json.dumps({c["rho"]: {
        "case": c["case"],
        "matched_state": (c["matched"]["state"] if c["matched"] else None),
        "D": (c["matched"]["D_C_minus_core_point"] if c["matched"] else None),
    } for c in cells}, indent=1))
    print("done", round(time.time() - t0, 1), "s")


def _wording(core_state, c_state, matched):
    if core_state != "CERTIFIED FEASIBLE" or c_state != "CERTIFIED FEASIBLE":
        return ("no matched-QoS comparison: one or both schedulers are NOT "
                f"certified feasible (core={core_state}, C={c_state})")
    state = matched["state"]
    if state == "CERTIFIED_GAIN":
        return "B0-core is CERTIFIED faster than C at matched certified QoS"
    if state == "CERTIFIED_LOSS":
        return ("B0-core is CERTIFIED slower than C at matched certified "
                "QoS -- the minimality claim does NOT hold under matched QoS")
    return ("B0-core is not certified slower than C at matched certified "
            "QoS (UNRESOLVED: no certified difference)")


if __name__ == "__main__":
    main()
