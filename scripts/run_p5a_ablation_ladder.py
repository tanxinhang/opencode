"""P5-A ablation ladder: mechanism attribution of the CA-FRIDS gain
(advice/017 section 13).

The reviewer attack is: "is the CA gain just owner routing?"  This gate
decomposes the gain into a minimal mechanism ladder on the REGISTERED
congested boundary cell (geom=2, rho=1.8) at the FROZEN calibrated
policy-B operating point (delta=1, same stopping thresholds for every
arm -- the fair scheduler-only / mechanism-only comparison):

    A   : FRIDS-v2  (reference; local task price, neutral admission)
    B00 : owner_arch_flat  (owner-directed evidence plane, FLAT index --
          NO task price, NO receiver price, neutral admission)
    B0  : B00 + dynamic task price pi_q = y_q/(D_q+eps)
    B1  : B0  + receiver airtime price lambda_j
    C   : B1  + density admission   (= full CA-FRIDS)

    D_owner_bundle = J_A   - J_B00   (architecture BUNDLE: owner-directed
          evidence plane + REMOVAL of v2's local-deficit price -- NOT pure
          routing, advice/018 section 7)
    D_pi        = J_B00 - J_B0    (task-deficit coordination)
    D_lambda    = J_B0  - J_B1    (receiver-capacity steering)
    D_admission = J_B1  - J_C     (density admission)

    A / B00 / B0 are ALSO the F1 / O0 / O1 cells of the 2x2 core
    mechanism table (advice/018 section 8); a full-mesh FLAT arm F0
    (FRIDS-v2 with ``task_price=False``) completes the grid::

              | Flat info     | Deficit-aware    |
        ------+---------------+------------------+
        Full-mesh  | (F0)  | (F1 = A)          |
        Owner-dir  | (O0=B00) | (O1=B0)        |

    which separates the OWNER-ARCHITECTURE effect (delta_architecture_flat)
    from the TASK-PRICE effect (delta_task_owner / delta_task_mesh) plus
    their interaction -- answering "owner routing vs task price" without a
    caveat.

    Every delta is ALSO recomputed on the error-aware estimand
    ``J_risk = max_q E[T_q^risk | H1]`` (an H0 declaration under H1 is
    charged T_max; advice/018 section 5), so a gain certified on plain J
    must be certified on J_risk too before it counts as useful evidence
    rather than a bought-by-earlier-wrong-decisions gain.

J is the held-out matched-policy worst-target E[T|H1] (pooled).  Every
arm consumes the SAME held-out CRN exogen tape per block, so each
consecutive delta is a paired comparison with a per-block bootstrap CI.
QoS (anytime-valid, 32-stream familywise, exactly as P4.2b) is reported
per arm at the frozen operating point (arms may differ in QoS status --
that is itself part of the mechanism attribution).

Running at the FROZEN common point (not matched-QoS) is intentional:
the ladder asks WHY the architecture reaches the same evidence faster at
the SAME stopping policy, before any threshold re-tuning is admitted.
"""
from __future__ import annotations

import argparse
import hashlib
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
from uav_otfs_isac.frids import simulate_frids_v2
from uav_otfs_isac.qos import anytime_qos_status

SPEC = 0.05
SCALE = (16, 8)
N_STREAMS = 32


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            text=True).strip()
    except Exception:
        return "unknown"


def _git_tree() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=str(PROJECT_ROOT),
            text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True)
        return bool(out.strip())
    except Exception:
        return True


def _sha16(text) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _config_hash(args) -> str:
    fields = {
        "geom": args.geom, "rho": args.rho,
        "test_seed": args.test_seed, "test_cell_runs": args.test_cell_runs,
        "test_mc": args.test_mc,
        "max_steps": args.max_steps, "alpha": args.alpha, "beta": args.beta,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode,
        "calib_seed": args.calib_seed, "calib_verify": args.calib_verify,
        "calib_n_runs": args.calib_n_runs,
    }
    return _sha16(json.dumps(fields, sort_keys=True, indent=0))


def _empty_acc(q):
    return {key: [0] * q for key in ("n_H0", "n_H1", "n_FA", "n_MD")}


def _acc(out, acc):
    for key in ("n_H0", "n_H1", "n_FA", "n_MD"):
        acc[key] = [a + b for a, b in zip(acc[key], out["raw_counts"][key])]


def _run_arm(runner, bounds, n_runs, seed, max_steps, exog_blocks):
    """Run ONE arm over ``test_mc`` held-out CRN blocks (all blocks share
    the same frozen thresholds).

    Stores, per block AND per target, the RAW pooled statistics
    (``n_h1``, ``sum_h1_delay``, ``sum_h1_delay_risk``) so the bootstrap can
    re-derive the SAME pooled estimand the table reports (advice/018 P1:
    ``J = max_q sum_b S_bq / sum_b N_bq`` -- the old per-block-then-mean
    bootstrap was a different estimand).  Also returns the pooled
    risk-adjusted delay ``J_risk = max_q sum_h1_delay_risk/n_h1``, per-target
    anytime-valid FA/MD bounds, and the control-plane bits per cycle (the
    gated ledger: disabled buses are NOT charged -- advice/018 section 6).
    """
    acc = _empty_acc(len(bounds))
    pool_s, pool_n = np.zeros(len(bounds)), np.zeros(len(bounds))
    pool_risk = np.zeros(len(bounds))
    block_n = []          # per block: (q,) target H1 counts
    block_s = []          # per block: (q,) target H1 delay sums
    block_risk = []       # per block: (q,) target H1 risk-adjusted sums
    ctrl_bits = []
    for mc, tape in enumerate(exog_blocks):
        out = runner(bounds, n_runs, seed + mc, tape)
        _acc(out, acc)
        b_n = np.asarray(out["pool"]["n_h1"], dtype=float)
        b_s = np.asarray(out["pool"]["sum_h1_delay"], dtype=float)
        b_r = np.asarray(out["pool"]["sum_h1_delay_risk"], dtype=float)
        block_n.append(b_n)
        block_s.append(b_s)
        block_risk.append(b_r)
        pool_s += b_s
        pool_n += b_n
        pool_risk += b_r
        ctrl_bits.append(float(out["comm"]["control_bits_per_cycle"]))
    J = float(np.max(pool_s / np.maximum(pool_n, 1.0)))
    J_risk = float(np.max(pool_risk / np.maximum(pool_n, 1.0)))
    qos, bounds_fam = anytime_qos_status(
        acc["n_H0"], acc["n_H1"], acc["n_FA"], acc["n_MD"],
        0.05, 0.05, delta_fam=0.05, n_streams=N_STREAMS,
        ret_bounds=True)
    return {
        "J": J, "J_risk": J_risk,
        "block_n": [list(b) for b in block_n],
        "block_s": [list(b) for b in block_s],
        "block_risk": [list(b) for b in block_risk],
        "qos": qos,
        "fa_md_lo_hi": {"FA_lo": bounds_fam["FA_lo"],
                        "FA_hi": bounds_fam["FA_hi"],
                        "MD_lo": bounds_fam["MD_lo"],
                        "MD_hi": bounds_fam["MD_hi"]},
        "ctrl_bits_per_cycle": float(np.mean(ctrl_bits)) if ctrl_bits else 0.0,
    }


def _pooled_pool(blocks_n, blocks_s):
    """Pooled per-target ``(N_q, S_q)`` from a list of per-block arrays."""
    N = np.sum(np.stack(blocks_n, axis=0), axis=0)
    S = np.sum(np.stack(blocks_s, axis=0), axis=0)
    return N, S


def _pooled_j(N, S):
    """The EXACT reported estimand: ``J = max_q S_q/N_q`` (pooled over
    blocks of one arm -- advice/018 section 2)."""
    return float(np.max(S / np.maximum(N, 1.0)))


def _pooled_delta_ci(prev_n, prev_v, cur_n, cur_v,
                     n_boot=10000, seed=0):
    """advice/018 P1: paired per-block bootstrap on the EXACT pooled
    estimand ``J = max_q sum_b V_bq / sum_b N_bq`` where ``V`` is the
    per-target value sums (plain ``sum_h1_delay`` or risk-adjusted
    ``sum_h1_delay_risk``) and ``N`` the per-target H1 counts.  Each
    repeat draws block indices ``b*``, re-pools prev and cur over the
    resampled blocks, and computes
        J_prev*(r) = max_q sum_{b in b*} V_bq / sum_{b in b*} N_bq
        J_cur*(r)  = ...
        delta*(r)  = J_prev*(r) - J_cur*(r).
    The 95% CI is over these delta*.  Return ``(point, lo, hi)`` with
    ``point = J_prev - J_cur`` from the FULL pool (the number the table
    actually reports).  This is the estimand-CORRECT bootstrap -- the old
    per-block-then-mean bootstrap answered a different statistic
    (advice/018 section 2)."""
    B = len(prev_n)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    P_N, P_V = _pooled_pool(prev_n, prev_v)
    C_N, C_V = _pooled_pool(cur_n, cur_v)
    point = _pooled_j(P_N, P_V) - _pooled_j(C_N, C_V)
    for b in range(n_boot):
        idx = rng.integers(0, B, size=B)
        pN = np.sum(np.stack([prev_n[j] for j in idx], axis=0), axis=0)
        pV = np.sum(np.stack([prev_v[j] for j in idx], axis=0), axis=0)
        cN = np.sum(np.stack([cur_n[j] for j in idx], axis=0), axis=0)
        cV = np.sum(np.stack([cur_v[j] for j in idx], axis=0), axis=0)
        boot[b] = _pooled_j(pN, pV) - _pooled_j(cN, cV)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _pooled_j_delta_ci(prev_n, prev_s, cur_n, cur_s,
                       n_boot=10000, seed=0):
    """The reported plain-delay delta (``V = sum_h1_delay``)."""
    return _pooled_delta_ci(prev_n, prev_s, cur_n, cur_s, n_boot, seed)


def _interaction_delta_ci(f0n, f0s, f1n, f1s, o0n, o0s, o1n, o1s,
                          n_boot=10000, seed=0):
    """advice/018 section 8: paired bootstrap CI for the 2x2 interaction
    ``(J_F0 - J_O0) - (J_F1 - J_O1)`` -- is the OWNER-ARCHITECTURE effect
    the same at flat vs deficit-aware information?  A single resampled
    block set drives ALL FOUR cells (they share the same CRN tape), so the
    interaction is a paired comparison.  ``point`` uses the four FULL-pool
    J values; the CI is over the bootstrap repeats."""
    B = len(f0n)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)

    def _j(blocks_n, blocks_s, idx):
        N = np.sum(np.stack([blocks_n[j] for j in idx], axis=0), axis=0)
        S = np.sum(np.stack([blocks_s[j] for j in idx], axis=0), axis=0)
        return _pooled_j(N, S)

    F0 = _pooled_j(*_pooled_pool(f0n, f0s))
    F1 = _pooled_j(*_pooled_pool(f1n, f1s))
    O0 = _pooled_j(*_pooled_pool(o0n, o0s))
    O1 = _pooled_j(*_pooled_pool(o1n, o1s))
    point = (F0 - O0) - (F1 - O1)
    for b in range(n_boot):
        idx = rng.integers(0, B, size=B)
        arch_flat = _j(f0n, f0s, idx) - _j(o0n, o0s, idx)
        arch_deficit = _j(f1n, f1s, idx) - _j(o1n, o1s, idx)
        boot[b] = arch_flat - arch_deficit
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/p5a_ablation_ladder.json")
    parser.add_argument("--geom", type=int, default=2,
                        help="the registered congested boundary cell geometry")
    parser.add_argument("--rho", type=float, default=1.8,
                        help="congested rho_target of the registered cell")
    parser.add_argument("--test-seed", type=int, default=400000,
                        help="FRESH held-out seed namespace (disjoint from "
                             "P4.2 cert 100000, P4.1b discovery 7, threshold "
                             "calibration 100, P4.2b cal 200000 / test 300000)")
    parser.add_argument("--test-cell-runs", type=int, default=1500)
    parser.add_argument("--test-mc", type=int, default=8,
                        help="held-out blocks = test_cell_runs * test_mc")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=SPEC)
    parser.add_argument("--beta", type=float, default=SPEC)
    parser.add_argument("--pi-bits", type=int, default=10)
    parser.add_argument("--lam-bits", type=int, default=10)
    parser.add_argument("--price-mode", default="global_simplex",
                        choices=("global_simplex", "owner_local"))
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    parser.add_argument("--calib-n-runs", type=int, default=300)
    args = parser.parse_args()
    t0 = time.time()

    k, q = SCALE
    sc = build_distributed_scenario(np.random.default_rng(args.geom),
                                    k_uavs=k, q_targets=q)
    bt = calibrate_target_bounds(sc, args.alpha, args.beta,
                                 n_runs=args.calib_n_runs,
                                 seed=args.calib_seed,
                                 llr_bits=TOKEN_LLR_BITS,
                                 verify_runs=args.calib_verify)
    am = build_airtime_model(sc, rho_target=args.rho)
    bounds = [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(q)]

    # shared held-out CRN exogen blocks (same tapes drive every arm)
    exog_blocks = [
        build_exogenous_tape(args.test_seed + mc, args.test_cell_runs,
                             q, k, args.max_steps)
        for mc in range(args.test_mc)
    ]

    # --- arm runners ----------------------------------------------------
    def run_v2(bounds, n_runs, seed, tape):
        return simulate_frids_v2(sc, bounds, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode="local", airtime=am, exog=tape)

    def run_f0(bounds, n_runs, seed, tape):
        # 2x2 core cell F0 (advice/018 section 8): FRIDS-v2 FULL-MESH with
        # the FLAT index -- no local deficit price, no mirror descent
        # (``task_price=False``).  This is the full-mesh + flat-g reference
        # that makes O0 = B00 an ISOLATED owner-architecture ablation
        # (compared with F1 = A) instead of a bundle.
        return simulate_frids_v2(sc, bounds, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode="local", airtime=am, exog=tape,
                                 task_price=False)

    def run_b00(bounds, n_runs, seed, tape):
        # owner_arch_flat (advice/018 section 7): CA owner-directed evidence
        # plane WITHOUT task price and WITHOUT receiver price (flat pi),
        # neutral admission -- the isolated owner-directed architecture.
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 task_price=False, airtime_price=False,
                                 admission_policy="neutral", exog=tape)

    def run_b0(bounds, n_runs, seed, tape):
        # B00 + dynamic task price pi_q = y_q/(D_q+eps); no receiver price.
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 task_price=True, airtime_price=False,
                                 admission_policy="neutral", exog=tape)

    def run_b1(bounds, n_runs, seed, tape):
        # B0 + receiver airtime price lambda_j; still neutral admission.
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 task_price=True, airtime_price=True,
                                 admission_policy="neutral", exog=tape)

    def run_c(bounds, n_runs, seed, tape):
        # full CA-FRIDS: + density admission.
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 task_price=True, airtime_price=True,
                                 admission_policy="density", exog=tape)

    arms = {
        "A": (run_v2, "FRIDS-v2 (reference)"),
        "F0": (run_f0, "full-mesh + flat g (2x2 core cell)"),
        "B00": (run_b00, "owner_arch_flat (owner-directed evidence, flat index)"),
        "B0": (run_b0, "owner-directed + task-deficit price"),
        "B1": (run_b1, "owner-directed + task price + receiver price"),
        "C": (run_c, "full CA-FRIDS (+ density admission)"),
    }
    results = {}
    for name, (runner, label) in arms.items():
        results[name] = _run_arm(runner, bounds, args.test_cell_runs,
                                 args.test_seed, args.max_steps, exog_blocks)
        print(f"arm {name:>3} ({label}): J={results[name]['J']:.4f} "
              f"qos={results[name]['qos']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # --- consecutive deltas (CI-AWARE three-state, advice/018 sect 2-3) ----
    def _delta(prev, cur):
        """Delta on the SAME pooled estimand as the reported table J
        (advice/018 section 2: the bootstrap re-derives pooled
        ``J*(r)=max_q sum_b S_bq / sum_b N_bq`` -- NOT the mean of the
        per-block max; ``point = J_prev - J_cur`` from the FULL pool, with a
        per-block bootstrap CI).  The sign is CI-AWARE three-state
        (advice/018 section 3): ``CERTIFIED_GAIN`` iff CI_lo > 0,
        ``CERTIFIED_LOSS`` iff CI_hi < 0, otherwise UNRESOLVED -- a point
        estimate alone is NOT certification.  Wording is sign-aware: a
        certified loss reads ``J_cur > J_prev by |d| (lambda HURTS
        worst-target delay)``, never the self-contradictory ``J_cur <
        J_prev by -x``."""
        if cur is None:
            return None
        d, lo, hi = _pooled_j_delta_ci(results[prev]["block_n"],
                                       results[prev]["block_s"],
                                       results[cur]["block_n"],
                                       results[cur]["block_s"])
        base = float(np.max(
            results[prev]["block_s"] / np.maximum(
                results[prev]["block_n"], 1.0))) if results[prev]["block_s"] else 1.0
        rel = d / max(base, 1e-12)
        if lo > 0.0:
            state = "CERTIFIED_GAIN"
        elif hi < 0.0:
            state = "CERTIFIED_LOSS"
        else:
            state = "UNRESOLVED"
        sign = ("a CERTIFIED gain mechanism" if state == "CERTIFIED_GAIN"
                else "a CERTIFIED loss mechanism" if state == "CERTIFIED_LOSS"
                else "unresolved (the point is not certified)")
        return {"point": d, "ci95": [lo, hi], "rel": rel,
                "state": state,
                "is_certified_gain": bool(state == "CERTIFIED_GAIN"),
                "is_certified_loss": bool(state == "CERTIFIED_LOSS"),
                "wording": (
                    f"J_{cur} {'<' if d >= 0 else '>'} J_{prev} by "
                    f"{abs(d):.4f} ({abs(rel) * 100.0:.1f}%; "
                    f"95% CI [{lo:.4f}, {hi:.4f}]; {state}: {sign})")}

    deltas = {
        "D_owner_bundle": _delta("A", "B00"),
        "D_pi": _delta("B00", "B0"),
        "D_lambda": _delta("B0", "B1"),
        "D_admission": _delta("B1", "C"),
    }

    # risk-adjusted deltas (advice/018 section 5): the SAME pooled
    # bootstrap, but on the error-aware estimand
    # ``J_risk = max_q sum_b sum_h1_delay_risk / sum_b n_h1`` where an H0
    # declaration under H1 is charged T_max.  A mechanism whose plain-J
    # gain is NOT reproduced on J_risk is NOT a useful-evidence gain -- it
    # is a bought-by-earlier-wrong-decisions gain.
    risk_deltas = {}
    for name, (pv, cv) in (("D_owner_bundle", ("A", "B00")),
                           ("D_pi", ("B00", "B0")),
                           ("D_lambda", ("B0", "B1")),
                           ("D_admission", ("B1", "C"))):
        rd, rlo, rhi = _pooled_delta_ci(results[pv]["block_n"],
                                        results[pv]["block_risk"],
                                        results[cv]["block_n"],
                                        results[cv]["block_risk"])
        if rlo > 0.0:
            rstate = "CERTIFIED_GAIN"
        elif rhi < 0.0:
            rstate = "CERTIFIED_LOSS"
        else:
            rstate = "UNRESOLVED"
        risk_deltas[name] = {
            "point": rd, "ci95": [rlo, rhi],
            "state": rstate,
            "is_certified_gain": bool(rstate == "CERTIFIED_GAIN"),
            "is_certified_loss": bool(rstate == "CERTIFIED_LOSS"),
        }

    # --- 2x2 core mechanism table (advice/018 section 8) ----------------
    # A / B00 / B0 are the F1 / O0 / O1 cells; the F0 arm (full-mesh +
    # flat g, ``task_price=False``) completes the grid:
    #        | Flat info    | Deficit-aware |
    # -------+--------------+---------------+
    # Full-mesh | (F0)      | (F1 = A)      |
    # Owner-dir | (O0 = B00)| (O1 = B0)     |
    # This separates OWNER-ARCHITECTURE from TASK-PRICE without the
    # D_owner_bundle caveat:
    #   delta_architecture_flat = J_F0 - J_O0   (owner value at flat info)
    #   delta_task_owner        = J_O0 - J_O1   (task price within owner)
    #   delta_task_mesh         = J_F0 - J_F1   (task price within mesh)
    #   delta_interaction       = arch_flat - arch_deficit (is the owner
    #                            effect the same with and without the price?)
    d_arch_flat = _delta("F0", "B00")
    d_task_owner = _delta("B00", "B0")
    d_task_mesh = _delta("F0", "A")
    int_point, int_lo, int_hi = _interaction_delta_ci(
        results["F0"]["block_n"], results["F0"]["block_s"],
        results["A"]["block_n"], results["A"]["block_s"],
        results["B00"]["block_n"], results["B00"]["block_s"],
        results["B0"]["block_n"], results["B0"]["block_s"])
    if int_lo > 0.0:
        int_state = "CERTIFIED_POSITIVE_INTERACTION"
    elif int_hi < 0.0:
        int_state = "CERTIFIED_NEGATIVE_INTERACTION"
    else:
        int_state = "UNRESOLVED"
    mechanism_2x2 = {
        "cells": {
            "F0": {"J": results["F0"]["J"],
                   "desc": "full-mesh + flat g (task_price=False)"},
            "F1": {"J": results["A"]["J"],
                   "desc": "full-mesh + local deficit price (= arm A)"},
            "O0": {"J": results["B00"]["J"],
                   "desc": "owner-directed + flat g (= arm B00)"},
            "O1": {"J": results["B0"]["J"],
                   "desc": "owner-directed + task-deficit price (= arm B0)"},
        },
        "deltas": {
            "delta_architecture_flat": d_arch_flat,
            "delta_task_owner": d_task_owner,
            "delta_task_mesh": d_task_mesh,
            "delta_interaction": {
                "point": int_point, "ci95": [int_lo, int_hi],
                "state": int_state,
                "wording": (
                    f"interaction (J_F0-J_O0) - (J_F1-J_O1) = "
                    f"{int_point:.4f}; 95% CI [{int_lo:.4f}, {int_hi:.4f}]; "
                    f"{int_state} -- {('owner effect is STRONGER with the '
                                        'deficit price' if int_point > 0 else
                                        'owner effect is weaker with the '
                                        'deficit price')}")},
        },
    }

    # cumulative mechanism decomposition
    cum = results["A"]["J"] - results["C"]["J"]
    ladder = {
        "A": results["A"]["J"],
        "B00": results["B00"]["J"],
        "B0": results["B0"]["J"],
        "B1": results["B1"]["J"],
        "C": results["C"]["J"],
        "cumulative_owner_to_full": float(cum),
        "decomposition": {
            "D_owner_bundle": deltas["D_owner_bundle"]["point"],
            "D_pi": deltas["D_pi"]["point"],
            "D_lambda": deltas["D_lambda"]["point"],
            "D_admission": deltas["D_admission"]["point"],
        },
    }

    # mechanism-dominant verdict (advice/017 section 13 + advice/018
    # section 3): the SINGLE largest CERTIFIED POSITIVE consecutive delta
    # names the mechanism that explains most of the CA-FRIDS gain at the
    # frozen operating point (audit/017 13 #1: a NEGATIVE delta -- e.g.
    # D_lambda here -- is a HARM, not a gain, so it must never be selected
    # as dominant).  Advice/018 section 3: a POINT estimate is not
    # certification -- only a delta whose bootstrap CI_lo > 0
    # (``is_certified_gain``) may be dominant; if NO certified positive
    # delta exists the verdict is "no dominant positive mechanism" instead
    # of silently picking an UNRESOLVED point.
    gains = {k: d for k, d in deltas.items() if d["is_certified_gain"]}
    if gains:
        dom = max(gains, key=lambda k: gains[k]["point"])
        dominant = dom
        dominant_point = gains[dom]["point"]
        dominant_rel = gains[dom]["rel"]
        c_total = results["A"]["J"] - results["C"]["J"]
        net_ratio = dominant_point / max(c_total, 1e-12)
        if dom == "D_owner_bundle":
            dom_note = ("The owner-directed EVIDENCE ARCHITECTURE (with "
                        "v2's local deficit price removed) is the dominant "
                        "positive mechanism -- architecturally the gain "
                        "does NOT come from the deficit price alone.  This "
                        "is the BUNDLE (topology + flat index), NOT pure "
                        "routing (advice/017 section 13, advice/018 "
                        "section 7).")
        elif dom == "D_pi":
            dom_note = ("The DETECTION-DEFICIT task coordination (the "
                        "detection-deficit price pi_q = y_q/(D_q+eps)) is the "
                        "dominant positive mechanism -- a first-tier "
                        "ALGORITHM contribution (advice/017 section 13).")
        elif dom == "D_lambda":
            dom_note = ("THE RECEIVER-CAPACITY (airtime) price is the "
                        "dominant positive mechanism -- the congestion "
                        "price is what pays for the delay gain here.")
        else:
            dom_note = ("the density admission is the dominant positive "
                        "mechanism at this cell.")
    else:
        dominant = None
        dominant_point = 0.0
        dominant_rel = 0.0
        c_total = results["A"]["J"] - results["C"]["J"]
        net_ratio = 0.0
        dom_note = ("NO positive mechanism dominates: every consecutive "
                    "delta is <= 0 at this cell (all mechanisms either "
                    "harm or are ~neutral on worst-target delay) -- the "
                    "advice/017 section 11 ladder verdict does NOT apply.")
    gate = {
        "objective": "held-out matched-policy worst-target E[T|H1] pooled",
        "arms": results,
        "deltas": deltas,
        "risk_adjusted_deltas": risk_deltas,
        "mechanism_2x2": mechanism_2x2,
        "ladder": ladder,
        "dominant_mechanism": {
            "key": dominant, "point": dominant_point,
            "rel": dominant_rel,
            "ratio_to_net_A_to_C_gain": float(net_ratio),
            "note": dom_note,
        },
        "interpretation": (
            "The largest CERTIFIED positive delta names the mechanism that "
            "explains most of the CA-FRIDS gain at the frozen operating "
            "point (advice/018 section 3: only CI_lo > 0 qualifies).  "
            "Whichever of {D_owner_bundle, D_pi, D_lambda, D_admission} "
            "dominates is the mechanism to lead with in the paper -- if "
            "D_owner_bundle it is the ARCHITECTURE BUNDLE (owner-directed "
            "evidence plane + removal of v2's local deficit price, NOT "
            "pure routing), if D_pi it is the detection-deficit task "
            "coordination, if D_lambda it is the receiver-capacity "
            "steering, if D_admission it is the density admission.  The "
            "ratio_to_net_A_to_C_gain is the dominant sequential increment "
            "relative to the FINAL NET A->C delay improvement -- a "
            "sequential-increment ratio, NOT a causal contribution share, "
            "because the ladder is order-dependent and is not a Shapley "
            "decomposition (advice/018 section 4).  The risk_adjusted_deltas "
            "block re-checks every delta on the error-aware estimand "
            "J_risk = max_q E[T_q^risk|H1] (an H0 declaration under H1 is "
            "charged T_max, advice/018 section 5), so a gain certified on "
            "plain J must also be certified on J_risk to be reported as a "
            "useful-evidence gain rather than a bought-by-misses one."),
        "caveats": [
            "The ladder runs at the FROZEN policy-B operating point "
            "(delta=1, shared by every arm), so it answers the "
            "mechanism-attribution question ONLY at that operating point.  "
            "It is NOT the matched-QoS frontier (P4.2b answers that) and "
            "NOT an anytime-valid QoS certificate (P4.2/registered do "
            "that): every arm QoS is EXPECTED to be UNCERTAIN here.",
            "D_lambda may be negative (the receiver airtime price steering "
            "can HURT worst-target E[T|H1] at the frozen point): per "
            "advice/017 section 7, lambda is registered as a "
            "feasibility/capacity-steering mechanism (and P4.1b found "
            "airtime value in the feasibility/comm-efficiency layer, not "
            "the delay layer), so a negative delay delta at one cell is "
            "the expected honest reading -- the paper must NOT claim "
            "lambda reduces stopping delay.",
            "D_owner_bundle is NOT a pure architecture-only ablation (audit "
            "finding, advice/017 section 13): arm A (FRIDS-v2) is the "
            "full-mesh local-replica broadcast with its OWN local deficit "
            "price, while arm B00 is the CA owner-directed evidence plane "
            "WITH the deficit price removed (flat index).  "
            "D_owner_bundle bundles two changes: (i) topology full-mesh -> "
            "owner-point routing, (ii) dropping v2's local deficit price.  "
            "So the honest reading is \\\"owner-directed evidence plane "
            "(deficit price removed)\\\", NOT \\\"owner routing alone\\\".  "
            "If D_owner_bundle dominates, the paper must say: "
            "owner-directed architecture AND the removal of v2's "
            "local-deficit steering together explain the gain -- do NOT "
            "claim pure routing.  The 2x2 core table (F0/F1/O0/O1) in "
            "``mechanism_2x2`` removes this ambiguity by isolating the "
            "owner effect at FLAT information (delta_architecture_flat).",
        ],
    }
    params = {
        "gate_id_base": "p5a-ablation-ladder",
        "scale": [k, q], "geom": args.geom, "regime": "congested",
        "rho": args.rho,
        "test_episodes": args.test_cell_runs * args.test_mc,
        "max_steps": args.max_steps, "alpha": args.alpha, "beta": args.beta,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode,
        "git_commit": _git_sha(), "git_dirty": _git_dirty(),
        "config_hash": _config_hash(args),
        "seed_scheme": (
            "FRESH held-out seed namespace 400000+mc shared by ALL arms "
            "(disjoint from P4.2 cert 100000, P4.1b discovery 7, threshold "
            "calibration 100, P4.2b cal 200000 / test 300000), so every "
            "consecutive delta is a genuine PAIRED comparison"),
        "frozen": ["geom=2 congested rho=1.8 (registered boundary cell)",
                   "FRIDS-v2 / CA-FRIDS schedulers unchanged",
                   "FIXED calibrated policy-B stopping thresholds "
                   "(delta=1, same for every arm -- scheduler-only / "
                   "mechanism-only comparison)",
                   "no scheduler / price / geometry / seed tuning"],
    }
    payload = {
        "gate_id": "p5a-ablation-ladder",
        "params": params,
        "metrics": gate,
        "runtime_s": round(time.time() - t0, 1),
        "provenance": {
            "git_commit": _git_sha(), "git_tree": _git_tree(),
            "git_dirty": _git_dirty(), "config_hash": _config_hash(args),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()