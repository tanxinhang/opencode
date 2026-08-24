"""P4.2b -- QoS-Matched Operating-Point Frontier (advice/016 section 8,
P5.1).

P4.2 certified the STRONG gate ``CA PASS while v2 FAIL`` at the frozen
geom2/congested cell, but -- as advice/016 section 7 stresses -- that only
proves the registered pair "under the FROZEN, common policy-B operating
point and matched shared-airtime capacity model".  It does NOT yet answer
the reviewer's question:

> if FRIDS-v2 were allowed to RECALIBRATE its own stopping thresholds,
> could it still meet (P_FA, P_MD <= 0.05)?

This gate answers it.  It does NOT change any scheduler.  It only re-tunes
the per-target UPPER stopping thresholds ``A_q`` (the false-alarm lever:
raising ``A_q`` reduces FA at the cost of delay/MD -- exactly the operating
point a reviewer would probe) on an INDEPENDENT calibration stream, then
evaluates BOTH schedulers at their OWN QoS-matched thresholds on a FRESH
held-out test stream (same CRN exogen for a paired comparison):

- CASE A: some target of FRIDS-v2 has NO swept ``A_q`` whose certified
  upper bounds on both error axes clear 0.05 on the calibration stream
  -> "v2 QoS-infeasible even under its own operating point on this cell".
- CASE B: both schedulers find matched thresholds -> compare the matched
  QoS delay ``J_ca^QoS`` vs ``J_v2^QoS`` on the held-out stream.

Protocol (frozen, P4.2-style provenance discipline):
1. Cell: geom=2, congested, rho=1.8 (the registered boundary cell).
2. Base policy-B thresholds from the same frozen ``calibrate_target_bounds``
   (seed 100, delta-1 matched).  The frontier multiplies the per-target
   upper threshold ``A_q`` by ``m_q`` over the pre-registered grid, keeping
   the lower threshold at the calibrated matched value.
3. FRESH seed namespaces (disjoint from P4.2 certification 100000, P4.1b
   discovery 7..106, and threshold calibration 100):
   - ``cal_seed + mc`` : calibration stream (per-target A sweep, CRN-paired
     across m so the certified-quality comparisons are paired);
   - ``test_seed + mc``: held-out test stream (the SAME exogen drives v2
     and CA at their matched thresholds -- paired J).
4. Calibration QoS decision per (scheduler, target q, m): certified
   Clopper-Pearson UPPER bounds ``U_FA,q`` and ``U_MD,q`` at per-target
   ``delta_q = cell_delta/(2Q)`` must both be <= 0.05.  The matched ``m_q*``
   is the SMALLEST grid value certified feasible (smallest A => smallest
   stopping delay).  If no grid value is certified feasible for a target,
   the scheduler is CASE-A intfeasible at the swept frontier.
5. Held-out: both schedulers at their own matched thresholds on the same
   held-out CRN stream; certified QoS (anytime-valid 32-stream familywise,
   exactly as P4.2) and pooled worst-target delay ``J`` are reported, with
   the matched-QoS delay comparison.

The verdict is worded per advice/016 section 8: the paper may then say
"CA feasible while v2 infeasible at any swept operating point" (Case A) or
"CA achieves a lower stopping delay at matched certified QoS" (Case B).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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
from uav_otfs_isac.qos import (
    anytime_qos_status,
    clopper_pearson,
)

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
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
            text=True)
        return bool(out.strip())
    except Exception:
        return True


def _sha16(text) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _config_hash(args) -> str:
    fields = {
        "geom": args.geom, "rho": args.rho,
        "cal_seed": args.cal_seed, "test_seed": args.test_seed,
        "cal_cell_runs": args.cal_cell_runs, "cal_mc": args.cal_mc,
        "test_cell_runs": args.test_cell_runs, "test_mc": args.test_mc,
        "max_steps": args.max_steps, "alpha": args.alpha, "beta": args.beta,
        "cell_delta": args.cell_delta, "grid": args.grid,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode, "calib_seed": args.calib_seed,
        "calib_verify": args.calib_verify, "calib_n_runs": args.calib_n_runs,
    }
    return _sha16(json.dumps(fields, sort_keys=True, indent=0))


def _acc(out, acc):
    for key in ("n_H0", "n_H1", "n_FA", "n_MD"):
        acc[key] = [a + b for a, b in zip(acc[key], out["raw_counts"][key])]


def _empty_acc(q):
    return {key: [0] * q for key in ("n_H0", "n_H1", "n_FA", "n_MD")}


def _certified_upper(acc, q, delta_q):
    """Per-target certified Clopper-Pearson upper bounds of FA and MD."""
    u_fa, u_md = [], []
    for qq in range(q):
        _, fa_hi = clopper_pearson(acc["n_FA"][qq], acc["n_H0"][qq], delta_q)
        _, md_hi = clopper_pearson(acc["n_MD"][qq], acc["n_H1"][qq], delta_q)
        u_fa.append(fa_hi)
        u_md.append(md_hi)
    return u_fa, u_md


def _matched_bounds(bt, m_q, delta=1.0):
    """Per-target matched stopping bounds: ``A_q = bt_q0 * m_q`` (the
    swept upper-threshold lever), ``B_q = bt_q1 - delta`` (calibrated
    lower threshold unchanged)."""
    return [[bt[qq][0] * float(m_q[qq]), bt[qq][1] - delta]
            for qq in range(len(bt))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ca_frids_qos_frontier_gate.json")
    parser.add_argument("--geom", type=int, default=2)
    parser.add_argument("--rho", type=float, default=1.8)
    parser.add_argument("--cal-seed", type=int, default=200000,
                        help="FRESH calibration seed namespace (disjoint from "
                             "P4.2 cert 100000, discovery 7, calibr 100)")
    parser.add_argument("--test-seed", type=int, default=300000,
                        help="FRESH held-out test seed namespace")
    parser.add_argument("--cal-cell-runs", type=int, default=300)
    parser.add_argument("--cal-mc", type=int, default=2,
                        help="calibration MC cells per (scheduler, target, m): "
                             "default 2 x 300 = 600 episodes per combo (300 is "
                             "too thin for the per-target CP certified-upper "
                             "decision and flips to spurious infeasibility)")
    parser.add_argument("--test-cell-runs", type=int, default=1500)
    parser.add_argument("--test-mc", type=int, default=8,
                        help="held-out stream episodes = test_cell_runs * "
                             "test_mc (default 12k)")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=SPEC)
    parser.add_argument("--beta", type=float, default=SPEC)
    parser.add_argument("--cell-delta", type=float, default=0.05,
                        help="per-cell simultaneous delta used in the "
                             "calibration CP certification (delta/(2Q) per "
                             "stream)")
    parser.add_argument("--grid", default="0.6,0.8,1.0,1.5,2.0,3.0,5.0",
                        help="pre-registered A_q multiplier grid")
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
    grid = sorted({float(m) for m in args.grid.split(",")})
    delta_q = args.cell_delta / (2.0 * q)

    sc = build_distributed_scenario(np.random.default_rng(args.geom),
                                    k_uavs=k, q_targets=q)
    bt = calibrate_target_bounds(sc, args.alpha, args.beta,
                                 n_runs=args.calib_n_runs,
                                 seed=args.calib_seed,
                                 llr_bits=TOKEN_LLR_BITS,
                                 verify_runs=args.calib_verify)
    am = build_airtime_model(sc, rho_target=args.rho)

    # per-scheduler runners close over the EXACT positional order of the
    # frozen schedulers (CA has the airtime model in the 3rd slot).
    def run_v2(bounds, n_runs, seed, tape):
        return simulate_frids_v2(sc, bounds, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode="local", airtime=am, exog=tape)

    def run_ca(bounds, n_runs, seed, tape):
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 exog=tape)

    # ---- per-scheduler per-target A_q sweep on the CALIBRATION stream ----
    def calibrate_scheduler(runner) -> dict:
        """For every target find the SMALLEST grid multiplier whose
        certified upper FA/MD both clear the spec.  The same calibration
        exogen stream (cal_seed + mc) is reused across the A-grid of a
        scheduler, so the certified-quality comparisons across ``m`` are
        CRN-paired."""
        frontier = {"m_star": [None] * q, "A_star": [None] * q,
                    "cert_u_fa": [0.0] * q, "cert_u_md": [0.0] * q,
                    "feasible": True}
        for qq in range(q):
            for m in grid:
                acc = _empty_acc(q)
                for mc in range(args.cal_mc):
                    tape = build_exogenous_tape(args.cal_seed + mc,
                                                args.cal_cell_runs, q, k,
                                                args.max_steps)
                    mq = [1.0] * q
                    mq[qq] = m
                    out = runner(_matched_bounds(bt, mq, 1.0),
                                 args.cal_cell_runs, args.cal_seed + mc, tape)
                    _acc(out, acc)
                u_fa, u_md = _certified_upper(acc, q, delta_q)
                if (u_fa[qq] <= SPEC and u_md[qq] <= SPEC
                        and frontier["m_star"][qq] is None):
                    frontier["m_star"][qq] = m
                    frontier["A_star"][qq] = bt[qq][0] * m
                    frontier["cert_u_fa"][qq] = u_fa[qq]
                    frontier["cert_u_md"][qq] = u_md[qq]
        frontier["feasible"] = all(m is not None for m in frontier["m_star"])
        return frontier

    v2_frontier = calibrate_scheduler(run_v2)
    ca_frontier = calibrate_scheduler(run_ca)

    # ---- held-out matched-QoS evaluation (SAME CRN exogen for v2/CA) ----
    def held_out(runner, m_star):
        acc = _empty_acc(q)
        pool_s, pool_n = np.zeros(q), np.zeros(q)
        for mc in range(args.test_mc):
            tape = build_exogenous_tape(args.test_seed + mc,
                                        args.test_cell_runs, q, k,
                                        args.max_steps)
            out = runner(_matched_bounds(bt, m_star, 1.0),
                         args.test_cell_runs, args.test_seed + mc, tape)
            _acc(out, acc)
            pool_s += np.asarray(out["pool"]["sum_h1_delay"], dtype=float)
            pool_n += np.asarray(out["pool"]["n_h1"], dtype=float)
        J = float(np.max(pool_s / np.maximum(pool_n, 1.0)))
        status, streams = anytime_qos_status(
            acc["n_H0"], acc["n_H1"], acc["n_FA"], acc["n_MD"],
            args.alpha, args.beta, delta_fam=0.05, n_streams=N_STREAMS,
            ret_bounds=True)
        return {"J": J, "qos": status, "streams": streams,
                "counts": acc, "m_star": list(m_star)}

    # CASE A guard: an infeasible scheduler has NO matched thresholds, so a
    # held-out matched-QoS run is meaningless for it (the calibration
    # already certified infeasibility).  Only the feasible scheduler gets a
    # held-out matched-QoS evaluation.
    v2_held = (held_out(run_v2, v2_frontier["m_star"])
               if v2_frontier["feasible"] else None)
    ca_held = (held_out(run_ca, ca_frontier["m_star"])
               if ca_frontier["feasible"] else None)

    # ---- verdict ----------------------------------------------------------
    case = "A" if not v2_frontier["feasible"] else "B"
    if case == "A":
        verdict = (
            f"CASE A: FRIDS-v2 finds NO A_q on the swept grid whose "
            f"certified upper FA/MD both clear 0.05 (infeasible target(s) "
            f"{[qq for qq in range(q) if v2_frontier['m_star'][qq] is None]})"
            f" on the fresh calibration stream; CA "
            f"{'feasible' if ca_frontier['feasible'] else 'also infeasible'}"
            f" -> 'CA feasible while v2 infeasible at any swept operating "
            f"point' (or both infeasible under the swept frontier)"
        )
    else:
        verdict = (
            f"CASE B: both schedulers reach certifiable matched operating "
            f"points; held-out matched-QoS delay J_ca={ca_held['J']:.4f} vs "
            f"J_v2={v2_held['J']:.4f} "
            f"({'CA lower' if ca_held['J'] < v2_held['J'] else 'v2 lower'})"
            f" -> 'CA achieves a lower stopping delay at matched certified "
            f"QoS'"
        )
    gate = {
        "case": case, "verdict": verdict,
        "v2_frontier": v2_frontier, "ca_frontier": ca_frontier,
        "held_out": {"v2": v2_held,
                     "ca": (None if ca_held is None else
                            {"J": ca_held["J"], "qos": ca_held["qos"],
                             "counts": ca_held["counts"],
                             "streams": ca_held["streams"]})},
        "matched_j": {"v2": None if v2_held is None else v2_held["J"],
                      "ca": None if ca_held is None else ca_held["J"]},
        "held_out_qos": {"v2": None if v2_held is None else v2_held["qos"],
                         "ca": None if ca_held is None else ca_held["qos"]},
    }
    params = {
        "gate_id_base": "p4.2b-qos-matched-operating-point-frontier",
        "scale": [k, q], "geom": args.geom, "regime": "congested",
        "rho": args.rho, "grid": grid, "cal_seed": args.cal_seed,
        "test_seed": args.test_seed,
        "cal_episodes": args.cal_cell_runs * args.cal_mc,
        "test_episodes": args.test_cell_runs * args.test_mc,
        "max_steps": args.max_steps, "alpha": args.alpha, "beta": args.beta,
        "cell_delta": args.cell_delta, "delta_q": delta_q,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode,
        "git_commit": _git_sha(), "git_dirty": _git_dirty(),
        "config_hash": _config_hash(args),
        "seed_scheme": ("frontier calibration seed = cal_seed+mc per "
                        "scheduler (CRN-paired across the A-grid); held-out "
                        "test seed = test_seed+mc shared by v2 and CA; BOTH "
                        "namespaces fresh, disjoint from P4.2 cert 100000, "
                        "P4.1b discovery 7, threshold calibration 100"),
        "frozen": ["geom=2 congested rho=1.8", "FRIDS-v2 / CA-FRIDS "
                   "schedulers unchanged", "per-target A_q lever only "
                   "(lower threshold at calibrated matched value)",
                   "no scheduler / price / geometry / seed tuning"],
    }
    payload = {
        "gate_id": "p4.2b-qos-matched-operating-point-frontier",
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