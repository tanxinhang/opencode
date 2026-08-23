"""P4.2 ANYTIME-valid QoS certification gate (advice/015).

The verdict of P4.1b (``ca_frids_gate_formal.json``) is CODE/CERTIFICATE/
PHYSICAL-MODEL(-conditional) PASS but not full-theory PASS, and the
single most valuable next step is the STATISTICAL certification upgrade:
P4.2 certifies the boundary cell on a FRESH stream with TRUE anytime-valid
(time-uniform) Bernoulli confidence sequences instead of the frozen
Clopper-Pearson + alpha-spending scheme.

The P4.2 protocol is FROZEN exactly as advised (advice/015 sections 3-4):

1. Frozen objects: ``geom=2, regime=congested, rho=1.8``, code at HEAD,
   fixed policy-B thresholds (``calibrate_target_bounds`` at seed 100,
   ``delta=1`` matched B), global-simplex task price, ``pi_bits=lam_bits=10``.
2. Fresh certification seed namespace: ``cert_seed_base + mc`` is
   DISJOINT from the P4.1b discovery seeds (calibration seed 100, gate
   seed 7), so the P4.1b data do the boundary-cell discovery and the P4.2
   data do the independent confirmation -- no post-selection bias.
3. One pre-registered 60k-episode stream, read at cumulative PREFIXES
   ``0:7.5k -> 0:15k -> 0:30k -> 0:60k`` (never four fresh randomizations).
4. All ``2 algorithms x 8 targets x 2 errors = 32`` Bernoulli streams are
   certified simultaneously with total familywise ``delta = 0.05``:
   every stream gets ``delta_s = 0.05/32`` (union bound; CRN needs no
   independence).
5. Frozen decisions at every prefix:
      PASS(A)  <=>  max_q U^A_FA,q <= 0.05  and  max_q U^A_MD,q <= 0.05
      FAIL(A)  <=>  exists q: L^A_FA,q > 0.05  or  L^A_MD,q > 0.05
   with the STRONG gate ``PASS(CA) and FAIL(v2)``; early stopping is
   legal once it is achieved, and at 60k a still-UNCERTAIN cell is
   reported as "statistical resolution insufficient" -- no unplanned MC
   increase.
6. Every stage is serialized: incremental + cumulative ``n_H0/n_H1/n_FA/
   n_MD``, per-target CS bounds, first crossing stage, scenario/threshold/
   code-tree/seed-interval/CRN-tape-fingerprint provenance.

The certificate itself is the Beta-mixture e-process of advice/015
section 2 (``uav_otfs_isac.qos.beta_mixture_cs``): for every stream the
confidence set ``C_n = {p : M_n(p) < 1/delta_s}`` with ``M_n`` the
Beta(1/2,1/2)-mixture likelihood ratio.  Ville's inequality makes it
time-uniform, so the four prefix looks and even a data-dependent early
stop need NO alpha-spending.
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
from uav_otfs_isac.qos import anytime_qos_status

SPEC_FA = 0.05
SPEC_MD = 0.05
N_STREAMS = 32            # 2 algorithms x 8 targets x 2 error axes
SCALE = (16, 8)


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def _sha16(text) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def _config_hash(args) -> str:
    fields = {
        "geom": args.geom, "rho": args.rho, "cell_runs": args.cell_runs,
        "mc_seeds": args.mc_seeds, "cert_seed": args.cert_seed,
        "max_steps": args.max_steps, "alpha": args.alpha, "beta": args.beta,
        "delta_fam": args.delta_fam, "n_streams": N_STREAMS,
        "mixture_a": 0.5, "mixture_b": 0.5,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode, "calib_seed": args.calib_seed,
        "calib_verify": args.calib_verify, "calib_n_runs": args.calib_n_runs,
        "prefix_cells": args.prefix_cells,
    }
    return _sha16(json.dumps(fields, sort_keys=True, indent=0))


def _scenario_hash(sc) -> str:
    return _sha16(json.dumps(_jsonable(sc), sort_keys=True))


def _bounds_hash(bounds) -> str:
    return _sha16(json.dumps(_jsonable(bounds), sort_keys=True))


def _tape_fingerprint(tape) -> str:
    """Deterministic CRN-tape fingerprint (advice/015 section 4 item 5):
    the seed/shape header plus a strided digest of every uniform block's
    actual bytes -- cheap enough to serialize per stage and strong enough
    to pin the exact exogenous realization."""
    h = hashlib.sha256()
    h.update(f"{tape.seed}:{tape.n_runs}:{tape.q}:{tape.k}:"
             f"{tape.max_steps}".encode("utf-8"))
    for name, arr in (("H", tape.U_H), ("obs", tape.U_obs),
                      ("link", tape.U_link), ("mfac", tape.U_mfac),
                      ("adm", tape.U_adm), ("adm_x", tape.U_adm_extra),
                      ("pol", tape.U_policy)):
        flat = np.asarray(arr).reshape(-1)
        n = int(flat.size)
        stride = max(1, int(n // 997))
        idx = np.arange(0, n, stride) % n
        h.update(name.encode("utf-8"))
        h.update(flat[np.sort(idx)].tobytes())
    return h.hexdigest()[:16]


def matched_bounds(bt, q, delta=1.0):
    """FROZEN fixed calibrated policy B (delta 1) -- the same two-threshold
    stopping policy for BOTH schedulers (the fair scheduler-only
    comparison of the P3.4 gate)."""
    return [[bt[qq][0], bt[qq][1] - delta] for qq in range(q)]


def _acc(out, acc):
    for key in ("n_H0", "n_H1", "n_FA", "n_MD"):
        acc[key] = [a + b for a, b in zip(acc[key], out["raw_counts"][key])]


def _empty_acc(q):
    return {key: [0] * q for key in ("n_H0", "n_H1", "n_FA", "n_MD")}


def _j_pooled(acc_pool):
    n = np.asarray(acc_pool["n_h1"], dtype=float)
    s = np.asarray(acc_pool["sum_h1_delay"], dtype=float)
    values = np.where(n > 0, s / np.maximum(n, 1e-12), 0.0)
    return float(np.max(values)) if len(values) else 0.0


def _acc_pool(out, acc):
    for key in ("n_h1", "sum_h1_delay"):
        acc[key] = acc[key] + np.asarray(out["pool"][key], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ca_frids_anytime_cert_gate.json")
    parser.add_argument("--geom", type=int, default=2,
                        help="frozen boundary geometry (advice/015: geom=2)")
    parser.add_argument("--rho", type=float, default=1.8,
                        help="frozen congested airtime rho_target")
    parser.add_argument("--cert-seed", type=int, default=100000,
                        help="FRESH certification seed namespace (disjoint "
                             "from calibration seed 100 and gate seed 7)")
    parser.add_argument("--cell-runs", type=int, default=1500,
                        help="episodes per MC cell (the tape is chunked for "
                             "memory; prefixes are cumulative cells)")
    parser.add_argument("--mc-seeds", type=int, default=40,
                        help="MC cells; total episodes = cell_runs * mc_seeds "
                             "(default 40*1500 = 60k)")
    parser.add_argument("--prefix-cells", default="5,10,20,40",
                        help="cumulative MC-cell prefixes (default maps to "
                             "7.5k/15k/30k/60k at 1500 runs/cell)")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=SPEC_FA)
    parser.add_argument("--beta", type=float, default=SPEC_MD)
    parser.add_argument("--delta-fam", type=float, default=0.05,
                        help="total familywise level over the 32 streams")
    parser.add_argument("--pi-bits", type=int, default=10)
    parser.add_argument("--lam-bits", type=int, default=10)
    parser.add_argument("--price-mode", default="global_simplex",
                        choices=("global_simplex", "owner_local"))
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    parser.add_argument("--calib-n-runs", type=int, default=300)
    parser.add_argument("--no-early-stop", action="store_true",
                        help="run every prefix even after the strong gate")
    args = parser.parse_args()
    t0 = time.time()

    k, q = SCALE
    delta_s = args.delta_fam / N_STREAMS
    prefixes = sorted({int(p) for p in args.prefix_cells.split(",")}
                      | {args.mc_seeds})
    prefixes = [p for p in prefixes if 0 < p <= args.mc_seeds]

    # ---- frozen certification cell (advice/015 section 4 item 1) --------
    sc = build_distributed_scenario(np.random.default_rng(args.geom),
                                    k_uavs=k, q_targets=q)
    bt = calibrate_target_bounds(sc, args.alpha, args.beta,
                                 n_runs=args.calib_n_runs,
                                 seed=args.calib_seed,
                                 llr_bits=TOKEN_LLR_BITS,
                                 verify_runs=args.calib_verify)
    bounds = matched_bounds(bt, q, 1.0)
    am = build_airtime_model(sc, rho_target=args.rho)

    acc_v2 = _empty_acc(q)
    acc_ca = _empty_acc(q)
    pool_v2 = {"n_h1": np.zeros(q, dtype=float),
               "sum_h1_delay": np.zeros(q, dtype=float)}
    pool_ca = {"n_h1": np.zeros(q, dtype=float),
               "sum_h1_delay": np.zeros(q, dtype=float)}
    stages = []
    first_cross = {"v2": None, "ca": None}
    done_at = None
    strong_gate_at = None

    for mc in range(args.mc_seeds):
        cell_seed = args.cert_seed + mc          # ONE pre-registered stream
        tape = build_exogenous_tape(cell_seed, args.cell_runs, q, k,
                                    args.max_steps)
        fp = _tape_fingerprint(tape)
        inc_v2 = _empty_acc(q)
        inc_ca = _empty_acc(q)
        out_v2 = simulate_frids_v2(
            sc, bounds, n_runs=args.cell_runs, seed=cell_seed,
            max_steps=args.max_steps, raw_counts=True,
            price_mode="local", airtime=am, exog=tape)
        _acc(out_v2, acc_v2)
        _acc(out_v2, inc_v2)
        _acc_pool(out_v2, pool_v2)
        out_ca = simulate_ca_frids(
            sc, bounds, am, n_runs=args.cell_runs, seed=cell_seed,
            max_steps=args.max_steps, raw_counts=True,
            price_mode=args.price_mode, pi_bits=args.pi_bits,
            lam_bits=args.lam_bits, exog=tape)
        _acc(out_ca, acc_ca)
        _acc(out_ca, inc_ca)
        _acc_pool(out_ca, pool_ca)

        if (mc + 1) in prefixes:
            idx = len(stages)
            status_v2, st_v2 = anytime_qos_status(
                acc_v2["n_H0"], acc_v2["n_H1"], acc_v2["n_FA"],
                acc_v2["n_MD"], args.alpha, args.beta,
                delta_fam=args.delta_fam, n_streams=N_STREAMS,
                ret_bounds=True)
            status_ca, st_ca = anytime_qos_status(
                acc_ca["n_H0"], acc_ca["n_H1"], acc_ca["n_FA"],
                acc_ca["n_MD"], args.alpha, args.beta,
                delta_fam=args.delta_fam, n_streams=N_STREAMS,
                ret_bounds=True)
            for algo, st, acc in (("v2", st_v2, acc_v2),
                                  ("ca", st_ca, acc_ca)):
                if first_cross[algo] is None:
                    first_cross[algo] = {
                        "FA": {qq: None for qq in range(q)},
                        "MD": {qq: None for qq in range(q)}}
                for err in ("FA", "MD"):
                    for qq in range(q):
                        if (first_cross[algo][err][qq] is None
                                and st[err + "_lo"][qq] > (SPEC_FA if err
                                                           == "FA"
                                                           else SPEC_MD)):
                            first_cross[algo][err][qq] = idx
            v2_pass = status_v2 == "PASS"
            ca_pass = status_ca == "PASS"
            v2_fail = status_v2 == "FAIL"
            ca_fail = status_ca == "FAIL"
            strong = ca_pass and v2_fail
            stages.append({
                "stage": idx, "prefix_cells": mc + 1,
                "episodes": (mc + 1) * args.cell_runs,
                "delta_s": delta_s,
                "incremental": {"v2": inc_v2, "ca": inc_ca},
                "cumulative": {"v2": dict(acc_v2), "ca": dict(acc_ca)},
                "streams": {"v2": st_v2, "ca": st_ca},
                "status": {"v2": status_v2, "ca": status_ca},
                "gate": {"v2_pass": v2_pass, "ca_pass": ca_pass,
                         "v2_fail": v2_fail, "ca_fail": ca_fail,
                         "strong_gate": strong},
                "j_v2_pooled": _j_pooled(pool_v2),
                "j_ca_pooled": _j_pooled(pool_ca),
                "crn_tape_fingerprint": fp,
            })
            if strong and strong_gate_at is None:
                strong_gate_at = idx
                done_at = idx
            print(f"stage {idx} at {(mc+1)*args.cell_runs} episodes: "
                  f"v2={status_v2} ca={status_ca} "
                  f"(strong={strong}) [{time.time()-t0:.0f}s]", flush=True)
            if done_at is not None and not args.no_early_stop:
                break

    # ---- final gate ------------------------------------------------------
    last = stages[-1]
    final_v2 = last["status"]["v2"]
    final_ca = last["status"]["ca"]
    resolution_insufficient = not last["gate"]["strong_gate"]
    if last["gate"]["v2_fail"] and final_ca == "UNCERTAIN":
        resolution_insufficient = True
    gate = {
        "strong_gate": bool(strong_gate_at is not None),
        "strong_gate_at_stage": (strong_gate_at if strong_gate_at is not None
                                 else None),
        "strong_gate_episodes": (stages[strong_gate_at]["episodes"]
                                 if strong_gate_at is not None else None),
        "ca_final": final_ca,
        "v2_final": final_v2,
        "ca_pass": bool(last["gate"]["ca_pass"]),
        "v2_fail": bool(last["gate"]["v2_fail"]),
        "v2_pass": bool(last["gate"]["v2_pass"]),
        "ca_fail": bool(last["gate"]["ca_fail"]),
        "resolution_insufficient": bool(resolution_insufficient),
        "first_crossing_stage": first_cross,
        "verdict": (
            f"P4.2 STRONG gate achieved: CA certified PASS while v2 "
            f"certified FAIL at stage {strong_gate_at} "
            f"({stages[strong_gate_at]['episodes']} episodes)"
            if strong_gate_at is not None
            else (f"at {last['episodes']} episodes CA={final_ca}, "
                  f"v2={final_v2}: statistical resolution insufficient at "
                  "the frozen budget; no unplanned MC increase")
        ),
    }
    params = {
        "scale": [k, q], "geom": args.geom, "regime": "congested",
        "rho": args.rho, "cell_runs": args.cell_runs,
        "mc_seeds": args.mc_seeds, "prefix_cells": prefixes,
        "prefix_episodes": [int(p * args.cell_runs) for p in prefixes],
        "cert_seed": args.cert_seed, "max_steps": args.max_steps,
        "alpha": args.alpha, "beta": args.beta,
        "delta_family": args.delta_fam, "n_streams": N_STREAMS,
        "delta_per_stream": delta_s, "mixture_a": 0.5, "mixture_b": 0.5,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode,
        "calib_seed": args.calib_seed, "calib_verify": args.calib_verify,
        "calib_n_runs": args.calib_n_runs,
        "scenario_hash": _scenario_hash(sc),
        "bounds_hash": _bounds_hash(bounds),
        "git_commit": _git_sha(), "git_tree": _git_tree(),
        "git_dirty": _git_dirty(), "config_hash": _config_hash(args),
        "seed_scheme": ("FRESH certification seed namespace: "
                        "tape seed = cert_seed + mc (disjoint from "
                        "P4.1b discovery seeds 7 and 100); ONE "
                        "pre-registered stream, prefixes read cumulatively "
                        "-- never four fresh randomizations"),
        "capacity_model": ("shared airtime primitive: sum tau_ij/T_air <= 1; "
                           "offer -> admission -> link; v2 neutral, CA "
                           "density admission (frozen P4.1b code)"),
        "evidence_mode": "owner-only evidence plane (Dual-Bus)",
        "protocol": [
            "FIXED calibrated policy B (delta 1, same stopping policy for "
            "both schedulers)",
            "TRUE anytime-valid certification: Beta(1/2,1/2)-mixture "
            "e-process M_n(p), Ville inequality, delta_s = 0.05/32 per "
            "stream (Bonferroni over 32 streams; union bound, CRN ok)",
            "PASS(A) iff max_q U_FA,q <= 0.05 and max_q U_MD,q <= 0.05; "
            "FAIL(A) iff exists q: L_FA,q > 0.05 or L_MD,q > 0.05",
            "pre-registered prefix looks 0:7.5k/15k/30k/60k on ONE stream; "
            "early stopping legal once PASS(CA) and FAIL(v2) hold",
            "at 60k a UNCERTAIN cell is reported as resolution "
            "insufficient; no unplanned MC increase"],
        "frozen": ["geom=2 congested rho=1.8", "FRIDS-v2",
                   "global-simplex task price", "pi_bits=lam_bits=10",
                   "two-threshold stopping", "calibrated policy-matched B",
                   "no scheduler/threshold/geometry/seed changes"],
    }
    payload = {
        "gate_id": "p4.2-anytime-valid-qos-cert",
        "params": params,
        "metrics": gate,
        "stages": stages,
        "runtime_s": round(time.time() - t0, 1),
        "provenance": {
            "git_commit": _git_sha(), "git_tree": _git_tree(),
            "git_dirty": _git_dirty(), "config_hash": _config_hash(args),
            "scenario_hash": _scenario_hash(sc),
            "bounds_hash": _bounds_hash(bounds),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()