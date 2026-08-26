"""P5-MIN Minimality Gate, cross-seed (advice/019 section 7, multi-seed
robustness of the advice/019 section 5 B0-lite reparameterization).

The registered ladder (``run_p5a_ablation_ladder.py``) attributes the CA
gain on ONE frozen cell (geom=2, rho=1.8, scale=(16,8), test_seed=400000).
That answers mechanism attribution at a single operating point; it does
NOT certify that the minimal algorithm -- B0-lite (Normalization-Free
Distributed Deficit Pricing, advice/019 section 5) -- holds up across
independent geometry / congestion / scale / MC draws.  This gate closes
that gap with a cross-seed Minimality test:

    arms (advice/019 section 7 table):
      B0-lite : task price on,  lambda OFF, neutral admission, NORM_FREE
      B0-D    : task price on,  lambda OFF, density admission, NORM_FREE
      B1      : task price on,  lambda ON,  neutral admission
      C       : task price on,  lambda ON,  density admission  (full CA)
      B0      : task price on,  lambda OFF, neutral admission, NORMALIZED
                -- a CONTROL arm: B0-lite vs B0 must be ~policy-equivalent
                (the norm-free reparameterization changes nothing, only
                removes the global-Z reduction and its 10 bits/cycle).

    grid:
      geoms    = {0, 1, 2}                 (independent geometry seeds)
      rho      = {0.7, 1.2, 1.8}           (uncongested / boundary / heavy)
      scales   = {(8,4), (16,8)}
      test     = 3 independent held-out MC seed namespaces, each with its
                 own CRN block tape (all arms of a cell share the tape of
                 their test seed -- paired within seed).

    verdict (advice/019 section 7 final criteria):
      - matched-QoS feasible (per-arm anytime-valid QoS reported; at the
        frozen point arms are EXPECTED UNCERTAIN -- the matched-QoS cert
        is P4.2b's job, this gate is the frozen-point mechanism gate);
      - J_B0-lite <= J_C + delta_J  (cross-seed pooled, and per-seed
        consistency fraction);
      - B_ctrl,B0-lite << B_ctrl,C  (80 vs 250 bits/cycle at (16,8));
      - hard airtime feasibility preserved (budget_feasible_fraction ~ 1);
      - B0-lite ~ B0 (norm-free is a strict reparameterization, so the
        delay delta must be ~0 with a CI that is not a certified change).

Every cell reports pooled J / J_risk / control bits / QoS / budget
feasibility per arm, plus paired per-block bootstrap CIs (blocks
concatenated ACROSS test seeds -- the bootstrap resampling unit is the
block, advice/019 section 8).  The cross-seed verdict is the fraction of
test seeds on which B0-lite beats C within delta_J, and the pooled
paired delta with its CI.

This is the "freeze the algorithm" gate: if B0-lite is not certified
worse than C on delay anywhere in the grid (and control bits drop
3x), the advice/019 recommendation "FREEZE B0-lite as the final
algorithm" is supported by cross-seed evidence rather than one cell.
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
    nested_scenario_subsets,
)
from scripts.run_p5a_ablation_ladder import (
    N_STREAMS,
    SPEC,
    _ci_state,
    _pooled_delta_ci,
    _pooled_j,
    _run_arm,
    cell_sign_consistency,
    hierarchical_block_bootstrap,
)

# FRESH held-out seed namespaces for the gate (disjoint from P4.2 cert
# 100000, P4.1b discovery 7, threshold calibration 100, P4.2b cal
# 200000/test 300000, and the P5-A ladder 400000).  Test seed ``s`` uses
# base ``TEST_BASE + s`` and its CRN blocks are ``TEST_BASE + s + mc``.
TEST_BASE = 600000


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
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
    return _sha16(json.dumps({
        "geoms": args.geoms, "rhos": args.rhos, "scales": args.scales,
        "test_seeds": args.test_seeds,
        "test_cell_runs": args.test_cell_runs, "test_mc": args.test_mc,
        "max_steps": args.max_steps, "alpha": args.alpha, "beta": args.beta,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode, "delta_j": args.delta_j,
        "calib_seed": args.calib_seed, "calib_verify": args.calib_verify,
        "calib_n_runs": args.calib_n_runs,
    }, sort_keys=True, indent=0))


def _pooled_blocks(seed_results, key):
    """Concatenate per-block arrays across the test seeds of one cell."""
    out = []
    for res in seed_results:
        out.extend(res[key])
    return out


def _cell_pooled_j(seed_results):
    """Pooled per-target (N, S) over ALL blocks of ALL test seeds."""
    N = np.sum(np.stack(_pooled_blocks(seed_results, "block_n"), axis=0),
               axis=0)
    S = np.sum(np.stack(_pooled_blocks(seed_results, "block_s"), axis=0),
               axis=0)
    return _pooled_j(N, S)


def _run_mc(args, scales, _scenario, _bounds, _airtime, t0):
    """Run the full MC grid and return the raw per-cell data (before any
    cross-seed aggregation).  Extracted so the gate can cache/restore this
    expensive step (advice/019 multi-seed validation)."""
    cells = []
    for geom in args.geoms:
        for rho in args.rhos:
            for (k, q) in scales:
                sc = _scenario(geom, k, q)
                bounds = _bounds(geom, k, q)
                am = _airtime(geom, k, q, rho)
                cell = {
                    "geom": geom, "rho": rho, "scale": [k, q],
                    "seed_results": [], "arms": {},
                }
                for s in args.test_seeds:
                    exog_blocks = [
                        build_exogenous_tape(TEST_BASE + s + mc,
                                             args.test_cell_runs,
                                             q, k, args.max_steps)
                        for mc in range(args.test_mc)
                    ]

                    def _lite(bounds_, n_runs, seed_, tape):
                        return simulate_ca_frids(
                            sc, bounds_, am, n_runs, seed=seed_,
                            max_steps=args.max_steps, raw_counts=True,
                            price_mode=args.price_mode,
                            pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                            psi_bits=args.pi_bits, psi_lo=args.psi_lo, psi_hi=args.psi_hi,
                            task_price=True, airtime_price=False,
                            norm_free=True, admission_policy="neutral",
                            audit=args.audit, exog=tape)

                    def _b0d(bounds_, n_runs, seed_, tape):
                        return simulate_ca_frids(
                            sc, bounds_, am, n_runs, seed=seed_,
                            max_steps=args.max_steps, raw_counts=True,
                            price_mode=args.price_mode,
                            pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                            task_price=True, airtime_price=False,
                            norm_free=True, admission_policy="density",
                            exog=tape)

                    def _b1(bounds_, n_runs, seed_, tape):
                        return simulate_ca_frids(
                            sc, bounds_, am, n_runs, seed=seed_,
                            max_steps=args.max_steps, raw_counts=True,
                            price_mode=args.price_mode,
                            pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                            task_price=True, airtime_price=True,
                            admission_policy="neutral", exog=tape)

                    def _c(bounds_, n_runs, seed_, tape):
                        return simulate_ca_frids(
                            sc, bounds_, am, n_runs, seed=seed_,
                            max_steps=args.max_steps, raw_counts=True,
                            price_mode=args.price_mode,
                            pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                            task_price=True, airtime_price=True,
                            admission_policy="density", exog=tape)

                    def _b0(bounds_, n_runs, seed_, tape):
                        # normalized control: B0-lite must be ~equivalent
                        return simulate_ca_frids(
                            sc, bounds_, am, n_runs, seed=seed_,
                            max_steps=args.max_steps, raw_counts=True,
                            price_mode=args.price_mode,
                            pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                            task_price=True, airtime_price=False,
                            norm_free=False, admission_policy="neutral",
                            audit=args.audit, exog=tape)

                    seed_res = {"test_seed": s, "arms": {}}
                    for name, runner in (
                        ("B0-lite", _lite), ("B0-D", _b0d),
                        ("B1", _b1), ("C", _c), ("B0", _b0),
                    ):
                        out = _run_arm(runner, bounds, args.test_cell_runs,
                                       TEST_BASE + s, args.max_steps,
                                       exog_blocks)
                        seed_res["arms"][name] = out
                    cell["seed_results"].append(seed_res)
                cells.append(cell)
                print(f"cell geom={geom} rho={rho} scale=({k},{q}) done "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/p5min_robustness_gate.json")
    parser.add_argument("--geoms", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rhos", type=float, nargs="+",
                        default=[0.7, 1.2, 1.8],
                        help="uncongested / boundary / heavy congestion")
    parser.add_argument("--scales", nargs="+", default=["16,8", "8,4"],
                        help="K,Q pairs, e.g. '16,8' '8,4'")
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="independent held-out MC seed indices; base "
                             "seed = 600000 + s, blocks = base + mc")
    parser.add_argument("--test-cell-runs", type=int, default=250)
    parser.add_argument("--test-mc", type=int, default=6,
                        help="CRN blocks per test seed")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=SPEC)
    parser.add_argument("--beta", type=float, default=SPEC)
    parser.add_argument("--pi-bits", type=int, default=10)
    parser.add_argument("--lam-bits", type=int, default=10)
    parser.add_argument("--psi-lo", type=float, default=-12.0,
                        help="registered psi_bus lower range (advice/001 "
                             "P0-4: a tighter range gives finer resolution "
                             "but risks saturation)")
    parser.add_argument("--psi-hi", type=float, default=2.5)
    parser.add_argument("--price-mode", default="global_simplex",
                        choices=("global_simplex", "owner_local"))
    parser.add_argument("--delta-j", type=float, default=0.05,
                        help="allow J_B0-lite <= J_C + delta_J (advice/019 "
                             "section 7)")
    parser.add_argument("--audit", action="store_true",
                        help="run the action-invariance audit (advice/020 "
                             "section 2-3) on the B0-lite and B0 arms and "
                             "certify the finite-bit action-error bound "
                             "(norm-free is an approximation after "
                             "quantization, not a strict reparameterization)")
    parser.add_argument("--action-error-thresh", type=float, default=None,
                        help="DEPRECATED name for --min-margin-ok: the "
                             "required B0-lite margin_ok_fraction (the "
                             "conservative P(margin>2*eps_psi) "
                             "certificate) when --audit is on.  Default "
                             "None = do not gate on margin; the primary "
                             "finite-bit gate is --max-action-change")
    parser.add_argument("--max-action-change", type=float, default=0.0,
                        help="max allowed B0-lite action_change_rate for "
                             "the finite-bit action-distortion certification "
                             "when --audit is on (default 0.0 = no action is "
                             "ever flipped by the finite-bit broadcast)")
    parser.add_argument("--min-margin-ok", type=float, default=None,
                        help="optional lower bound on the B0-lite "
                             "margin_ok_fraction certificate when --audit "
                             "is on (default None = not gated)")
    parser.add_argument("--max-psi-sat", type=float, default=0.0,
                        help="max allowed B0-lite psi_sat_rate (clipping "
                             "fraction) for the finite-bit certificate "
                             "when --audit is on (advice/001 P0-4; default "
                             "0.0 = no saturation is certified)")
    parser.add_argument("--air-normalize", default="mesh",
                        choices=("mesh", "owner"),
                        help="airtime normalization (advice/020 section 5-7): "
                             "'mesh' derives T_air from the full-mesh "
                             "always-report load ratio rho_target (the "
                             "legacy confound); 'owner' derives T_air from "
                             "the balanced owner-directed offered load so "
                             "that rho_owner is MATCHED at every scale -- "
                             "the capacity-regime-controlled comparison")
    parser.add_argument("--rho-owner", type=float, default=None,
                        help="target owner-directed load ratio when "
                             "--air-normalize owner (defaults to the same "
                             "values as --rhos)")
    parser.add_argument("--nested", action="store_true",
                        help="build (8,4) as a nested subset of a (16,8) "
                             "master scenario (same U2U/sensing "
                             "realizations, advice/020 section 8) instead "
                             "of two independent draws")
    parser.add_argument("--hier-boot", type=int, default=10000,
                        help="bootstrap repeats for the hierarchical "
                             "(cell -> seed -> block) cross-scenario CI "
                             "and cell sign consistency (advice/020 "
                             "section 12)")
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    parser.add_argument("--calib-n-runs", type=int, default=300)
    parser.add_argument("--cache", type=str, default=None,
                        help="path to a raw-cell JSON cache (the per-block "
                             "MC data, BEFORE aggregation).  If the file "
                             "exists and its config hash matches, the "
                             "expensive MC loop is skipped and only the "
                             "aggregation/verdict is recomputed -- makes "
                             "the gate crash-tolerant (e.g. an aggregation "
                             "bug no longer wastes the 15-min MC).  "
                             "Default: <output>.cells.json")
    args = parser.parse_args()
    t0 = time.time()

    scales = []
    for s in args.scales:
        kk, qq = s.split(",")
        scales.append((int(kk), int(qq)))

    # calibration depends only on (geom, scale) -- cache per (geom, scale)
    bound_cache: dict[tuple[int, tuple[int, int]], list] = {}
    air_cache: dict[tuple[int, tuple[int, int], float], dict] = {}
    master_cache: dict[int, dict] = {}

    def _scenario(geom, k, q):
        if args.nested:
            # advice/020 section 8: one (16,8) master per geom, sliced to
            # nested subsets so the two scales share U2U/sensing draws.
            if geom not in master_cache:
                master_cache[geom] = build_distributed_scenario(
                    np.random.default_rng(geom), k_uavs=16, q_targets=8)
            subs = nested_scenario_subsets(master_cache[geom])
            if (k, q) in subs:
                return subs[(k, q)]
        return build_distributed_scenario(np.random.default_rng(geom),
                                          k_uavs=k, q_targets=q)

    def _bounds(geom, k, q):
        key = (geom, (k, q))
        if key not in bound_cache:
            sc = _scenario(geom, k, q)
            bt = calibrate_target_bounds(
                sc, args.alpha, args.beta,
                n_runs=args.calib_n_runs, seed=args.calib_seed,
                llr_bits=TOKEN_LLR_BITS, verify_runs=args.calib_verify)
            bound_cache[key] = [[bt[qq][0], bt[qq][1] - 1.0]
                                for qq in range(q)]
        return bound_cache[key]

    def _airtime(geom, k, q, rho):
        key = (geom, (k, q), rho, args.air_normalize)
        if key not in air_cache:
            if args.air_normalize == "owner":
                air_cache[key] = build_airtime_model(
                    _scenario(geom, k, q),
                    rho_owner=args.rho_owner if args.rho_owner is not None
                    else rho)
            else:
                air_cache[key] = build_airtime_model(
                    _scenario(geom, k, q), rho_target=rho)
        return air_cache[key]

    cells = []
    cache_path = Path(args.cache) if args.cache else \
        Path(str(Path(args.output).with_suffix("")) + ".cells.json")
    cfg_hash = _config_hash(args)
    loaded = False
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if raw.get("config_hash") == cfg_hash \
                    and raw.get("grid") == {
                        "geoms": args.geoms, "rhos": args.rhos,
                        "scales": [list(s) for s in scales],
                        "test_seeds": args.test_seeds}:
                cells = raw["cells"]
                loaded = True
                print(f"loaded {len(cells)} cells from cache "
                      f"{cache_path} (skipping MC)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"cache load failed ({exc}); re-running MC", flush=True)

    if not loaded:
        cells = _run_mc(args, scales, _scenario, _bounds, _airtime, t0)
        # persist the raw per-cell block data BEFORE aggregation so an
        # aggregation/verdict bug never wastes the MC loop again
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "config_hash": cfg_hash,
            "grid": {
                "geoms": args.geoms, "rhos": args.rhos,
                "scales": [list(s) for s in scales],
                "test_seeds": args.test_seeds},
            "cells": cells,
        }, indent=2), encoding="utf-8")
        print(f"saved raw cells -> {cache_path}", flush=True)

    # ---- per-cell cross-seed pooling -------------------------------------
    arms_order = ["B0-lite", "B0-D", "B1", "C", "B0"]
    for cell in cells:
        sres = cell["seed_results"]
        for name in arms_order:
            blocks_n = _pooled_blocks([r["arms"][name] for r in sres],
                                      "block_n")
            blocks_s = _pooled_blocks([r["arms"][name] for r in sres],
                                      "block_s")
            N = np.sum(np.stack(blocks_n, axis=0), axis=0)
            S = np.sum(np.stack(blocks_s, axis=0), axis=0)
            cell["arms"][name] = {
                "J": _pooled_j(N, S),
                "J_risk": float(np.max(
                    np.sum(np.stack(_pooled_blocks(
                        [r["arms"][name] for r in sres], "block_risk"),
                        axis=0), axis=0)
                    / np.maximum(N, 1.0))),
                "control_bits_per_cycle": float(np.mean([
                    r["arms"][name]["ctrl_bits_per_cycle"] for r in sres])),
                "qos": [r["arms"][name]["qos"] for r in sres],
                "budget_feasible": float(np.mean([
                    _budget_feasible(r["arms"][name]) for r in sres])),
                "audit": _pooled_audit(
                    [r["arms"][name].get("audit") for r in sres]),
            }
        # pooled paired deltas across ALL test-seed blocks (the block is
        # the bootstrap unit, advice/019 section 8; blocks of every seed
        # are concatenated so the CI covers seed variation too).
        def _delta(a, b):
            d, lo, hi = _pooled_delta_ci(
                _pooled_blocks([r["arms"][a] for r in sres], "block_n"),
                _pooled_blocks([r["arms"][a] for r in sres], "block_s"),
                _pooled_blocks([r["arms"][b] for r in sres], "block_n"),
                _pooled_blocks([r["arms"][b] for r in sres], "block_s"))
            return {"point": d, "ci95": [lo, hi], "state": _ci_state(lo, hi)}
        cell["deltas"] = {
            "D_C_minus_lite": _delta("C", "B0-lite"),
            "D_B0_minus_lite": _delta("B0", "B0-lite"),
            "D_B1_minus_lite": _delta("B1", "B0-lite"),
            "D_C_minus_B0D": _delta("C", "B0-D"),
        }
        # per-seed consistency of the minimality criterion
        lite_j = [r["arms"]["B0-lite"]["J"] for r in sres]
        c_j = [r["arms"]["C"]["J"] for r in sres]
        cell["per_seed_J"] = {
            "B0-lite": lite_j, "C": c_j,
            "lite_minus_C": [a - b for a, b in zip(lite_j, c_j)],
        }
        cell["minimality_frac"] = float(np.mean([
            1.0 if (a <= b + args.delta_j) else 0.0
            for a, b in zip(lite_j, c_j)]))
        cell["verdict"] = _verdict(cell, args.delta_j,
                                   args.max_action_change,
                                   args.min_margin_ok,
                                   args.max_psi_sat)

    # ---- overall aggregation ----------------------------------------------
    fracs = [c["minimality_frac"] for c in cells]
    lite_j_all = [c["arms"]["B0-lite"]["J"] for c in cells]
    c_j_all = [c["arms"]["C"]["J"] for c in cells]
    # pooled over the whole grid: concat all cells' blocks
    overall = {
        "n_cells": len(cells),
        "n_test_seeds": len(args.test_seeds),
        "minimality_frac_mean": float(np.mean(fracs)) if fracs else 0.0,
        "minimality_frac_min": float(np.min(fracs)) if fracs else 0.0,
        "minimality_frac_cells_100pct": float(np.sum(
            [1.0 for f in fracs if f >= 1.0])),
        "mean_J_lite": float(np.mean(lite_j_all)) if lite_j_all else 0.0,
        "mean_J_C": float(np.mean(c_j_all)) if c_j_all else 0.0,
        "mean_lite_minus_C": float(np.mean(
            [a - b for a, b in zip(lite_j_all, c_j_all)])) if cells else 0.0,
        "verdict_pass_fraction": float(np.mean(
            [1.0 if c["verdict"]["pass"] else 0.0 for c in cells])),
    }
    if cells:
        # HIERARCHICAL cross-scenario statistics (advice/020 section 12):
        # the plain pooled CI is a fixed-grid-mixture estimand, NOT a
        # cross-geometry generalization claim.  Report the hierarchical
        # (cell -> seed -> block) bootstrap CI plus the per-cell sign
        # consistency of D_C_minus_lite as the primary cross-scenario
        # summary.
        cell_sign_deltas = [c["deltas"]["D_C_minus_lite"] for c in cells]
        overall["cell_sign_consistency"] = cell_sign_consistency(
            cell_sign_deltas)
        overall["hierarchical_D_C_minus_lite"] = hierarchical_delta_over_cells(
            cells, n_boot=args.hier_boot, seed=777)
        # Pooled over the whole grid.  Blocks from DIFFERENT scales have
        # different target counts (q differs), so they cannot be stacked
        # together: pool per-scale, then report each scale separately
        # (pooling across geometries/rhos/seeds within one scale is valid
        # -- all share the same q).
        by_scale: dict[tuple, list] = {}
        for c in cells:
            by_scale.setdefault(tuple(c["scale"]), []).append(c)
        pooled = {"per_scale": {}}
        for (k, q), scale_cells in by_scale.items():
            scale_block = {"per_scale_name": f"({k},{q})"}
            for name in ("B0-lite", "B0-D", "B1", "C", "B0"):
                bn = [b for c in scale_cells for b in _pooled_blocks(
                    [r["arms"][name] for r in c["seed_results"]], "block_n")]
                bs = [b for c in scale_cells for b in _pooled_blocks(
                    [r["arms"][name] for r in c["seed_results"]], "block_s")]
                if not bn:
                    continue
                N = np.sum(np.stack(bn, axis=0), axis=0)
                S = np.sum(np.stack(bs, axis=0), axis=0)
                scale_block[f"pooled_J_{name}"] = _pooled_j(N, S)
            lite_all_n = [b for c in scale_cells for b in _pooled_blocks(
                [r["arms"]["B0-lite"] for r in c["seed_results"]], "block_n")]
            lite_all_s = [b for c in scale_cells for b in _pooled_blocks(
                [r["arms"]["B0-lite"] for r in c["seed_results"]], "block_s")]
            c_all_n = [b for c in scale_cells for b in _pooled_blocks(
                [r["arms"]["C"] for r in c["seed_results"]], "block_n")]
            c_all_s = [b for c in scale_cells for b in _pooled_blocks(
                [r["arms"]["C"] for r in c["seed_results"]], "block_s")]
            d, lo, hi = _pooled_delta_ci(c_all_n, c_all_s, lite_all_n,
                                         lite_all_s)
            scale_block["pooled_D_C_minus_lite"] = {
                "point": d, "ci95": [lo, hi], "state": _ci_state(lo, hi)}
            scale_block["n_cells"] = len(scale_cells)
            scale_block["verdict_pass_fraction"] = float(np.mean(
                [1.0 if c["verdict"]["pass"] else 0.0 for c in scale_cells]))
            pooled["per_scale"][f"{k}x{q}"] = scale_block
        overall["pooled"] = pooled

    payload = {
        "gate_id": "p5min-robustness-gate",
        "params": {
            "grid": {
                "geoms": args.geoms, "rhos": args.rhos,
                "scales": [list(s) for s in scales],
                "test_seeds": args.test_seeds,
                "test_episodes_per_seed": args.test_cell_runs * args.test_mc,
                "total_episodes_per_arm": args.test_cell_runs * args.test_mc
                * len(args.test_seeds) * len(args.geoms) * len(args.rhos)
                * len(scales),
            },
            "arms": {
                "B0-lite": "task price + norm_free + neutral (advice/019 "
                           "section 5)",
                "B0-D": "task price + norm_free + density",
                "B1": "task price + lambda + neutral",
                "C": "full CA-FRIDS (task + lambda + density)",
                "B0": "normalized control (B0-lite must be ~equal)",
            },
            "delta_j": args.delta_j,
            "seed_scheme": (
                f"held-out base {TEST_BASE}+s, CRN blocks base+mc; disjoint "
                "from P4.2 cert 100000 / P4.1b discovery 7 / calibration 100 "
                "/ P4.2b cal 200000 / test 300000 / P5-A ladder 400000"),
            "block_bootstrap": (
                "per-block bootstrap with blocks concatenated ACROSS test "
                "seeds, so the CI covers seed variation (advice/019 s8)"),
            "git_commit": _git_sha(), "git_dirty": _git_dirty(),
            "config_hash": _config_hash(args),
        },
        "metrics": {
            "cells": cells,
            "overall": overall,
            "interpretation": (
                "minimality_frac = fraction of the N test seeds per cell on "
                "which J_B0-lite <= J_C + delta_J.  verdict.pass requires: "
                "(i) J_B0-lite <= J_C + delta_J pooled AND on every test "
                "seed, (ii) B0-lite vs B0 not a certified loss (norm-free is "
                "a reparameterization), (iii) control_bits(B0-lite) < "
                "control_bits(C), (iv) budget feasibility preserved.  "
                "D_C_minus_lite > 0 (CERTIFIED_GAIN) means B0-lite is "
                "CERTIFIED faster than C on worst-target delay -- the "
                "Minimality verdict that freezes B0-lite (advice/019 s7).  "
                "QoS at the frozen point is EXPECTED UNCERTAIN; matched-QoS "
                "certification is P4.2b, not this gate."),
        },
        "runtime_s": round(time.time() - t0, 1),
        "provenance": {
            "git_commit": _git_sha(), "git_dirty": _git_dirty(),
            "config_hash": _config_hash(args),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("overall:", json.dumps(overall, indent=1))
    print("done", round(time.time() - t0, 1), "s")


def _budget_feasible(arm_out) -> float:
    return float(arm_out.get("budget_feasible", 1.0))


def hierarchical_delta_over_cells(cells, n_boot=10000, seed=777):
    """Build the hierarchical (cell -> seed -> block) bootstrap input for
    ``D_C_minus_lite = J_C - J_B0-lite`` across the whole grid (advice/020
    section 12).  Returns ``{point, ci95, state}`` where the CI covers
    scenario (cell) variation, not just pooled-mixture block noise."""
    hier_cells = []
    for c in cells:
        seed_blocks = []
        for r in c["seed_results"]:
            lite_n = _pooled_blocks([r["arms"]["B0-lite"]], "block_n")
            lite_s = _pooled_blocks([r["arms"]["B0-lite"]], "block_s")
            c_n = _pooled_blocks([r["arms"]["C"]], "block_n")
            c_s = _pooled_blocks([r["arms"]["C"]], "block_s")
            # hierarchical_block_bootstrap computes J_prev - J_cur with the
            # FIRST pair as prev and the THIRD as cur.  We want
            # D_C_minus_lite = J_C - J_B0-lite, so prev = C, cur = B0-lite.
            seed_blocks.append((c_n, c_s, lite_n, lite_s))
        hier_cells.append({"seed_blocks": seed_blocks})
    d, lo, hi = hierarchical_block_bootstrap(hier_cells, n_boot, seed)
    return {"point": d, "ci95": [lo, hi], "state": _ci_state(lo, hi)}



def _pooled_audit(audits) -> dict | None:
    """Pool the per-block action-invariance audit (advice/020 section 2-3)
    across the blocks of all test seeds.  Each block audit carries the
    aggregate ``margin_ok_fraction`` / ``action_change_rate`` and
    ``margin_samples``; pooling weights by sample count so the pooled
    fraction is the total-ok / total-samples over all blocks.  Returns
    None when no audit was collected (``--audit`` off or no samples)."""
    valid = [a for a in audits if a and a.get("margin_samples", 0) > 0]
    if not valid:
        return None
    tot = float(sum(a["margin_samples"] for a in valid))
    ok = float(sum(a["margin_ok_fraction"] * a["margin_samples"]
                   for a in valid))
    chg = float(sum(a["action_change_rate"] * a["margin_samples"]
                    for a in valid))
    return {
        "margin_ok_fraction": ok / max(tot, 1.0),
        "margin_samples": tot,
        "action_change_rate": chg / max(tot, 1.0),
        "eps_pi": valid[0].get("eps_pi", 0.0),
        "eps_psi": valid[0].get("eps_psi", 0.0),
        "psi_sat_rate": float(sum(
            a.get("psi_sat_rate", 0.0) * a["margin_samples"] for a in valid)
            / max(tot, 1.0)),
        "n_cycles": float(sum(a["n_cycles"] for a in valid)),
    }


def _verdict(cell, delta_j, max_action_change=0.0,
             min_margin_ok=None, max_psi_sat=0.0) -> dict:
    """Advice/019 section 7 final criteria, evaluated on the cross-seed
    pooled deltas plus the per-seed consistency fraction.  When the
    action-invariance audit was collected (``--audit``), criterion (v)
    certifies the finite-bit action distortion of B0-lite: the deployed
    norm-free form is an APPROXIMATION whose action distortion must be
    certified (advice/020 section 2-3), NOT a strict reparameterization.

    The PRIMARY freeze gate is the EMPIRICAL action preservation
    ``action_change_rate <= max_action_change`` (default 0.0 = no action is
    ever flipped by the finite-bit broadcast).  ``margin_ok_fraction`` (the
    conservative ``P(margin > 2*eps_psi)`` certificate) is reported and,
    when ``min_margin_ok`` is given, also gated; ``psi_sat_rate`` (clipping)
    is gated by ``max_psi_sat`` because the certificate is only valid while
    ``psi`` stays in range (advice/001 P0-4)."""
    reasons = []
    ok = True
    lite = cell["arms"]["B0-lite"]
    c = cell["arms"]["C"]
    b0 = cell["arms"]["B0"]
    # (i) delay minimality: pooled and per-seed
    d = cell["deltas"]["D_C_minus_lite"]
    if d["point"] >= -delta_j:
        pass
    else:
        ok = False
        reasons.append(f"J_B0-lite exceeds J_C by {abs(d['point']):.3f} "
                       f"(> delta_J {delta_j})")
    if cell["minimality_frac"] < 1.0:
        ok = False
        reasons.append(f"only {cell['minimality_frac']:.2f} of test seeds "
                       "satisfy J_B0-lite <= J_C + delta_J")
    # (ii) norm-free vs normalized B0 delay: the finite-bit forms can
    # legitimately differ at quantization-bin boundaries / near-tied
    # actions (advice/020 section 2).  A certified LOSS is still a FAIL --
    # the deployed equivalence is an approximation, not strict.
    d0 = cell["deltas"]["D_B0_minus_lite"]
    if d0["state"] == "CERTIFIED_LOSS":
        ok = False
        reasons.append("B0-lite is CERTIFIED slower than normalized B0 "
                       "(the finite-bit norm-free form is NOT an "
                       "equivalent reparameterization here; advice/020 "
                       "section 2)")
    # (iii) control bits must drop
    if lite["control_bits_per_cycle"] >= c["control_bits_per_cycle"]:
        ok = False
        reasons.append("B0-lite control bits not below C")
    # (iv) airtime feasibility preserved
    if lite["budget_feasible"] < 1.0 - 1e-6 or c["budget_feasible"] \
            < 1.0 - 1e-6:
        ok = False
        reasons.append("budget feasibility not preserved (hard airtime "
                       "constraint broken)")
    # (v) finite-bit action distortion certification (only when audited)
    lite_audit = lite.get("audit")
    if lite_audit is not None:
        acr = lite_audit["action_change_rate"]
        if acr > max_action_change:
            ok = False
            reasons.append(
                f"B0-lite finite-bit action distortion not certified "
                f"(action_change_rate {acr:.4f} > {max_action_change})")
        if min_margin_ok is not None and \
                lite_audit["margin_ok_fraction"] < min_margin_ok:
            ok = False
            reasons.append(
                f"B0-lite margin certificate "
                f"{lite_audit['margin_ok_fraction']:.3f} < "
                f"{min_margin_ok}")
        # (vi) psi saturation Gate (advice/001 P0-4): the finite-bit
        # certificate m_i > 2*eps_psi is only valid while psi_q stays IN
        # [psi_lo, psi_hi]; a non-negligible clipping rate means the
        # registered-range certificate does NOT cover the actual distortion,
        # so the cell must be rejected (or the range re-calibrated).
        sat = lite_audit.get("psi_sat_rate", 0.0)
        if sat > max_psi_sat:
            ok = False
            reasons.append(
                f"B0-lite psi saturation not certified "
                f"(psi_sat_rate {sat:.4f} > {max_psi_sat})")
    return {
        "pass": ok,
        "reason": "; ".join(reasons) if reasons else
        "B0-lite minimality holds (delay, norm-free equivalence, control "
        "bits, airtime feasibility, finite-bit action-error bound)",
    }


if __name__ == "__main__":
    main()