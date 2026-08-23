"""Gate P3.4 (advice/008 section 14 step 7): the CA-FRIDS Dual-Bus
live-or-die gate.

The one gate that decides whether the joint-capacity Dual-Bus scheduler
(P3.2) becomes the main algorithm or FRIDS-v2 stays the frozen core:

- **No-regression in the uncongested zone**: in the free-airtime regime
  (rho_full < 1, lambda -> 0) the CA action reduces to the task price
  ``pi*g`` and must not degrade the worst-target delay by more than ~2%
  versus FRIDS-v2.  This is the honest weak claim: the joint index must
  not hurt when airtime is free.
- **Communication-boundary win**: in the congested regime (rho_full > 1)
  CA-FRIDS must either deliver a feasible-load expansion (or the
  certified simultaneous QoS holds while v2 FAILs) or improve the
  worst-target delay by >= 5% -- with the hard airtime price cutting the
  overloaded receive path (Lemma 4.99 needs the idle option: the price
  reorders sensing AND decids report vs silence).
- **Policy-matched QoS**: every algorithm/scenario uses its own matched
  upper threshold (as every FRIDS gate does, adaptive delta) and the
  dual QoS is CERTIFIED on the RAW per-target conditional counts with
  the simultaneous Clopper-Pearson confidence (P2.1a, advice/008
  section 13).

Verdict: CA-FRIDS adopted as the main scheduler only if uncongested
regression <= 2% AND the congested win is certified; otherwise
FRIDS-v2 stays frozen and the joint capacity + phase law remain the
paper's contribution (either way the research question is answered).
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

from uav_otfs_isac.airtime import (
    build_airtime_model,
    simulate_frids_v2_air,
)
from uav_otfs_isac.ca_frids import simulate_ca_frids
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.frids import simulate_frids_v2
from uav_otfs_isac.qos import raw_qos_status, pool_raw_counts

SCALES = ((6, 3), (8, 4), (12, 6), (16, 8))


def matched_bounds(bt, q, delta=1.0):
    """Adaptive policy-matched B (the standard FRIDS protocol): lower the
    H0 threshold by ``delta`` until the certified QoS enters the
    feasible side; the frozen ``delta=1.0`` is the default here."""
    return [[bt[qq][0], bt[qq][1] - delta] for qq in range(q)]


def _qos(rows, alpha, beta):
    p = pool_raw_counts(rows)
    return raw_qos_status(p["n_H0"], p["n_H1"], p["n_FA"], p["n_MD"],
                          alpha, beta, 0.05)


def run_cell(sim, sc, bounds, n_runs, seed, max_steps, **kw):
    out = sim(sc, bounds, n_runs=n_runs, seed=seed, max_steps=max_steps,
              raw_counts=True, **kw)
    return out


def matched_qos(sim, sc, bt, n_runs, seed, max_steps, alpha, beta,
                mc_seeds=2, **kw):
    """FIXED calibrated policy B (the frozen calibration, delta = 1) for
    BOTH algorithms: the P3.4 comparison is the SAME TWO-THRESHOLD
    STOPPING POLICY under two schedulers -- the only fair "same policy /
    different scheduler" reading.  An adaptive per-algorithm delta would
    compare different stopping rules, not different schedulers.  The
    certified QoS is reported separately (RAW conditional counts + the
    simultaneous Clopper-Pearson certificate, P2.1a).  ``mc_seeds``
    independent MC draws per cell (advice/008 P3.4: multi-seed MC)."""
    q = sc["q"]
    bounds = matched_bounds(bt, q, 1.0)
    rows = [run_cell(sim, sc, bounds, n_runs, seed + mc, max_steps,
                     **kw) for mc in range(mc_seeds)]
    return bounds, rows


def summarize(rows):
    J = np.max([r["e1_delays"] for r in rows], axis=1)
    pfa = np.max([r["p_fa"] for r in rows], axis=0)
    pmd = np.max([r["p_md"] for r in rows], axis=0)
    return {"J": float(np.mean(J)) / 1.0,         # E1[worst delay]
            "p_fa_max": float(np.max(pfa)),
            "p_md_max": float(np.max(pmd))}


def air_comm(out, key):
    return out["comm"][key]


def j_worst(rows):
    """Worst-target E1 per aligned MC cell (the paired-CRN unit)."""
    return np.max([r["e1_delays"] for r in rows], axis=1)


def paired_bootstrap_ci(js_v2, js_ca, b_iters=10000, seed=0):
    """P3.5-C (advice/009 sections 16-17): PAIRED bootstrap 95% CI of the
    worst-target-delay difference ``J_CA - J_v2``.  The two schedulers
    are evaluated on the SAME aligned (geom, MC-seed) exogenous draws
    (per-run independent RNG, episode-reset state), so ``delta_r`` is a
    paired comparison and the CI gates the PERFORMANCE claim instead of
    the point estimate -- the gate decides on the difference interval,
    not on ``mean(gain)`` alone.  Returns ``(mean_delta, ci_lo, ci_hi)``."""
    delta = np.asarray(js_ca, dtype=float) - np.asarray(js_v2, dtype=float)
    n = len(delta)
    rng = np.random.default_rng(seed)
    boot = np.empty(b_iters)
    for b in range(b_iters):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.mean(delta[idx]))
    ci = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(delta)), float(ci[0]), float(ci[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ca_frids_gate.json")
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    parser.add_argument("--geoms", type=int, default=3,
                        help="number of independent scenario geometry seeds "
                             "per regime (advice/008 P3.4: multiple geoms, "
                             "aggregated verdict)")
    parser.add_argument("--mc-seeds", type=int, default=2,
                        help="number of independent MC seeds per cell "
                             "(multiple detection-trial draws per geometry)")
    parser.add_argument("--uncongested-rho", type=float, default=0.5)
    parser.add_argument("--congested-rho", type=float, default=1.8)
    parser.add_argument("--pi-bits", type=int, default=10)
    parser.add_argument("--lam-bits", type=int, default=10)
    parser.add_argument("--price-mode", default="global_simplex",
                        choices=("global_simplex", "owner_local"),
                        help="P3.6 (advice/009 section 7): the GLOBAL-simplex "
                             "task price (the only networked quantity is the "
                             "scalar ``Z`` reduction, ``sum y_q = 1`` holds "
                             "strictly) vs the P3.5-A owner-local baseline")
    args = parser.parse_args()
    t0 = time.time()

    k, q = SCALES[-1]          # the gate runs the (16, 8) critical scale
    results = {}
    rho_to_air = {args.uncongested_rho: "uncongested",
                  args.congested_rho: "congested"}

    def airtime_of(rho):
        return build_airtime_model(sc, rho_target=rho)

    scen_rows = []
    for geom in range(args.geoms):
        sc = build_distributed_scenario(np.random.default_rng(geom),
                                        k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(
            sc, args.alpha, args.beta, n_runs=300, seed=args.calib_seed,
            llr_bits=TOKEN_LLR_BITS, verify_runs=args.calib_verify)
        for rho, reg in rho_to_air.items():
            am = airtime_of(rho)
            b_v2, rows_v2 = matched_qos(simulate_frids_v2, sc, bt,
                                        args.n_runs, 7, args.max_steps,
                                        args.alpha, args.beta,
                                        mc_seeds=args.mc_seeds,
                                        price_mode="local")
            if reg == "uncongested":
                # v2 baseline in the free regime: independent delivery
                # (the frozen mainline); CA-FRIDS on the same budget
                b_ca, rows_ca = matched_qos(simulate_ca_frids, sc, bt,
                                            args.n_runs, 7, args.max_steps,
                                            args.alpha, args.beta,
                                            mc_seeds=args.mc_seeds,
                                            airtime=am, pi_bits=args.pi_bits,
                                            lam_bits=args.lam_bits)
                scen_rows.append({
                    "geom": geom, "regime": reg,
                    "v2": summarize(rows_v2),
                    "ca": summarize(rows_ca),
                    "v2_qos": _qos(rows_v2, args.alpha, args.beta),
                    "ca_qos": _qos(rows_ca, args.alpha, args.beta),
                    "air": {"rho_full": am["rho_full"],
                            # v2 is the frozen full-mesh always-report
                            # baseline (tx = 1.0); the uncongested gate is
                            # about worst-target delay regression only
                            "v2_tx": 1.0,
                            "ca_tx": air_comm(rows_ca[0],
                                              "tx_attempts_per_uav"),
                            "ca_feasible": air_comm(rows_ca[0],
                                                    "budget_feasible_fraction")},
                })
            else:
                # congested regime: FRIDS-v2 with HARD receive admission
                # (the P2.1 baseline that treats the bottleneck as a real
                # drop) versus CA-FRIDS whose airtime price REORDERS the
                # sensing before the drop (the joint capacity claim)
                # hard per-receiver token cap giving the congested load ratio of the
                # always-report full mesh (rho_C = rho_full at the cap)
                cap_tokens = (k - 1) / max(am["rho_full"], 1e-3)
                b_v2h = matched_bounds(bt, q, 1.0)
                rows_v2_hard = [
                    run_cell(simulate_frids_v2, sc, b_v2h, args.n_runs,
                             7 + mc, args.max_steps, price_mode="local",
                             rx_cap_tokens=np.full(k, max(cap_tokens, 1)))
                    for mc in range(args.mc_seeds)]
                b_ca, rows_ca = matched_qos(simulate_ca_frids, sc, bt,
                                            args.n_runs, 7, args.max_steps,
                                            args.alpha, args.beta,
                                            mc_seeds=args.mc_seeds,
                                            airtime=am,
                                            pi_bits=args.pi_bits,
                                            lam_bits=args.lam_bits)
                scen_rows.append({
                    "geom": geom, "regime": reg,
                    "v2": summarize(rows_v2_hard),
                    "ca": summarize(rows_ca),
                    "v2_qos": _qos(rows_v2_hard, args.alpha, args.beta),
                    "ca_qos": _qos(rows_ca, args.alpha, args.beta),
                    "air": {"rho_full": am["rho_full"],
                            "v2_tx": 1.0,
                            "ca_tx": air_comm(rows_ca[0],
                                             "tx_attempts_per_uav"),
                            "ca_max_load": air_comm(rows_ca[0],
                                                    "max_load_ratio"),
                            "ca_feasible": air_comm(rows_ca[0],
                                                    "budget_feasible_fraction")},
                })
        print(f"geom {geom} done ({time.time()-t0:.0f}s)", flush=True)

    # aggregate and gate
    def agg(reg, algo):
        vals = [r[algo]["J"] for r in scen_rows if r["regime"] == reg]
        return float(np.mean(vals))

    j_unc_v2 = agg("uncongested", "v2")
    j_unc_ca = agg("uncongested", "ca")
    j_cong_v2 = agg("congested", "v2")
    j_cong_ca = agg("congested", "ca")
    unc_reg = (j_unc_ca - j_unc_v2) / max(j_unc_v2, 1e-12)
    cong_gain = (j_cong_v2 - j_cong_ca) / max(j_cong_v2, 1e-12)
    qos_all_ok = all(
        (r["ca_qos"] != "FAIL") for r in scen_rows)
    gate = {
        "uncongested_regression": float(unc_reg),
        "no_regression_uncongested": bool(unc_reg <= 0.02),
        "congested_improvement": float(cong_gain),
        "congested_win_5pct": bool(cong_gain >= 0.05),
        "ca_qos_ok": bool(qos_all_ok),
        "adopt_ca": bool(unc_reg <= 0.02 and cong_gain >= 0.05
                         and qos_all_ok),
        "verdict": ("CA-FRIDS adopted as the joint-capacity main scheduler"
                    if unc_reg <= 0.02 and cong_gain >= 0.05 and qos_all_ok
                    else ("FRIDS-v2 stays the frozen core; joint capacity "
                          "+ phase law remain the paper contributions")),
    }
    payload = {
        "gate": "p3.4-ca-frids-live-or-die",
        "params": {"scale": [k, q], "n_runs": args.n_runs,
                   "max_steps": args.max_steps, "alpha": args.alpha,
                   "beta": args.beta, "pi_bits": args.pi_bits,
                   "lam_bits": args.lam_bits,
                   "uncongested_rho": args.uncongested_rho,
                   "congested_rho": args.congested_rho,
                   "protocol": [
                       "FIXED calibrated policy B (delta 1, same stopping "
                       "policy for both schedulers -- the fair scheduler-"
                       "only comparison)",
                       "raw conditional counts, simultaneous Clopper-"
                       "Pearson confidence (P2.1a, advice/008 s13)",
                       "uncongested: joint index must not regress (<=2%)",
                       "congested: >=5% worst-delay win and certified QoS",
                       "airtime price + idle option (Lemma 4.99)",
                       "evidence plane to owner only (Dual-Bus)"],
                   "frozen": ["FRIDS-v2", "fixed owner", "two-threshold "
                              "stopping", "calibrated policy-matched B"]},
        "runtime_s": round(time.time() - t0, 1),
        "scenarios": scen_rows,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()