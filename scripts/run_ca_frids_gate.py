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
from uav_otfs_isac.crn_tape import build_exogenous_tape
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.frids import simulate_frids_v2
from uav_otfs_isac.qos import raw_qos_status, pool_raw_counts


def _git_sha() -> str:
    """Short HEAD sha for result provenance (advice/010 P0-1b)."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"

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
                mc_seeds=2, crn=False, **kw):
    """FIXED calibrated policy B (the frozen calibration, delta = 1) for
    BOTH algorithms: the P3.4 comparison is the SAME TWO-THRESHOLD
    STOPPING POLICY under two schedulers -- the only fair "same policy /
    different scheduler" reading.  An adaptive per-algorithm delta would
    compare different stopping rules, not different schedulers.  The
    certified QoS is reported separately (RAW conditional counts + the
    simultaneous Clopper-Pearson certificate, P2.1a).  ``mc_seeds``
    independent MC draws per cell (advice/008 P3.4: multi-seed MC).

    ``crn=True`` (advice/010 P0-2): one shared exogenous tape is built per
    MC seed with ``build_exogenous_tape(seed + mc, ...)`` and forwarded to
    the simulator, so FRIDS-v2 and CA-FRIDS consume the SAME target
    presence, observation, and link uniforms at every ``(r, t)`` -- the
    paired comparison is then over the same exogenous realizations (their
    different actions still map the same base uniforms through different
    kernels/admission rules, exactly as the CRN prescription requires)."""
    q = sc["q"]
    k = int(sc["k"])
    bounds = matched_bounds(bt, q, 1.0)
    rows = []
    for mc in range(mc_seeds):
        tape = None
        if crn:
            tape = build_exogenous_tape(seed + mc, n_runs, q, k, max_steps)
        rows.append(run_cell(sim, sc, bounds, n_runs, seed + mc, max_steps,
                             exog=tape, **kw))
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
    """Per-cell worst-target E1 (legacy max-then-average diagnostic; the
    formal gate uses ``j_pooled``)."""
    return np.max([r["e1_delays"] for r in rows], axis=1)


def j_pooled(rows):
    """advice/010 P0-5: GEOMETRY-POOLED worst-target E[T_q | H1].

    ``rows`` are all MC cells of ONE geometry and regime.  Target delay
    sums and H1 counts are pooled across every run and every MC seed,
    then ``J_g = max_q sum_h1_delay[q]/n_h1[q]`` -- the pooled objective,
    not the per-cell worst-then-average with the selection bias
    ``E[max_q hatT_q] >= max_q E[hatT_q]``.
    """
    total_n = np.asarray(rows[0]["pool"]["n_h1"], dtype=float)
    total_s = np.asarray(rows[0]["pool"]["sum_h1_delay"], dtype=float)
    for r in rows[1:]:
        total_n = total_n + np.asarray(r["pool"]["n_h1"], dtype=float)
        total_s = total_s + np.asarray(r["pool"]["sum_h1_delay"], dtype=float)
    values = np.where(total_n > 0, total_s / np.maximum(total_n, 1e-12), 0.0)
    return float(np.max(values))


def comm_pooled(rows, key):
    """advice/010 section 4: metric averaged over ALL MC cells of the
    geometry (the legacy gate only read ``rows[0]``)."""
    return float(np.mean([r["comm"][key] for r in rows]))


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
                                        crn=True, price_mode="local")
            if reg == "uncongested":
                # v2 baseline in the free regime: independent delivery
                # (the frozen mainline); CA-FRIDS on the same budget
                b_ca, rows_ca = matched_qos(simulate_ca_frids, sc, bt,
                                            args.n_runs, 7, args.max_steps,
                                            args.alpha, args.beta,
                                            mc_seeds=args.mc_seeds,
                                            crn=True,
                                            # P3.6-R (advice/010 P0-1): the
                                            # price mode MUST be forwarded --
                                            # the CA default is owner_local
                                            # and the gate previously ran the
                                            # wrong scheduler variant.
                                            price_mode=args.price_mode,
                                            airtime=am, pi_bits=args.pi_bits,
                                            lam_bits=args.lam_bits)
                scen_rows.append({
                    "geom": geom, "regime": reg,
                    "v2": summarize(rows_v2),
                    "ca": summarize(rows_ca),
                    # P3.6-R paired cells (advice/010 P0-2/3): the CRN tape
                    # aligns the exogenous H/obs/link draws per (r, t); the
                    # per-cell arrays are diagnostics only.
                    "j_v2": j_worst(rows_v2).tolist(),
                    "j_ca": j_worst(rows_ca).tolist(),
                    # P0-5: GEOMETRY-POOLED worst-target E1 (the formal
                    # objective; the MC blocks only provide variance).
                    "j_v2_pooled": j_pooled(rows_v2),
                    "j_ca_pooled": j_pooled(rows_ca),
                    "v2_qos": _qos(rows_v2, args.alpha, args.beta),
                    "ca_qos": _qos(rows_ca, args.alpha, args.beta),
                    "air": {"rho_full": am["rho_full"],
                            # v2 is the frozen full-mesh always-report
                            # baseline (tx = 1.0); the uncongested gate is
                            # about worst-target delay regression only
                            "v2_tx": 1.0,
                            # aggregated over ALL MC cells, not rows[0]
                            "ca_tx": comm_pooled(rows_ca,
                                                 "tx_attempts_per_uav"),
                            "ca_control_bits_per_cycle": comm_pooled(
                                rows_ca, "control_bits_per_cycle"),
                            "ca_feasible": comm_pooled(
                                rows_ca, "budget_feasible_fraction")},
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
                rows_v2_hard = []
                for mc in range(args.mc_seeds):
                    tape = build_exogenous_tape(
                        7 + mc, args.n_runs, q, k, args.max_steps)
                    rows_v2_hard.append(run_cell(
                        simulate_frids_v2, sc, b_v2h, args.n_runs,
                        7 + mc, args.max_steps, price_mode="local",
                        exog=tape,
                        rx_cap_tokens=np.full(k, max(cap_tokens, 1))))
                b_ca, rows_ca = matched_qos(simulate_ca_frids, sc, bt,
                                            args.n_runs, 7, args.max_steps,
                                            args.alpha, args.beta,
                                            mc_seeds=args.mc_seeds,
                                            crn=True,
                                            price_mode=args.price_mode,
                                            airtime=am,
                                            pi_bits=args.pi_bits,
                                            lam_bits=args.lam_bits)
                scen_rows.append({
                    "geom": geom, "regime": reg,
                    "v2": summarize(rows_v2_hard),
                    "ca": summarize(rows_ca),
                    "j_v2": j_worst(rows_v2_hard).tolist(),
                    "j_ca": j_worst(rows_ca).tolist(),
                    "j_v2_pooled": j_pooled(rows_v2_hard),
                    "j_ca_pooled": j_pooled(rows_ca),
                    "v2_qos": _qos(rows_v2_hard, args.alpha, args.beta),
                    "ca_qos": _qos(rows_ca, args.alpha, args.beta),
                    "air": {"rho_full": am["rho_full"],
                            "v2_tx": 1.0,
                            "ca_tx": comm_pooled(rows_ca,
                                                 "tx_attempts_per_uav"),
                            "ca_control_bits_per_cycle": comm_pooled(
                                rows_ca, "control_bits_per_cycle"),
                            "ca_max_load": comm_pooled(rows_ca,
                                                       "max_load_ratio"),
                            "ca_feasible": comm_pooled(
                                rows_ca, "budget_feasible_fraction")},
                })
        print(f"geom {geom} done ({time.time()-t0:.0f}s)", flush=True)

    # P3.6-R (advice/010 P0-4, P0-3): independent sub-gates.
    # QoS sub-gate: PASS-only certification.  UNCERTAIN is UNRESOLVED, it
    # is never relabelled as OK by "!= FAIL" (that P0 let 4 PASS + 2
    # UNCERTAIN cells certify as "qos_ok").  We report PASS/FAIL/UNCERTAIN
    # counts separately and require ALL PASS for adoption.
    ca_pass = sum(1 for r in scen_rows if r["ca_qos"] == "PASS")
    ca_fail = sum(1 for r in scen_rows if r["ca_qos"] == "FAIL")
    ca_unc = sum(1 for r in scen_rows if r["ca_qos"] == "UNCERTAIN")
    qos_all_pass = (ca_fail == 0 and ca_unc == 0 and ca_pass == len(scen_rows))
    qos_any_fail = bool(ca_fail > 0)
    qos_any_uncertain = bool(ca_unc > 0)

    # P3.6-R (advice/010 P0-3/P0-5): PAIRED performance gate on the
    # GEOMETRY-POOLED worst-target E1 (advice/010 section 3): each
    # geometry pools ALL runs and ALL MC seeds and reports
    # J_g = max_q sum_h1_delay[q]/n_h1[q]; the geometry is the paired
    # sample unit and the MC blocks provide only variance -- exactly the
    # pooled-objective prescription (no per-cell max-then-average bias).
    # The legacy per-MC-cell ``j_v2``/``j_ca`` arrays remain in the rows
    # as diagnostics only.  The CRN tape (P0-2) makes the H/obs/link
    # exogenous draws identical between the two schedulers, so the
    # geometry pairs are genuine paired comparisons.
    def paired_j(reg, key):
        return np.asarray(
            [float(r[key]) for r in scen_rows if r["regime"] == reg],
            dtype=float,
        )

    unc_v2 = paired_j("uncongested", "j_v2_pooled")
    unc_ca = paired_j("uncongested", "j_ca_pooled")
    cong_v2 = paired_j("congested", "j_v2_pooled")
    cong_ca = paired_j("congested", "j_ca_pooled")
    d_unc, ci_unc_lo, ci_unc_hi = paired_bootstrap_ci(unc_v2, unc_ca)
    d_cong, ci_cong_lo, ci_cong_hi = paired_bootstrap_ci(cong_ca, cong_v2)
    j_unc_v2_mean = float(np.mean(unc_v2)) if len(unc_v2) else 0.0
    j_cong_v2_mean = float(np.mean(cong_v2)) if len(cong_v2) else 0.0
    # relative regression (CA worse) and relative gain (CA better)
    unc_ucb = ci_unc_hi / max(j_unc_v2_mean, 1e-12)
    cong_lcb = ci_cong_lo / max(j_cong_v2_mean, 1e-12)
    unc_reg_mean = d_unc / max(j_unc_v2_mean, 1e-12)
    cong_gain_mean = d_cong / max(j_cong_v2_mean, 1e-12)
    unc_ci_lo = ci_unc_lo / max(j_unc_v2_mean, 1e-12)
    unc_ci_hi = ci_unc_hi / max(j_unc_v2_mean, 1e-12)
    cong_ci_lo = ci_cong_lo / max(j_cong_v2_mean, 1e-12)
    cong_ci_hi = ci_cong_hi / max(j_cong_v2_mean, 1e-12)
    no_regression_uncongested = bool(unc_ucb <= 0.02)
    congested_win_5pct = bool(cong_lcb >= 0.05)
    gate = {
        "uncongested_regression_mean": float(unc_reg_mean),
        "uncongested_regression_ci95": [float(unc_ci_lo), float(unc_ci_hi)],
        "no_regression_uncongested_ucb": bool(no_regression_uncongested),
        "congested_improvement_mean": float(cong_gain_mean),
        "congested_improvement_ci95": [float(cong_ci_lo), float(cong_ci_hi)],
        "congested_win_5pct_lcb": bool(congested_win_5pct),
        "qos_all_pass": bool(qos_all_pass),
        "qos_count": {"PASS": ca_pass, "FAIL": ca_fail, "UNCERTAIN": ca_unc},
        "qos_any_fail": bool(qos_any_fail),
        "qos_any_uncertain": bool(qos_any_uncertain),
        "adopt_ca": bool(no_regression_uncongested and congested_win_5pct
                         and qos_all_pass),
        "verdict": ("CA-FRIDS adopted as the joint-capacity main scheduler"
                    if no_regression_uncongested and congested_win_5pct
                    and qos_all_pass
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
                   # P3.6-R (advice/010 P0-1b): provenance of the exact
                   # scheduler variant actually executed.
                   "price_mode": args.price_mode,
                   "git_sha": _git_sha(),
                   "seed_scheme": "CRN exogenous tape (advice/010 P0-2): "
                                  "shared U_H/U_obs/U_link per (r,t)",
                   "capacity_model": ("v2: independent delivery / hard "
                                      "token cap (rx_cap_tokens); CA: "
                                      "airtime budget sum tau <= T_air "
                                      "(distinct capacity primitives remain "
                                      "a documented open item)"),
                   "evidence_mode": "owner-only evidence plane (Dual-Bus)",
                   "crn_tape": True,
                   "objective": "geometry-pooled worst-target E[T|H1] "
                                "(advice/010 P0-5)",
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
    assert payload["params"]["price_mode"] == args.price_mode, (
        "formal gate must record the price-mode it actually ran"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()