"""Gate G10: Resource-Conserving Sensing Audit (advice/024).

FRIDS-v2 is FROZEN.  Two questions about the System-Bottleneck sensing
headroom (18.2% from a uniform +4 dB boost):

G10-A (replication): replicate the +4 dB sensing headroom over
N scenario draws at K=16/Q=8.  Life gate: the 95% CI lower bound of the
paired gap must exceed 5% for sensing to be confirmed as the main
bottleneck.

G10-B (energy-conserving oracle): does the +4 dB headroom come from MORE
total sensing energy (hardware headroom, low algorithmic value) or from
REALLOCATING the existing energy (allocatable headroom)?  The oracle
gives the weak (worst) target a high sensing-power cap (~+4 dB) and the
easy targets a low cap, keeping the average per-UAV cap (and the measured
per-UAV sensing energy) equal to the current system.  The gap

    Delta_E = (J_current - J_same_energy_oracle) / J_current

classifies (advice/024 section 5): Case A (< 3%: allocation already good,
+4 dB is hardware -> go to fixed-TB OTFS, no power algorithm), Case B
(5-10%: light sensing-energy extension), Case C (> 10%: sensing resource
allocation is the real bottleneck -> energy-priced FRIDS, G10-D).

Writes ``results/sensing_resource_audit.json``.
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
)
from uav_otfs_isac.frids import g_reliable, simulate_frids_v2

SNR_SHIFT = 4.0
POWERS = (1.0, 2.0, 3.0, 4.0, 5.0)


def eval_sim(sc, bounds, n_runs, seeds, max_steps, **kw):
    J = []
    power = None
    for seed in range(seeds):
        out = simulate_frids_v2(sc, bounds, n_runs=n_runs,
                                seed=seed * 1000 + 7, max_steps=max_steps,
                                **kw)
        J.append(out["worst_target_delay"])
        if "sensing_power_per_uav" in out:
            power = out["sensing_power_per_uav"]
    row = {"J": float(np.mean(J))}
    if power is not None:
        row["sensing_power_per_uav"] = float(power)
    return row


def bootstrap_ci(deltas, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    draws = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/sensing_resource_audit.json")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=300)
    parser.add_argument("--b-delta", type=float, default=1.0)
    parser.add_argument("--n-scenarios", type=int, default=12,
                        help="scenario draws for the G10-A replication")
    parser.add_argument("--n-oracle-scenarios", type=int, default=5,
                        help="scenario draws for the G10-B energy oracle")
    parser.add_argument("--n-otfs-scenarios", type=int, default=5,
                        help="scenario draws for the G10-C fixed-TB OTFS audit")
    parser.add_argument("--otfs-grids", action="store_true",
                        default=True, help="run the G10-C fixed-TB OTFS audit")
    args = parser.parse_args()

    t0 = time.time()
    k, q = args.k, args.q

    # ---- G10-A: +4 dB sensing headroom replication --------------------
    g10a = {}
    for s in range(args.n_scenarios):
        sc = build_distributed_scenario(np.random.default_rng(s),
                                        k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(sc, args.alpha, args.beta, n_runs=300,
                                     seed=args.calib_seed,
                                     llr_bits=TOKEN_LLR_BITS,
                                     verify_runs=args.calib_verify)
        bounds = [[bt[qq][0], bt[qq][1] - args.b_delta] for qq in range(q)]
        sc_hi = build_distributed_scenario(np.random.default_rng(s),
                                           k_uavs=k, q_targets=q,
                                           snr_shift=SNR_SHIFT)
        bt_hi = calibrate_target_bounds(sc_hi, args.alpha, args.beta,
                                        n_runs=300, seed=args.calib_seed,
                                        llr_bits=TOKEN_LLR_BITS,
                                        verify_runs=args.calib_verify)
        bounds_hi = [[bt_hi[qq][0], bt_hi[qq][1] - args.b_delta]
                     for qq in range(q)]
        Jc = eval_sim(sc, bounds, args.n_runs, args.seeds, args.max_steps)
        Jh = eval_sim(sc_hi, bounds_hi, args.n_runs, args.seeds,
                      args.max_steps)
        g10a[str(s)] = {
            "J_current": Jc["J"], "J_plus4db": Jh["J"],
            "gap": float((Jc["J"] - Jh["J"]) / max(Jc["J"], 1e-12)),
        }
    gaps_a = [g10a[s]["gap"] for s in g10a]
    lo_a, hi_a = bootstrap_ci(gaps_a)
    g10a_summary = {
        "mean_gap": float(np.mean(gaps_a)),
        "ci95": [float(lo_a), float(hi_a)],
        "win_rate": float(np.mean([g > 0 for g in gaps_a])),
        "confirmed_bottleneck": bool(lo_a > 0.05),
    }
    print(f"[G10-A] +4dB gap {np.mean(gaps_a):+.1%} "
          f"CI [{lo_a:+.1%},{hi_a:+.1%}] win "
          f"{g10a_summary['win_rate']:.2f} confirmed "
          f"{g10a_summary['confirmed_bottleneck']}", flush=True)

    # ---- G10-B: energy-conserving power oracle ------------------------
    # current: per-target cap 2.0 everywhere (sum 2Q)
    cap_cur = np.full(q, 2.0)
    # oracle: weak target (0) high cap (~+4 dB from 2.0), easy targets
    # reduced, total sum of caps kept at 2Q (energy-neutral in expectation)
    cap_or = np.ones(q) * 1.0
    cap_or[0] = 5.0
    cap_or[1] = 2.0
    # distribute the remaining budget (2q - 7) over the easy targets as
    # 2.0's first, then 1.0's, so sum(cap_or) == 2q exactly
    rem = 2.0 * q - 7.0
    n_two = int(round(rem - (q - 2.0)))
    for qq in range(2, q):
        cap_or[qq] = 2.0 if (qq - 2) < n_two else 1.0
    assert abs(float(np.sum(cap_or)) - 2.0 * q) < 1e-9
    g10b = {}
    for s in range(args.n_oracle_scenarios):
        sc = build_distributed_scenario(np.random.default_rng(s),
                                        k_uavs=k, q_targets=q, powers=POWERS)
        bt_c = calibrate_target_bounds(sc, args.alpha, args.beta, n_runs=300,
                                       seed=args.calib_seed,
                                       llr_bits=TOKEN_LLR_BITS,
                                       verify_runs=args.calib_verify,
                                       power_cap=cap_cur)
        bounds_c = [[bt_c[qq][0], bt_c[qq][1] - args.b_delta]
                    for qq in range(q)]
        bt_o = calibrate_target_bounds(sc, args.alpha, args.beta, n_runs=300,
                                       seed=args.calib_seed,
                                       llr_bits=TOKEN_LLR_BITS,
                                       verify_runs=args.calib_verify,
                                       power_cap=cap_or)
        bounds_o = [[bt_o[qq][0], bt_o[qq][1] - args.b_delta]
                    for qq in range(q)]
        Jc = eval_sim(sc, bounds_c, args.n_runs, args.seeds, args.max_steps,
                      power_cap=cap_cur)
        Jo = eval_sim(sc, bounds_o, args.n_runs, args.seeds, args.max_steps,
                      power_cap=cap_or)
        g10b[str(s)] = {
            "J_current": Jc["J"], "J_oracle": Jo["J"],
            "power_current": Jc.get("sensing_power_per_uav"),
            "power_oracle": Jo.get("sensing_power_per_uav"),
            "dE": float((Jc["J"] - Jo["J"]) / max(Jc["J"], 1e-12)),
        }
        print(f"  [G10-B] s{s}: current {Jc['J']:.2f} (pow "
              f"{Jc.get('sensing_power_per_uav', float('nan')):.2f}) oracle "
              f"{Jo['J']:.2f} (pow "
              f"{Jo.get('sensing_power_per_uav', float('nan')):.2f}) "
              f"dE {g10b[str(s)]['dE']:+.1%}", flush=True)
    dEs = [g10b[s]["dE"] for s in g10b]
    lo_b, hi_b = bootstrap_ci(dEs)
    mean_de = float(np.mean(dEs))
    pow_c = np.mean([g10b[s]["power_current"] for s in g10b])
    pow_o = np.mean([g10b[s]["power_oracle"] for s in g10b])
    case = ("A (< 3%: allocation already good; +4 dB is hardware -> "
            "fixed-TB OTFS, no power algorithm)"
            if mean_de < 0.03
            else ("B (5-10%: medium headroom -> light sensing-energy "
                  "extension)"
                  if mean_de < 0.10
                  else "C (> 10%: sensing resource allocation is the real "
                       "bottleneck -> energy-priced FRIDS (G10-D))"))
    g10b_summary = {
        "mean_dE": float(mean_de),
        "ci95": [float(lo_b), float(hi_b)],
        "power_current_per_uav": float(pow_c),
        "power_oracle_per_uav": float(pow_o),
        "energy_neutral": bool(abs(pow_o - pow_c) / max(pow_c, 1e-12) < 0.05),
        "case": case,
        "cap_current": [float(x) for x in cap_cur],
        "cap_oracle": [float(x) for x in cap_or],
    }
    print(f"[G10-B] dE {mean_de:+.1%} CI [{lo_b:+.1%},{hi_b:+.1%}] "
          f"power {pow_c:.2f} -> {pow_o:.2f} | case: {case}", flush=True)

    gate = {
        "G10_A_sensing_confirmed": g10a_summary,
        "G10_B_energy_oracle": g10b_summary,
        "recommendation": (
            "confirm sensing is the bottleneck; then per case B/C do "
            "energy-priced FRIDS (G10-D) -- the same total energy, "
            "reallocated toward the weak target"
            if g10a_summary["confirmed_bottleneck"] and mean_de >= 0.03
            else (
                "sensing +4 dB is not confirmed as a >5% bottleneck over "
                "scenarios, OR the energy reallocation recovers < 3% (it "
                "is hardware headroom) -- do NOT do a power algorithm; "
                "audit fixed-TB OTFS (G10-C) next"
                if not g10a_summary["confirmed_bottleneck"] or mean_de < 0.03
                else "see cases")),
    }

    # ---- G10-C: fixed-TB OTFS evidence audit --------------------------
    g10c = {}
    if args.otfs_grids:
        grids = ((128, 32), (64, 64), (32, 128))    # N_d * N_l = 4096
        for s in range(args.n_otfs_scenarios):
            rng = np.random.default_rng(5000 + s)
            physics = {qq: (rng.random(), rng.random()) for qq in range(q)}
            per_grid = {}
            for grid in grids:
                sc_g = build_distributed_scenario(
                    np.random.default_rng(s), k_uavs=k, q_targets=q,
                    dd_grid=grid, dd_physics=physics)
                g_mean = float(np.mean([
                    g_reliable(sc_g, i, qq, sc_g["owner_of"])
                    for i in range(k) for qq in range(q)]))
                try:
                    bt_g = calibrate_target_bounds(
                        sc_g, args.alpha, args.beta, n_runs=300,
                        seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
                        verify_runs=args.calib_verify)
                    bounds_g = [[bt_g[qq][0], bt_g[qq][1] - args.b_delta]
                                for qq in range(q)]
                    Jg = eval_sim(sc_g, bounds_g, args.n_runs, args.seeds,
                                  args.max_steps)
                    per_grid[f"{grid[0]}x{grid[1]}"] = {
                        "J": Jg["J"], "g_mean": g_mean, "infeasible": False,
                    }
                except ValueError:
                    # the DD shape makes the (weak) target infeasible at
                    # the error operating point -- itself a finding
                    per_grid[f"{grid[0]}x{grid[1]}"] = {
                        "J": float(args.max_steps), "g_mean": g_mean,
                        "infeasible": True,
                    }
            g10c[str(s)] = {
                "physics": {str(qq): physics[qq] for qq in range(q)},
                "per_grid": per_grid,
            }
            print(f"  [G10-C] s{s}: "
                  + " | ".join(f"{gr}: J {per_grid[gr]['J']:.2f}"
                               + ("[INF]" if per_grid[gr]["infeasible"] else "")
                               + f" g {per_grid[gr]['g_mean']:.4f}"
                               for gr in per_grid), flush=True)
    if g10c:
        # across scenarios: max |dJ| between grids, and the g-scale spread
        j_by_grid = {gr: [g10c[s]["per_grid"][gr]["J"] for s in g10c]
                     for gr in ("128x32", "64x64", "32x128")}
        g_by_grid = {gr: [g10c[s]["per_grid"][gr]["g_mean"] for s in g10c]
                     for gr in ("128x32", "64x64", "32x128")}
        j_mean = {gr: float(np.mean(v)) for gr, v in j_by_grid.items()}
        j_min, j_max = min(j_mean.values()), max(j_mean.values())
        dJ_otfs = float((j_max - j_min) / max(j_min, 1e-12))
        # mean |g| spread across grids (the evidence scale sensitivity)
        g_spread = float(np.mean([
            (max(g_by_grid[gr][s] for gr in g_by_grid)
             - min(g_by_grid[gr][s] for gr in g_by_grid))
            / max(min(g_by_grid[gr][s] for gr in g_by_grid), 1e-12)
            for s in range(len(g10c))]))
        g10c_summary = {
            "J_by_grid_mean": {gr: float(j_mean[gr]) for gr in j_mean},
            "max_dJ_across_grids": float(dJ_otfs),
            "mean_g_spread_across_grids": float(g_spread),
            "otfs_enters_core": bool(dJ_otfs > 0.03),
            "verdict": (
                "OTFS DD resource SHAPE changes the evidence/FRIDS delay "
                "by > 3% under fixed TB -> OTFS enters the algorithm "
                "mechanism"
                if dJ_otfs > 0.03
                else "OTFS DD resource shape changes the delay by <= 3% "
                     "under fixed TB -> OTFS is an evidence-generation "
                     "waveform; the paper's claims do not depend on the "
                     "OTFS grid design"),
        }
        gate["G10_C_otfs_fixed_tb"] = g10c_summary
        print(f"[G10-C] J by grid {j_mean} | max dJ {dJ_otfs:.1%} | "
              f"g spread {g_spread:.1%} | verdict: "
              f"{g10c_summary['verdict']}", flush=True)

    payload = {
        "gate": "g10-resource-conserving-sensing-audit",
        "params": {
            "K": k, "Q": q, "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps, "alpha": args.alpha,
            "beta": args.beta, "calib_seed": args.calib_seed,
            "calib_verify": args.calib_verify, "b_delta": args.b_delta,
            "n_scenarios": args.n_scenarios,
            "n_oracle_scenarios": args.n_oracle_scenarios,
            "n_otfs_scenarios": args.n_otfs_scenarios,
            "snr_shift_db": SNR_SHIFT, "powers": list(POWERS),
            "otfs_grids_tb_product": 4096,
            "frozen": ["FRIDS-v2", "token", "owner", "U2U", "full mesh",
                       "calibration protocol"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "G10_A": g10a, "G10_A_summary": g10a_summary,
        "G10_B": g10b, "G10_B_summary": g10b_summary,
        "G10_C": g10c, "G10_C_summary": g10c_summary if g10c else None,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()