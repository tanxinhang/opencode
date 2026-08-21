"""Gate F0: distributed information audit (advice/005.md).

Stops the D3-A-style centralized deployment profile and runs the system
correction gate: the *same* distributed dual G-value decision rule with
bounded coordination is evaluated under four information structures that
differ only in what each UAV knows:

A. ``centralized``   -- global belief, same-cycle intents, perfect delivery
                        (oracle, offline audit only);
B. ``full_message``  -- local information sets, exact-evidence tokens
                        (decentralization cost);
C. ``compact_token`` -- local information sets, quantized 20-bit token
                        (communication/compression cost);
D. ``local_only``    -- zero communication (cooperation value).

The audit answers the three questions of advice/005 section 15 on the
worst-target H1 detection delay with calibrated two-threshold stopping,
reports the stability metrics of section 18, and writes
``results/distributed_audit_gate.json``.
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
    MODES,
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
    run_audit,
    simulate_system,
)

def token_feasibility_scan(scenario, alpha, beta, bits_list=(4, 5, 6)):
    """Feasibility of the calibrated two-threshold rule per token bit
    count (advice/005 section 20 Contribution 4: the infeasible region).
    With too few LLR bits the quantized evidence cannot meet the error
    constraints at all."""
    rows = {}
    for bits in bits_list:
        try:
            bounds = calibrate_target_bounds(
                scenario, alpha, beta, n_runs=300, llr_bits=bits,
            )
            rows[str(bits)] = {
                "feasible": True,
                "bounds": [[round(float(b[0]), 3), round(float(b[1]), 3)]
                           for b in bounds],
            }
        except ValueError:
            rows[str(bits)] = {"feasible": False}
    return rows


def calibration_seed_sensitivity(scenario, alpha, beta, n_runs, seeds,
                                 max_steps, eta, cal_verify,
                                 extra_seeds=(200, 300), scan_runs=300):
    """Range of the compact-token gap over calibration seeds.

    The two-stage calibration is MC-based; near-tied feasible boundaries
    can shift ``Delta_comm`` by a few tenths of a cycle.  The audit
    reports the official calibration seed (100) plus this range so the
    communication verdict is stated with its honest uncertainty.
    """
    q = scenario["q"]
    nu = tuple([1.0 / q] * q)
    rows = {}
    for cal_seed in (100,) + tuple(extra_seeds):
        be = calibrate_target_bounds(scenario, alpha, beta,
                                     n_runs=scan_runs, seed=cal_seed,
                                     verify_runs=cal_verify)
        bt = calibrate_target_bounds(scenario, alpha, beta,
                                     n_runs=scan_runs, seed=cal_seed,
                                     llr_bits=TOKEN_LLR_BITS,
                                     verify_runs=cal_verify)
        singles = build_target_values(scenario, be, horizon=max_steps,
                                      nu=nu)
        jb, jc = [], []
        for seed in range(seeds):
            out_b = simulate_system("full_message", scenario, be, singles,
                                    n_runs=n_runs, seed=seed * 1000 + 7,
                                    max_steps=max_steps, nu=nu, eta=eta)
            out_c = simulate_system("compact_token", scenario, bt, singles,
                                    n_runs=n_runs, seed=seed * 1000 + 7,
                                    max_steps=max_steps, nu=nu, eta=eta)
            jb.append(out_b["worst_target_delay"])
            jc.append(out_c["worst_target_delay"])
        jb_m, jc_m = float(np.mean(jb)), float(np.mean(jc))
        rows[str(cal_seed)] = {
            "bounds_exact": [[round(float(b[0]), 3), round(float(b[1]), 3)]
                             for b in be],
            "bounds_token": [[round(float(b[0]), 3), round(float(b[1]), 3)]
                             for b in bt],
            "J_full_message": round(jb_m, 3),
            "J_compact_token": round(jc_m, 3),
            "Delta_comm": round(jc_m - jb_m, 3),
            "Delta_comm_relative": round((jc_m - jb_m) / max(jb_m, 1e-12),
                                         4),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/distributed_audit_gate.json")
    parser.add_argument("--k-uavs", type=int, default=6)
    parser.add_argument("--q-targets", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify-runs", type=int, default=2000)
    parser.add_argument("--token-llr-bits", type=int, default=TOKEN_LLR_BITS,
                        help="LLR bits of the compact token (F1 sweep); "
                             "4 is infeasible in the tested scenario class")
    args = parser.parse_args()

    import uav_otfs_isac.distributed_audit as _da
    _da.TOKEN_LLR_BITS = args.token_llr_bits

    rng = np.random.default_rng(args.scenario_seed)
    scenario = build_distributed_scenario(
        rng, k_uavs=args.k_uavs, q_targets=args.q_targets,
    )
    t0 = time.time()
    feasibility = token_feasibility_scan(
        scenario, args.alpha, args.beta,
        bits_list=(4, 5, 6),
    )
    audit = run_audit(
        scenario,
        alpha=args.alpha, beta=args.beta,
        n_runs=args.n_runs, seeds=args.seeds,
        max_steps=args.max_steps, eta=args.eta,
        calib_seed=args.calib_seed,
        calib_verify_runs=args.calib_verify_runs,
    )
    sensitivity = calibration_seed_sensitivity(
        scenario, args.alpha, args.beta,
        n_runs=max(300, args.n_runs // 2), seeds=args.seeds,
        max_steps=args.max_steps, eta=args.eta,
        cal_verify=args.calib_verify_runs,
    )
    payload = {
        "gate": "f0-distributed-information-audit",
        "params": {
            "k_uavs": args.k_uavs, "q_targets": args.q_targets,
            "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "scenario_seed": args.scenario_seed, "eta": args.eta,
            "token_llr_bits": _da.TOKEN_LLR_BITS,
            "calib_seed": args.calib_seed,
            "calib_verify_runs": args.calib_verify_runs,
        },
        "runtime_s": round(time.time() - t0, 1),
        "token_feasibility": feasibility,
        "calibration_seed_sensitivity": sensitivity,
        **audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"K={args.k_uavs} UAVs, Q={args.q_targets} targets, "
          f"bounds_exact={audit['bounds_exact']}, "
          f"bounds_token={audit['bounds_token']}")
    print(f"token feasibility: {feasibility}")
    for cs, row in sensitivity.items():
        print(f"cal-seed {cs}: Delta_comm={row['Delta_comm']:+.3f} "
              f"({row['Delta_comm_relative']:+.2%})")
    print(f"{'mode':<14}{'worst E1[T]':>12}{'p_fa max':>10}{'p_md max':>10}"
          f"{'bits/cyc':>10}{'conflict':>10}{'duplicate':>10}"
          f"{'D_L':>10}")
    for mode in MODES:
        s = audit["modes"][mode]
        dl = s["belief_disagreement"]
        print(f"{mode:<14}{s['worst_target_delay']:>12.2f}"
              f"{max(s['p_fa']):>10.3f}{max(s['p_md']):>10.3f}"
              f"{s['mean_u2u_bits_per_cycle']:>10.1f}"
              f"{s['conflict_rate']:>10.3f}"
              f"{s['duplicate_sensing_rate']:>10.3f}"
              f"{'-' if dl is None else round(dl, 3):>10}")
    for name, g in audit["gaps"].items():
        print(f"Delta_{name:<10} = {g['value']:+.3f} "
              f"({g['relative']:+.3%})")
    for name, q_ in audit["questions"].items():
        print(f"Q-{name}: {q_['answer']} -- {q_['comment']}")
    print(f"ordering_holds={audit['ordering_holds']}, "
          f"error_constraints_met={audit['error_constraints_met']}, "
          f"passed={audit['passed']}")


if __name__ == "__main__":
    main()
