"""Gate F0-A fix experiments (advice/007 section 8).

F0-A classified the scale loss as starvation + over-concentration.  The
prescribed first fix is ONE scalar: the starvation-age term
``J' = J + eta_A * A_q(t)`` (cycles since the target's last service),
swept with everything else frozen; the comparison fix is the congestion
price curvature ``psi = -eta * n_q^gamma`` with ``gamma > 1``.  Each fix
is applied alone, at the two scales where Gate A failed (12,6) and
(16,8).  The winner (minimal worst-target delay subject to the error
constraints) is then run at formal MC.

Writes ``results/allocation_fix_gate.json``.
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

from uav_otfs_isac.competition_audit import simulate_competition_audit
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
)

FIX_SCALES = ((12, 6), (16, 8))
ETA_A_GRID = (0.0, 0.2, 0.4, 0.8)
GAMMA_GRID = (1.0, 1.5, 2.0)


def build_frozen(scenario, alpha, beta, max_steps, calib_seed):
    """Frozen per-scale machinery (token-mode calibration only; the
    compact-token mainline is the audit subject)."""
    bt = calibrate_target_bounds(scenario, alpha, beta,
                                 n_runs=300, seed=calib_seed,
                                 llr_bits=TOKEN_LLR_BITS,
                                 verify_runs=2000)
    q = scenario["q"]
    nu = tuple([1.0 / q] * q)
    singles = build_target_values(scenario, bt, horizon=max_steps, nu=nu)
    return bt, singles, nu


def eval_config(scenario, bt, singles, nu, n_runs, seeds, max_steps,
                eta_A=0.0, psi_gamma=1.0, eta=0.5, normalize_gains=False):
    """J, P_MD, P_FA, r_min, H_max_idle of one config."""
    J = []
    p_md = []
    p_fa = []
    r_min = []
    h_idle = []
    for seed in range(seeds):
        out = simulate_competition_audit(
            scenario, bt, singles, nu, n_runs=n_runs,
            seed=seed * 1000 + 7, max_steps=max_steps,
            eta=eta, psi_gamma=psi_gamma, eta_A=eta_A,
            normalize_gains=normalize_gains,
        )
        J.append(out["worst_target_delay"])
        p_md.append(max(out["p_md"]))
        p_fa.append(max(out["p_fa"]))
        r_min.append(out["r_min"])
        h_idle.append(out["H_max_idle"])
    return {
        "J": float(np.mean(J)),
        "p_md_max": float(np.max(p_md)),
        "p_fa_max": float(np.max(p_fa)),
        "r_min": float(np.mean(r_min)),
        "H_max_idle": float(np.mean(h_idle)),
    }


# configs: (name, kwargs).  The F0-A audit found the additive price and
# age terms numerically inert against the 1e9-scaled in-band gains; the
# normalized-index family (normalize_gains=True) puts the price on the
# decision scale, then sweeps ONE scalar at a time (price scale eta,
# price curvature gamma, age weight eta_A).
def configs():
    out = [("baseline", {})]
    out.append(("norm_base", {"normalize_gains": True}))
    for eta in (1.0, 2.0):
        out.append((f"norm_eta_{eta:g}",
                    {"normalize_gains": True, "eta": eta}))
    out.append(("norm_eta2_gamma2",
                {"normalize_gains": True, "eta": 2.0, "psi_gamma": 2.0}))
    for eta_A in (0.2, 0.4):
        out.append((f"norm_age_{eta_A:g}",
                    {"normalize_gains": True, "eta_A": eta_A}))
    out.append(("norm_eta2_age04",
                {"normalize_gains": True, "eta": 2.0, "eta_A": 0.4}))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/allocation_fix_gate.json")
    parser.add_argument("--n-runs", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--formal-n-runs", type=int, default=500)
    parser.add_argument("--formal-seeds", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=100)
    args = parser.parse_args()

    t0 = time.time()
    rows = {}
    for (k, q) in FIX_SCALES:
        rng = np.random.default_rng(args.scenario_seed)
        scenario = build_distributed_scenario(rng, k_uavs=k,
                                              q_targets=q)
        bt, singles, nu = build_frozen(scenario, args.alpha, args.beta,
                                       args.max_steps, args.calib_seed)
        scale = f"{k}_{q}"
        rows[scale] = {"k": k, "q": q, "configs": {}}
        for name, kwargs in configs():
            rows[scale]["configs"][name] = eval_config(
                scenario, bt, singles, nu, args.n_runs, args.seeds,
                args.max_steps, **kwargs)
        # winner: min J subject to the error constraints
        tol = 0.02
        candidates = [
            (name, cfg) for name, cfg in rows[scale]["configs"].items()
            if cfg["p_md_max"] <= args.beta + tol
            and cfg["p_fa_max"] <= args.alpha + tol
        ]
        best_name, best_cfg = min(candidates, key=lambda nc: nc[1]["J"])
        kwargs = dict(next(kw for n, kw in configs() if n == best_name))
        formal = eval_config(scenario, bt, singles, nu,
                             args.formal_n_runs, args.formal_seeds,
                             args.max_steps, **kwargs)
        rows[scale]["winner"] = {"name": best_name, "formal": formal}
        rows[scale]["baseline_formal"] = eval_config(
            scenario, bt, singles, nu, args.formal_n_runs,
            args.formal_seeds, args.max_steps)
    payload = {
        "gate": "f0a-allocation-fixes",
        "params": {
            "scales": [list(s) for s in FIX_SCALES],
            "n_runs": args.n_runs, "seeds": args.seeds,
            "formal_n_runs": args.formal_n_runs,
            "formal_seeds": args.formal_seeds,
            "max_steps": args.max_steps,
            "alpha": args.alpha, "beta": args.beta,
            "scenario_seed": args.scenario_seed,
            "calib_seed": args.calib_seed,
            "frozen": ["fixed owner", "full mesh", "19-bit token",
                       "calibrated two-threshold stopping", "current "
                       "scenario gen"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for scale, row in payload["rows"].items():
        print(f"scale {scale}:")
        print(f"  {'config':<16}{'J':>8}{'P_MD':>8}{'P_FA':>8}"
              f"{'r_min':>8}{'H_idle':>8}")
        for name, cfg in row["configs"].items():
            print(f"  {name:<16}{cfg['J']:>8.2f}{cfg['p_md_max']:>8.3f}"
                  f"{cfg['p_fa_max']:>8.3f}{cfg['r_min']:>8.3f}"
                  f"{cfg['H_max_idle']:>8.2f}")
        w = row["winner"]
        bf = row["baseline_formal"]
        print(f"  winner={w['name']}  formal J {bf['J']:.2f} -> "
              f"{w['formal']['J']:.2f} "
              f"(P_MD {bf['p_md_max']:.3f}->{w['formal']['p_md_max']:.3f}, "
              f"r_min {bf['r_min']:.3f}->{w['formal']['r_min']:.3f})")


if __name__ == "__main__":
    main()
