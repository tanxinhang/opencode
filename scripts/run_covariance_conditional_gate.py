"""Gate F0-G8C: Covariance-native conditional information (advice/018).

The scalar-rho common-factor model is replaced by the covariance-native
Schur-complement conditional information

    Delta G_{i|S,q} = (1/2) delta_{i|S}^2 / v_{i|S},
    delta_{i|S} = delta_i - c^T Sigma_SS^{-1} delta_S,
    v_{i|S}     = sigma_i^2 - c^T Sigma_SS^{-1} c,

from the true evidence covariance Sigma_q (OTFS/DD physics source).  The
gate compares, under the covariance-native correlated world (service
accounting = G_q(S_del) with the true Sigma):

1. FRIDS-v2 singleton (value = g_i, ignores correlation);
2. scalar-rho conditional FRIDS (value = common-factor Delta G at
   rho = mean off-diagonal covariance);
3. covariance-native conditional FRIDS (value = Schur Delta G);
4. oracle (covariance-native value + perfect-coalition estimate).

The Gaussian-evidence FRIDS simulation
(`simulate_gaussian_frids`) is used so the marginal profile and the
belief drift are consistent (the Gaussian layer is the system's evidence
model).  Three profile classes are mandatory (deep-audit lesson): 
homogeneous, heterogeneous, concentrated.  N_scenario = 10 per profile.

Life gate (advice/018 section 10):
- correlated-world mean gain (covariance vs singleton) >= 5%,
  bootstrap CI_95,low > 0, win rate >= 8/10;
- independent world |Delta J| <= 2%;
- concentrated profile: no > 2% regression;
- P_FA <= alpha, P_MD <= beta + eps_MC.

Writes ``results/covariance_conditional_gate.json``.
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

from uav_otfs_isac.covariance_conditional import (
    build_profile_moments,
    scalar_rho_from_covariance,
    shrink_covariance,
    simulate_gaussian_frids,
)

PROFILES = ("homogeneous", "heterogeneous", "concentrated")
K, Q = 8, 4


def eval_method(delta, sigma, owner, value_mode, n_runs, seeds, max_steps,
                alpha, beta, rho_s=0.0, coalition_mode="intent",
                delivery=None, reliable=False):
    J, md, fa = [], [], []
    for seed in range(seeds):
        out = simulate_gaussian_frids(
            delta, sigma, owner, alpha=alpha, beta=beta, n_runs=n_runs,
            seed=seed * 1000 + 7, max_steps=max_steps,
            value_mode=value_mode, rho_s=rho_s,
            coalition_mode=coalition_mode, delivery=delivery,
            reliable=reliable)
        J.append(out["worst_target_delay"])
        md.append(max(out["p_md"]))
        fa.append(max(out["p_fa"]))
    return {"J": float(np.mean(J)), "p_md_max": float(max(md)),
            "p_fa_max": float(max(fa))}


def bootstrap_ci(deltas, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    draws = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/covariance_conditional_gate.json")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--q", type=int, default=Q)
    parser.add_argument("--n-runs", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument("--reliable", action="store_true",
                        help="weight the value by the U2U delivery success "
                             "(FRIDS-v2 reliable-information convention)")
    args = parser.parse_args()

    t0 = time.time()
    k, q = args.k, args.q
    per_profile = {}
    for profile in PROFILES:
        cells = {}
        for s in range(args.n_scenarios):
            rng = np.random.default_rng(1000 + s + 100 * PROFILES.index(profile))
            m = build_profile_moments(k, q, profile, rng)
            delta, sigma, owner = m["delta"], m["sigma"], \
                [int(qq % k) for qq in range(q)]
            rho = scalar_rho_from_covariance(sigma)
            deliv = 0.6 + 0.35 * rng.random((k, k))
            deliv = 0.5 * (deliv + deliv.T)
            np.fill_diagonal(deliv, 1.0)
            kw = {"delivery": deliv, "reliable": args.reliable}
            single = eval_method(delta, sigma, owner, "singleton",
                                 args.n_runs, args.seeds, args.max_steps,
                                 args.alpha, args.beta, **kw)
            rho_m = eval_method(delta, sigma, owner, "rho",
                                args.n_runs, args.seeds, args.max_steps,
                                args.alpha, args.beta, rho_s=rho, **kw)
            cov = eval_method(delta, sigma, owner, "covariance",
                              args.n_runs, args.seeds, args.max_steps,
                              args.alpha, args.beta, **kw)
            oracle = eval_method(delta, sigma, owner, "covariance",
                                 args.n_runs, args.seeds, args.max_steps,
                                 args.alpha, args.beta,
                                 coalition_mode="perfect", **kw)
            cells[str(s)] = {
                "singleton": single, "rho": rho_m, "covariance": cov,
                "oracle": oracle,
                "rho_s": float(rho),
                "gain_cov_vs_single": float(
                    (single["J"] - cov["J"]) / max(single["J"], 1e-12)),
                "gain_cov_vs_rho": float(
                    (rho_m["J"] - cov["J"]) / max(rho_m["J"], 1e-12)),
                "gain_oracle_vs_cov": float(
                    (cov["J"] - oracle["J"]) / max(cov["J"], 1e-12)),
            }
        # independent-world check: Sigma = I (value = singleton)
        ind = {}
        for s in range(args.n_scenarios):
            rng = np.random.default_rng(2000 + s + 100 * PROFILES.index(profile))
            m = build_profile_moments(k, q, profile, rng)
            sig_i = shrink_covariance(np.eye(k), 0.0)
            owner = [int(qq % k) for qq in range(q)]
            a = eval_method(m["delta"], sig_i, owner, "singleton",
                            args.n_runs, args.seeds, args.max_steps,
                            args.alpha, args.beta)
            b = eval_method(m["delta"], sig_i, owner, "covariance",
                            args.n_runs, args.seeds, args.max_steps,
                            args.alpha, args.beta)
            ind[str(s)] = {"dJ": float((b["J"] - a["J"]) / max(a["J"], 1e-12))}
        gains = [cells[s]["gain_cov_vs_single"] for s in cells]
        lo, hi = bootstrap_ci(gains)
        per_profile[profile] = {
            "cells": cells,
            "gain_cov_vs_single_mean": float(np.mean(gains)),
            "ci95": [float(lo), float(hi)],
            "win_rate": float(np.mean([g > 0 for g in gains])),
            "independent_world": ind,
            "independent_world_max_abs_dJ": float(
                max(abs(v["dJ"]) for v in ind.values())),
            "concentrated_regression": float(
                max(-g for g in gains)) if profile == "concentrated" else 0.0,
        }
        print(f"  {profile}: cov-vs-single gain {np.mean(gains):+.1%} "
              f"CI [{lo:+.1%},{hi:+.1%}] win {per_profile[profile]['win_rate']:.2f} "
              f"| ind-world |dJ| {per_profile[profile]['independent_world_max_abs_dJ']:.2%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- life gate ---------------------------------------------------
    gate = {}
    for profile in PROFILES:
        p = per_profile[profile]
        gain = p["gain_cov_vs_single_mean"]
        lo = p["ci95"][0]
        win = p["win_rate"]
        ind_ok = p["independent_world_max_abs_dJ"] <= 0.02
        errors_ok = all(
            cells[str(s)]["covariance"]["p_md_max"] <= args.beta + 0.02
            and cells[str(s)]["covariance"]["p_fa_max"] <= args.alpha + 0.02
            for s in p["cells"])
        pass_ = bool(gain >= 0.05 and lo > 0.0 and win >= 0.8
                     and ind_ok and errors_ok)
        gate[profile] = {
            "gain": float(gain), "ci95_lo": float(lo), "win_rate": float(win),
            "independent_world_ok": bool(ind_ok), "errors_ok": bool(errors_ok),
            "pass": bool(pass_),
        }
    # overall: the covariance-native version is adopted only if it passes
    # on the correlated worlds AND never regresses the concentrated profile
    # by more than 2%
    concentrated_reg = per_profile["concentrated"]["concentrated_regression"]
    overall_pass = bool(
        gate["homogeneous"]["pass"] and gate["heterogeneous"]["pass"]
        and concentrated_reg <= 0.02)
    gate_summary = {
        "per_profile": gate,
        "concentrated_regression": float(concentrated_reg),
        "overall_pass": overall_pass,
        "verdict": (
            "adopt covariance-native conditional FRIDS (passes the life "
            "gate on homogeneous + heterogeneous, no concentrated "
            "regression)"
            if overall_pass
            else "rejected: keep FRIDS-v2, close the correlation-scheduler "
                 "main contribution"),
    }

    payload = {
        "gate": "f0g8c-covariance-native-conditional",
        "params": {
            "K": k, "Q": q, "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps, "alpha": args.alpha,
            "beta": args.beta, "n_scenarios": args.n_scenarios,
            "profiles": list(PROFILES),
            "value": "Schur Delta G = (1/2) delta_{i|S}^2 / v_{i|S} "
                     "from Sigma_q (OTFS/DD physics source)",
            "world": "covariance-native G_q(S_del) service accounting",
        },
        "runtime_s": round(time.time() - t0, 1),
        "per_profile": per_profile,
        "gate": gate_summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate_summary, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()