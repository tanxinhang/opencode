"""Deep audit B: F0-G8A/G8B robustness and mechanism (2026-08-19).

Probes:

1. G8-B statistical significance: Step1 (value swap, independent world,
   rho_s = 0.5) and Step2 (correlated world, rho_s = world_rho = 0.5)
   paired Delta J vs the proper references over scenario draws, with a
   paired bootstrap 95% CI and per-scenario signs.
2. G8-B coalition-staleness: the deployable intent-based coalition vs a
   perfect previous-cycle-serving-set oracle -- how much does the
   strictly-local 1-cycle-stale estimate cost?
3. G8-B rho_s monotonicity: the Step1 Delta J curve over
   rho_s in {0.0, 0.1, 0.2, 0.3, 0.5} -- is the improvement monotone or
   does it peak (overfitting risk)?
4. G8-A model sensitivity: the redundancy R_q(S) under a HETEROGENEOUS
   within-target profile (one dominant UAV) vs the tested homogeneous
   profile -- is the R ~ 80% verdict an artifact of the scenario
   generator's within-target homogeneity?

Writes ``results/deep_audit_g8.json``.
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

from uav_otfs_isac.conditional_frids import (
    observation_delta_matrix,
    reliable_delta_matrix,
    simulate_frids_v2_cond,
)
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.evidence_correlation import (
    build_delta_from_scenario,
    redundancy,
)
from uav_otfs_isac.frids import simulate_frids_v2

RHO_S_MONO = (0.0, 0.1, 0.2, 0.3, 0.5)


def eval_sim(sim, sc, bounds, n_runs, seeds, max_steps, **kw):
    J = []
    for seed in range(seeds):
        out = sim(sc, bounds, n_runs=n_runs, seed=seed * 1000 + 7,
                  max_steps=max_steps, **kw)
        J.append(out["worst_target_delay"])
    return float(np.mean(J))


def bootstrap_ci(deltas, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    draws = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/deep_audit_g8.json")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=400)
    parser.add_argument("--b-delta", type=float, default=1.0)
    parser.add_argument("--scenario-seeds", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    k, q = args.k, args.q

    # ---- probe 4 (G8-A model sensitivity), cheap, do first -----------
    sc0 = build_distributed_scenario(np.random.default_rng(0),
                                     k_uavs=k, q_targets=q)
    owner = sc0["owner_of"]
    delta_hom = build_delta_from_scenario(sc0, owner)
    R_hom = redundancy(delta_hom[:, 0], list(range(k)), 0.5)
    # heterogeneous profile: one dominant UAV, the rest weak (a -> 1)
    delta_het = np.zeros(k)
    delta_het[0] = float(np.max(delta_hom[:, 0]))
    delta_het[1:] = 0.15 * float(np.max(delta_hom[:, 0]))
    R_het = redundancy(delta_het, list(range(k)), 0.5)
    a_het = float(np.sum(delta_het) ** 2 / np.sum(delta_het ** 2))
    probe4 = {
        "R_homogeneous_rho05": float(R_hom),
        "R_heterogeneous_rho05": float(R_het),
        "alignment_heterogeneous": float(a_het),
        "finding": (
            "the R ~ 80% verdict is specific to the homogeneous "
            "within-target profile; a concentrated profile (one dominant "
            "UAV, alignment -> 1) has much smaller redundancy"
            if R_het < R_hom else "redundancy high for both profiles"),
    }

    # ---- G8-B probes over scenario draws -----------------------------
    per = {}
    for s in range(args.scenario_seeds):
        sc = build_distributed_scenario(np.random.default_rng(s),
                                        k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(sc, args.alpha, args.beta, n_runs=300,
                                     seed=args.calib_seed,
                                     llr_bits=TOKEN_LLR_BITS,
                                     verify_runs=args.calib_verify)
        bounds = [[bt[qq][0], bt[qq][1] - args.b_delta] for qq in range(q)]
        base = eval_sim(simulate_frids_v2, sc, bounds, args.n_runs,
                        args.seeds, args.max_steps)
        # step1 rho_s=0.5 (intent coalition)
        c05 = eval_sim(simulate_frids_v2_cond, sc, bounds, args.n_runs,
                       args.seeds, args.max_steps, rho_s=0.5, world_rho=0.0)
        # step1 perfect coalition
        c05p = eval_sim(simulate_frids_v2_cond, sc, bounds, args.n_runs,
                        args.seeds, args.max_steps, rho_s=0.5, world_rho=0.0,
                        coalition_mode="perfect")
        # step2 consistent (rho_s = world_rho = 0.5)
        cw = eval_sim(simulate_frids_v2_cond, sc, bounds, args.n_runs,
                      args.seeds, args.max_steps, rho_s=0.5, world_rho=0.5)
        # step2 singleton value under correlated world
        cw0 = eval_sim(simulate_frids_v2_cond, sc, bounds, args.n_runs,
                       args.seeds, args.max_steps, rho_s=0.0, world_rho=0.5)
        # monotonicity curve
        mono = {str(rho): eval_sim(simulate_frids_v2_cond, sc, bounds,
                                   args.n_runs, args.seeds, args.max_steps,
                                   rho_s=rho, world_rho=0.0)
                for rho in RHO_S_MONO}
        per[str(s)] = {
            "baseline": base,
            "step1_rho05": c05,
            "step1_perfect": c05p,
            "step2_consistent": cw,
            "step2_singleton_world05": cw0,
            "monotonicity": mono,
        }
        print(f"  scenario {s}: base {base:.3f} | cond0.5 {c05:.3f} "
              f"(dJ {(base-c05)/max(base,1e-12):+.1%}) | perfect "
              f"{c05p:.3f} | step2 cons {cw:.3f} vs single {cw0:.3f}",
              flush=True)

    # aggregate paired Delta J with bootstrap CI
    def paired_dJ(get_c, get_ref=None):
        djs = []
        for s in per:
            base = per[s]["baseline"] if get_ref is None else get_ref(per[s])
            c = get_c(per[s])
            djs.append((base - c) / max(base, 1e-12))
        lo, hi = bootstrap_ci(djs)
        return {"mean": float(np.mean(djs)),
                "ci95": [float(lo), float(hi)],
                "per_scenario": [float(x) for x in djs],
                "sign_consistent": bool(
                    np.sum(np.array(djs) > 0) >= len(djs) - 1
                    or np.sum(np.array(djs) < 0) >= len(djs) - 1)}

    step1 = paired_dJ(lambda p: p["step1_rho05"])
    step2 = paired_dJ(lambda p: p["step2_consistent"])
    # step2 consistent vs singleton value under the SAME correlated world
    step2_vs_single = paired_dJ(
        lambda p: p["step2_consistent"],
        lambda p: p["step2_singleton_world05"])
    # staleness: perfect vs intent coalition (step1 rho_s=0.5)
    staleness_djs = []
    for s in per:
        c = per[s]["step1_rho05"]
        cp = per[s]["step1_perfect"]
        staleness_djs.append((cp - c) / max(c, 1e-12))
    # monotonicity: is the step1 dJ curve monotone decreasing in rho_s?
    mono_summary = {}
    for s in per:
        mono = per[s]["monotonicity"]
        base = per[s]["baseline"]
        mono_summary[s] = {
            str(rho): float((base - mono[str(rho)]) / max(base, 1e-12))
            for rho in RHO_S_MONO}

    payload = {
        "audit": "deep-audit-B-g8",
        "params": {"K": k, "Q": q, "n_runs": args.n_runs,
                   "seeds": args.seeds, "max_steps": args.max_steps,
                   "alpha": args.alpha, "beta": args.beta,
                   "calib_seed": args.calib_seed,
                   "calib_verify": args.calib_verify,
                   "b_delta": args.b_delta,
                   "scenario_seeds": args.scenario_seeds,
                   "rho_s_monotonicity": list(RHO_S_MONO)},
        "runtime_s": round(time.time() - t0, 1),
        "probe4_G8A_model_sensitivity": probe4,
        "probe1_G8B_statistics": {
            "step1_rho05_vs_v2": step1,
            "step2_consistent_vs_v2": step2,
            "step2_consistent_vs_singleton_world": step2_vs_single,
        },
        "probe2_G8B_staleness": {
            "per_scenario_dJ_perfect_vs_intent": [float(x)
                                                  for x in staleness_djs],
            "mean": float(np.mean(staleness_djs)),
        },
        "probe3_G8B_monotonicity": mono_summary,
        "per_scenario": per,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "probe4_G8A_model_sensitivity", "probe1_G8B_statistics",
        "probe2_G8B_staleness")}, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()