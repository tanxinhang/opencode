"""Gate F0-G8B Step 1: conditional-reliable-information FRIDS value swap
(advice/013 section 7, advice/015 section 4).

Only the scheduling VALUE changes: `g_iq` (singleton reliable info) is
replaced by the KL-chain-rule conditional marginal
`Delta G_{i|S_q,q} = G_q(S_q union {i}) - G_q(S_q)` with the coalition
`S_q^{(i)}` inferred STRICTLY locally from the intents UAV i actually
received (a_{i,t} = pi_i(I_{i,t})).  The world stays independent
(world_rho = 0), so this isolates the ALLOCATION effect of the
conditional value: at rho_s = 0 the conditional scheduler is identical
to FRIDS-v2; at rho_s > 0 it discounts redundant reports and should
spread the sensing/concentration differently.

Step 2 (correlated world, world_rho > 0) is reported separately when
the allocation effect is established.

Life-or-death gate (Step 1): the conditional value at rho_s in {0.2,
0.5} must not regress the worst-target delay by more than 2% while
keeping the errors within beta + 2pp -- the conditional scheduler is
only adopted if it is at least as good as the frozen singleton mainline
under the independent world (a robustness requirement: believing
correlation that is not there must not hurt).

Writes ``results/conditional_frids_gate.json``.
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

from uav_otfs_isac.conditional_frids import simulate_frids_v2_cond
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.frids import simulate_frids_v2

RHO_S_GRID = (0.0, 0.2, 0.5)
WORLD_RHO_GRID = (0.0, 0.2, 0.5)


def eval_method(sim, sc, bounds, n_runs, seeds, max_steps, **kw):
    J, sd, md, fa = [], [], [], []
    e1_by_q = []
    for seed in range(seeds):
        out = sim(sc, bounds, n_runs=n_runs, seed=seed * 1000 + 7,
                  max_steps=max_steps, **kw)
        J.append(out["worst_target_delay"])
        md.append(max(out["p_md"]))
        fa.append(max(out["p_fa"]))
        e1_by_q.append(out["e1_delays"])
    return {
        "J": float(np.mean(J)),
        "J_sd": float(np.std(J, ddof=1)) if len(J) > 1 else 0.0,
        "p_md_max": float(max(md)),
        "p_fa_max": float(max(fa)),
        "e1_delays": [float(np.mean([r[q] for r in e1_by_q]))
                      for q in range(sc["q"])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/conditional_frids_gate.json")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=500)
    parser.add_argument("--b-delta", type=float, default=1.0)
    parser.add_argument("--scenario-seeds", type=int, default=1,
                        help="number of scenario draws (scenario_seed "
                             "0..N-1) to average over for robustness")
    args = parser.parse_args()

    t0 = time.time()
    k, q = args.k, args.q

    def run_scenario(seed: int):
        sc = build_distributed_scenario(np.random.default_rng(seed),
                                        k_uavs=k, q_targets=q)
        bt = calibrate_target_bounds(
            sc, args.alpha, args.beta, n_runs=300,
            seed=args.calib_seed, llr_bits=TOKEN_LLR_BITS,
            verify_runs=args.calib_verify)
        bounds = [[bt[qq][0], bt[qq][1] - args.b_delta] for qq in range(q)]
        baseline = eval_method(simulate_frids_v2, sc, bounds, args.n_runs,
                               args.seeds, args.max_steps)
        step1 = {}
        for rho_s in RHO_S_GRID:
            row = eval_method(simulate_frids_v2_cond, sc, bounds,
                              args.n_runs, args.seeds, args.max_steps,
                              rho_s=rho_s, world_rho=0.0)
            dJ = (row["J"] - baseline["J"]) / max(baseline["J"], 1e-12)
            step1[str(rho_s)] = {**row, "delta_J_vs_v2": float(dJ)}
        step2 = {}
        for world_rho in WORLD_RHO_GRID:
            if world_rho == 0.0:
                continue
            for rho_s in RHO_S_GRID:
                row = eval_method(simulate_frids_v2_cond, sc, bounds,
                                  args.n_runs, args.seeds, args.max_steps,
                                  rho_s=rho_s, world_rho=world_rho)
                dJ = (row["J"] - baseline["J"]) / max(baseline["J"], 1e-12)
                step2[f"world{world_rho}_rho{rho_s}"] = {
                    **row, "world_rho": float(world_rho),
                    "rho_s": float(rho_s), "delta_J_vs_v2": float(dJ),
                }
        return {"baseline": baseline, "step1": step1, "step2": step2}

    per_scenario = {}
    for s in range(args.scenario_seeds):
        per_scenario[str(s)] = run_scenario(s)
        b = per_scenario[str(s)]["baseline"]
        s1 = per_scenario[str(s)]["step1"]
        print(f"  scenario {s}: baseline J {b['J']:.3f} | "
              f"cond rho0.2 dJ {s1['0.2']['delta_J_vs_v2']:+.1%} | "
              f"cond rho0.5 dJ {s1['0.5']['delta_J_vs_v2']:+.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # aggregate over scenarios (paired)
    def agg(field, mode):
        vals = []
        for s in per_scenario:
            if mode == "baseline":
                v = per_scenario[s]["baseline"][field]
            elif mode.startswith("s1"):
                rho = mode.split(":")[1]
                v = per_scenario[s]["step1"][rho][field]
            else:
                key = mode.split(":", 1)[1]
                v = per_scenario[s]["step2"][key][field]
            vals.append(v)
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    baseline_mean, baseline_sd = agg("J", "baseline")
    step1 = {}
    for rho_s in RHO_S_GRID:
        m, sd = agg("J", f"s1:{rho_s}")
        dJ = (m - baseline_mean) / max(baseline_mean, 1e-12)
        step1[str(rho_s)] = {"J_mean": m, "J_sd": sd,
                             "delta_J_vs_v2": float(dJ)}
    step2 = {}
    for world_rho in WORLD_RHO_GRID:
        if world_rho == 0.0:
            continue
        for rho_s in RHO_S_GRID:
            key = f"world{world_rho}_rho{rho_s}"
            m, sd = agg("J", f"s2:{key}")
            dJ = (m - baseline_mean) / max(baseline_mean, 1e-12)
            step2[key] = {"J_mean": m, "J_sd": sd,
                          "delta_J_vs_v2": float(dJ)}

    baseline = {"J": baseline_mean, "J_sd": baseline_sd}
    print(f"[G8B] aggregated over {args.scenario_seeds} scenarios: "
          f"baseline {baseline_mean:.3f} | rho0.5 dJ "
          f"{step1['0.5']['delta_J_vs_v2']:+.1%}", flush=True)

    # ---- gates (aggregated over scenarios) ---------------------------
    # Step 1 gate: no > 2% regression at rho_s in {0.2, 0.5}; errors ok
    p_md_s1 = {rho: float(max(per_scenario[s]["step1"][str(rho)]["p_md_max"]
                              for s in per_scenario))
               for rho in RHO_S_GRID}
    gate_step1 = {
        "delta_J_at_rho02": float(step1["0.2"]["delta_J_vs_v2"]),
        "delta_J_at_rho05": float(step1["0.5"]["delta_J_vs_v2"]),
        "p_md_max": p_md_s1,
        "errors_ok": bool(all(v <= args.beta + 0.02 for v in p_md_s1.values())),
        "improves": bool(step1["0.5"]["delta_J_vs_v2"] < -0.03),
        "pass": bool(
            step1["0.2"]["delta_J_vs_v2"] <= 0.02
            and step1["0.5"]["delta_J_vs_v2"] <= 0.02
            and all(v <= args.beta + 0.02 for v in p_md_s1.values())),
    }
    # Step 2 gate: the consistent conditional scheduler
    # (rho_s = world_rho) vs the singleton scheduler under the SAME
    # correlated world (rho_s = 0, world_rho > 0) -- the correlated-world
    # comparison the singleton aggregation over-counts.
    step2_gate = {}
    for world_rho in (0.2, 0.5):
        key_c = f"world{world_rho}_rho{world_rho}"
        key_s = f"world{world_rho}_rho0.0"
        dJs = [(per_scenario[s]["step2"][key_c]["J"]
                - per_scenario[s]["step2"][key_s]["J"])
               / max(per_scenario[s]["step2"][key_s]["J"], 1e-12)
               for s in per_scenario]
        step2_gate[str(world_rho)] = {
            "mean_dJ_cond_vs_singleton": float(np.mean(dJs)),
            "helps": bool(np.mean(dJs) < -0.01),
        }

    gate = {
        "step1_value_swap": gate_step1,
        "step2_correlated_world": step2_gate,
        "verdict": (
            "step1 passed with a measured improvement: the conditional "
            "value (rho_s = 0.5) cuts the worst-target delay by "
            f"{step1['0.5']['delta_J_vs_v2']:.1%} over the frozen "
            "singleton mainline -- the conditional-KL diversity value "
            "replaces the retired congestion price"
            if gate_step1["pass"] and gate_step1["improves"]
            else (
                "step1 passed: conditional value swap is safe (no "
                "regression)"
                if gate_step1["pass"]
                else "step1 failed: conditional value regresses the "
                     "mainline; do not adopt")),
    }

    payload = {
        "gate": "f0g8b-conditional-frids",
        "params": {
            "K": k, "Q": q, "n_runs": args.n_runs, "seeds": args.seeds,
            "max_steps": args.max_steps, "alpha": args.alpha,
            "beta": args.beta, "calib_seed": args.calib_seed,
            "calib_verify": args.calib_verify, "b_delta": args.b_delta,
            "rho_s_grid": list(RHO_S_GRID),
            "world_rho_grid": list(WORLD_RHO_GRID),
            "scenario_seeds": args.scenario_seeds,
            "value": "Delta G_{i|S,q} = G(S union {i}) - G(S), "
                     "S from received intents (strictly local)",
            "frozen": ["FRIDS-v2 mirror descent", "policy-matched B",
                       "compact token", "fixed owner", "full mesh",
                       "current scenario gen"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "baseline": baseline,
        "step1_value_swap": step1,
        "step2_correlated_world": step2,
        "per_scenario": per_scenario,
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()