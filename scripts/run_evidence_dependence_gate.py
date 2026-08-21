"""Gate F0-G8A: Evidence-dependence audit (advice/013 section 5-8).

Does NOT change FRIDS.  Introduces a controlled common scatterer /
clutter correlation coefficient ``rho_s in {0, 0.2, 0.5, 0.8}`` and
measures whether the singleton reliable-information aggregation
``sum_{i in S} g_iq`` overestimates the joint detection information
``G_q(S)``, with the redundancy ratio

    R_q(S) = 1 - G_q(S) / sum_{i in S} g_iq,

and the KL-chain-rule conditional marginal ``Delta G_{i|S,q}`` (the
G8-B scheduling quantity).  The delay consequence is reported both
analytically (the full-coalition drift drops to ``G_q(S)``, so the
singleton-optimistic delay is understated by ``1/(1-R)``) and by a
Gaussian sequential test on the correlated stream.

Life gate (advice/013 section 8):

- if ``R_q < 5%`` and the delay consequence is negligible -> close the
  correlation direction (FRIDS's singleton aggregation is fine);
- if ``R_q > 10-20%`` with a clear delay penalty -> start the
  conditional-information FRIDS (G8-B).

Writes ``results/evidence_dependence_gate.json``.
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

from uav_otfs_isac.distributed_audit import build_distributed_scenario
from uav_otfs_isac.evidence_correlation import (
    alignment,
    build_delta_from_scenario,
    conditional_marginal,
    joint_kl_subset,
    redundancy,
    sequential_correlation_check,
)

RHO_GRID = (0.0, 0.2, 0.5, 0.8)
TOPK = 4   # serving coalition used for the sequential check


def topk_indices(delta: np.ndarray, q: int, k: int = TOPK) -> list:
    d = delta[:, q]
    return sorted(np.argsort(-d)[:k].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/evidence_dependence_gate.json")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--n-runs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seq-seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    k, q = args.k, args.q
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=k, q_targets=q)
    owner = sc["owner_of"]
    delta = build_delta_from_scenario(sc, owner)
    g_mat = np.array([[max(0.0, float(delta[i, qq]) ** 2 / 2.0)
                       for qq in range(q)] for i in range(k)])
    full = list(range(k))

    per_target = {}
    for qq in range(q):
        row = {}
        for rho in RHO_GRID:
            R = redundancy(delta[:, qq], full, rho)
            G = joint_kl_subset(delta[:, qq], full, rho)
            topk = topk_indices(delta, qq)
            R_topk = redundancy(delta[:, qq], topk, rho)
            row[str(rho)] = {
                "rho_s": float(rho),
                "g_sum": float(np.sum(g_mat[:, qq])),
                "G_joint_full": float(G),
                "R_full": float(R),
                "R_topk": float(R_topk),
                "alignment": float(alignment(delta[:, qq])),
                "delay_ratio_analytic": float(1.0 / max(1.0 - R, 1e-9)),
            }
        # conditional marginals at the highest correlation (G8-B preview)
        cond = []
        topk = topk_indices(delta, qq)
        for i in range(k):
            others = [j for j in full if j != i]
            dm = conditional_marginal(delta[:, qq], others, i, 0.8)
            cond.append({
                "uav": i,
                "singleton": float(g_mat[i, qq]),
                "conditional_given_rest": float(dm),
                "ratio": float(dm / max(g_mat[i, qq], 1e-12)),
            })
        cond.sort(key=lambda c: -c["singleton"])
        per_target[str(qq)] = {
            "g_sum": float(np.sum(g_mat[:, qq])),
            "g_profile": [float(x) for x in g_mat[:, qq]],
            "weak": bool(qq == 0),
            "correlation": row,
            "conditional_marginals_top8": cond[:8],
        }
        print(f"  q{qq}: R_full at rho=0.8 "
              f"{row['0.8']['R_full']:.3f} (topk "
              f"{row['0.8']['R_topk']:.3f}), delay ratio "
              f"{row['0.8']['delay_ratio_analytic']:.2f}, alignment "
              f"{row['0.8']['alignment']:.2f}", flush=True)

    # sequential check on the target with the most total reliable info
    worst_q = int(max(range(q), key=lambda qq: per_target[str(qq)]["g_sum"]))
    # the weak target is target 0 by construction; also report the
    # target with the largest full-set redundancy at rho=0.8
    maxred_q = int(max(range(q), key=lambda qq: per_target[str(qq)]
                       ["correlation"]["0.8"]["R_full"]))
    seq = {}
    for rho in RHO_GRID:
        coalition = topk_indices(delta, worst_q)
        out = sequential_correlation_check(
            delta[coalition, worst_q], rho, args.alpha, args.beta,
            n_runs=args.n_runs, max_steps=args.max_steps,
            seed=args.seq_seed)
        seq[str(rho)] = out
        print(f"  seq q{worst_q} rho={rho}: E1 {out['E1_T_independent']:.1f}"
              f" -> {out['E1_T_correlated']:.1f} (ratio "
              f"{out['delay_ratio_measured']:.2f}), drift "
              f"{out['independent_drift']:.3f} -> {out['correlated_drift']:.3f}",
              flush=True)

    # ---- life gate ---------------------------------------------------
    # the physical correlation regime is rho in {0.2, 0.5}; at rho = 0.8
    # the Gaussian noise-whitening of the private (orthogonal) components
    # can bend the redundancy back down for heterogeneous coalitions, so
    # the verdict is based on the physical regime (documented in the
    # FORMAL_PROOFS 5D non-claims).
    r_phys = {
        rho: {qq: per_target[str(qq)]["correlation"][str(rho)]["R_full"]
              for qq in range(q)} for rho in (0.2, 0.5)}
    max_r_phys = float(max(max(vals.values()) for vals in r_phys.values()))
    r8 = {qq: per_target[str(qq)]["correlation"]["0.8"]["R_full"]
          for qq in range(q)}
    max_r8 = float(max(r8.values()))
    seq_ratio = seq["0.8"]["delay_ratio_measured"]
    worst_ratio = float(1.0 / max(1.0 - max_r_phys, 1e-9))

    close_direction = bool(max_r_phys < 0.05 and worst_ratio < 1.05)
    start_g8b = bool(max_r_phys > 0.15 and worst_ratio > 1.10)
    gate = {
        "max_R_q_phys": float(max_r_phys),
        "max_R_q_rho08": float(max_r8),
        "argmax_R_phys": int(max(range(q), key=lambda qq: max(
            per_target[str(qq)]["correlation"]["0.2"]["R_full"],
            per_target[str(qq)]["correlation"]["0.5"]["R_full"]))),
        "delay_ratio_analytic_worst": float(worst_ratio),
        "delay_ratio_sequential_topk": float(seq_ratio),
        "close_direction": close_direction,
        "start_g8b": start_g8b,
        "verdict": (
            "close the correlation direction: singleton aggregation is "
            "fine (R < 5%)"
            if close_direction
            else (
                "correlation is significant; start the conditional-"
                "information FRIDS (G8-B)"
                if start_g8b
                else "intermediate: report the redundancy, no hard "
                     "verdict")),
    }

    payload = {
        "gate": "f0g8a-evidence-dependence-audit",
        "params": {
            "K": k, "Q": q, "alpha": args.alpha, "beta": args.beta,
            "rho_grid": list(RHO_GRID), "topk": TOPK,
            "n_runs": args.n_runs, "max_steps": args.max_steps,
            "seq_seed": args.seq_seed,
            "model": "Gaussian common factor: "
                     "Y_i = delta_i H1 + sqrt(rho_s) C + sqrt(1-rho_s) N_i; "
                     "delta_i = sqrt(2 * g_reliable_iq)",
            "frozen": ["FRIDS-v2 untouched", "fixed owner", "full mesh",
                       "current scenario gen (seed 0)"],
        },
        "runtime_s": round(time.time() - t0, 1),
        "per_target": per_target,
        "sequential": seq,
        "worst_q": int(worst_q),
        "maxred_q": int(maxred_q),
        "gate": gate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()