"""Gate A: predictive value of post-communication detection information.

The gate asks whether the detection information of a reporting link --
the KL drift ``I^+ = KL(p1||p0)`` of the accumulated log-likelihood ratio
and the Chernoff information -- predicts *future* detection performance
(exact sequential P_D at 4/8 cycles and the cycle count to P_D* = 0.9)
better than the classical proxies (raw SNR, post-communication
deflection, single-observation P_D gain).

Each link is a scalar-Gaussian per-report evidence model through the
quantizer, the BSC, and a detectable-erasure channel, with the moments of
the physical chain (gamma moment matching, ``l_acc`` accumulations) as in
the system scenario builder.  Predictive value is scored by Spearman
correlation over randomized links; the gate reports all correlations and
fails the new detection-information line only if KL/Chernoff are clearly
dominated by the classical proxies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.detection_information import (
    cycles_to_pd,
    post_communication_likelihoods,
    sequential_pd,
)
from uav_otfs_isac.reporting import (
    post_bsc_moments,
    quantizer_from_gaussian_range,
)


def sample_link(rng: np.random.Generator, l_acc: int) -> dict:
    """One randomized reporting link with physically consistent moments."""
    snr_db = float(rng.uniform(-3.0, 8.0))
    noncentrality = l_acc * 10 ** (snr_db / 10.0)
    mu0 = float(l_acc)
    var0 = float(l_acc)
    mu1 = mu0 + noncentrality
    var1 = var0 + 2.0 * noncentrality
    bits = int(rng.integers(1, 6))
    flip = float(rng.uniform(0.01, 0.15))
    success = float(rng.uniform(0.5, 0.98))
    edges, values = quantizer_from_gaussian_range(
        np.array([mu0]), np.array([[var0]]),
        np.array([mu1]), np.array([[var1]]),
        bits,
    )
    info = post_communication_likelihoods(
        mu0, var0, mu1, var1, edges, values,
        bits, flip, success,
    )
    mu0_p, var0_p = post_bsc_moments(
        mu0, var0, edges, values, bits, flip,
    )
    mu1_p, _ = post_bsc_moments(mu1, var1, edges, values, bits, flip)
    return {
        "mu0": mu0, "var0": var0, "mu1": mu1, "var1": var1,
        "bits": bits, "flip": flip, "success": success,
        "snr_raw": (mu1 - mu0) ** 2 / var0,
        "deflection": (mu1_p - mu0_p) ** 2 / var0_p,
        "kl_plus": float(info["kl_plus"]),
        "kl_minus": float(info["kl_minus"]),
        "chernoff": float(info["chernoff"]),
        "kl_quant": float(info["kl_quant"]),
        "kl_sensing": float(info["kl_sensing"]),
        "p0_y": info["p0_y"],
        "p1_y": info["p1_y"],
        "contraction_holds": (
            info["kl_plus"] <= info["kl_quant"] + 1e-12
            and info["kl_quant"] <= info["kl_sensing"] + 1e-12
        ),
    }


def evaluate(link: dict, alpha: float, pd_star: float, max_n: int) -> dict:
    """Exact sequential outcomes for one link (i.i.d. repeated reports)."""
    p0_y, p1_y = link["p0_y"], link["p1_y"]
    pd1 = float(sequential_pd(p1_y, p0_y, 1, alpha)["pd"])
    pd4 = float(sequential_pd(p1_y, p0_y, 4, alpha)["pd"])
    pd8 = float(sequential_pd(p1_y, p0_y, 8, alpha)["pd"])
    nstar = cycles_to_pd(p1_y, p0_y, alpha, pd_star, max_n)
    return {
        "delta_pd1": pd1 - alpha,
        "pd4": pd4,
        "pd8": pd8,
        "nstar": nstar if nstar is not None else max_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/detection_information_gate.json")
    parser.add_argument("--instances", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l-acc", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pd-star", type=float, default=0.9)
    parser.add_argument("--max-n", type=int, default=16)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    predictors = ["snr_raw", "deflection", "delta_pd1", "kl_plus",
                  "chernoff", "kl_quant", "kl_sensing"]
    outcomes = ["pd4", "pd8", "inv_nstar"]
    rows = []
    contraction_failures = 0
    unreachable = 0
    for _ in range(args.instances):
        link = sample_link(rng, args.l_acc)
        ev = evaluate(link, args.alpha, args.pd_star, args.max_n)
        if not link["contraction_holds"]:
            contraction_failures += 1
        if ev["nstar"] >= args.max_n:
            unreachable += 1
        rows.append({p: link[p] for p in predictors
                     if p in link} | {
            "delta_pd1": ev["delta_pd1"],
            "pd4": ev["pd4"],
            "pd8": ev["pd8"],
            "inv_nstar": 1.0 / ev["nstar"],
            "nstar": ev["nstar"],
        })

    table = {}
    best = {}
    for outcome in outcomes:
        column = np.asarray([r[outcome] for r in rows])
        table[outcome] = {}
        for predictor in predictors:
            values = np.asarray([r[predictor] for r in rows])
            rho, pvalue = spearmanr(values, column)
            table[outcome][predictor] = {
                "spearman": float(rho),
                "pvalue": float(pvalue),
            }
        ranked = sorted(
            table[outcome].items(), key=lambda kv: -abs(kv[1]["spearman"])
        )
        best[outcome] = ranked[0][0]
        best[outcome + "_rho"] = ranked[0][1]["spearman"]

    info_best_inv = max(abs(table["inv_nstar"][p]["spearman"])
                        for p in ("kl_plus", "chernoff"))
    info_best_pd8 = max(abs(table["pd8"][p]["spearman"])
                        for p in ("kl_plus", "chernoff"))
    passed = (
        abs(best["inv_nstar_rho"]) - info_best_inv <= 0.02
        or abs(best["pd8_rho"]) - info_best_pd8 <= 0.02
    )

    payload = {
        "gate": "detection-information-predictive-value",
        "instances": args.instances,
        "seed": args.seed,
        "l_acc": args.l_acc,
        "alpha": args.alpha,
        "pd_star": args.pd_star,
        "max_n": args.max_n,
        "contraction_fraction": 1.0 - contraction_failures / args.instances,
        "unreachable_fraction": unreachable / args.instances,
        "best_predictor_per_outcome": best,
        "spearman_table": table,
        "passed": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = f"{'predictor':<12}" + "".join(f"{o:>12}" for o in outcomes)
    print(header)
    for predictor in predictors:
        line = f"{predictor:<12}"
        for outcome in outcomes:
            rho = table[outcome][predictor]["spearman"]
            mark = " *" if best[outcome] == predictor else "  "
            line += f"{rho:>10.3f}{mark}"
        print(line)
    print(f"contraction_fraction={payload['contraction_fraction']:.3f} "
          f"unreachable_fraction={payload['unreachable_fraction']:.3f}")
    print(f"passed={passed}")


if __name__ == "__main__":
    main()