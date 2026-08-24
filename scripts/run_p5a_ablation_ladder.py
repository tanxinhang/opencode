"""P5-A ablation ladder: mechanism attribution of the CA-FRIDS gain
(advice/017 section 13).

The reviewer attack is: "is the CA gain just owner routing?"  This gate
decomposes the gain into a minimal mechanism ladder on the REGISTERED
congested boundary cell (geom=2, rho=1.8) at the FROZEN calibrated
policy-B operating point (delta=1, same stopping thresholds for every
arm -- the fair scheduler-only / mechanism-only comparison):

    A   : FRIDS-v2  (reference; local task price, neutral admission)
    B00 : owner-only routing  (CA architecture, NO task price, NO
          receiver price, neutral admission)
    B0  : B00 + dynamic task price pi_q = y_q/(D_q+eps)
    B1  : B0  + receiver airtime price lambda_j
    C   : B1  + density admission   (= full CA-FRIDS)

    D_owner     = J_A   - J_B00   (architecture: owner-directed evidence)
    D_pi        = J_B00 - J_B0    (task-deficit coordination)
    D_lambda    = J_B0  - J_B1    (receiver-capacity steering)
    D_admission = J_B1  - J_C     (density admission)

J is the held-out matched-policy worst-target E[T|H1] (pooled).  Every
arm consumes the SAME held-out CRN exogen tape per block, so each
consecutive delta is a paired comparison with a per-block bootstrap CI.
QoS (anytime-valid, 32-stream familywise, exactly as P4.2b) is reported
per arm at the frozen operating point (arms may differ in QoS status --
that is itself part of the mechanism attribution).

Running at the FROZEN common point (not matched-QoS) is intentional:
the ladder asks WHY the architecture reaches the same evidence faster at
the SAME stopping policy, before any threshold re-tuning is admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_otfs_isac.airtime import build_airtime_model
from uav_otfs_isac.ca_frids import simulate_ca_frids
from uav_otfs_isac.crn_tape import build_exogenous_tape
from uav_otfs_isac.distributed_audit import (
    TOKEN_LLR_BITS,
    build_distributed_scenario,
    calibrate_target_bounds,
)
from uav_otfs_isac.frids import simulate_frids_v2
from uav_otfs_isac.qos import anytime_qos_status

SPEC = 0.05
SCALE = (16, 8)
N_STREAMS = 32


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            text=True).strip()
    except Exception:
        return "unknown"


def _git_tree() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=str(PROJECT_ROOT),
            text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True)
        return bool(out.strip())
    except Exception:
        return True


def _sha16(text) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _config_hash(args) -> str:
    fields = {
        "geom": args.geom, "rho": args.rho,
        "test_seed": args.test_seed, "test_cell_runs": args.test_cell_runs,
        "test_mc": args.test_mc,
        "max_steps": args.max_steps, "alpha": args.alpha, "beta": args.beta,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode,
        "calib_seed": args.calib_seed, "calib_verify": args.calib_verify,
        "calib_n_runs": args.calib_n_runs,
    }
    return _sha16(json.dumps(fields, sort_keys=True, indent=0))


def _empty_acc(q):
    return {key: [0] * q for key in ("n_H0", "n_H1", "n_FA", "n_MD")}


def _acc(out, acc):
    for key in ("n_H0", "n_H1", "n_FA", "n_MD"):
        acc[key] = [a + b for a, b in zip(acc[key], out["raw_counts"][key])]


def _paired_ci(v2_blocks, ca_blocks, n_boot=10000, seed=0):
    """Paired per-block bootstrap 95% CI of the mean difference
    ``J_v2 - J_ca`` (positive = v2 worse, the CA arm wins).  The two arms
    share the SAME held-out CRN block, so the difference is paired."""
    delta = np.asarray(v2_blocks, dtype=float) - np.asarray(ca_blocks, dtype=float)
    n = len(delta)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.mean(delta[idx]))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(delta)), float(ci_lo), float(ci_hi)


def _run_arm(runner, bounds, n_runs, seed, max_steps, exog_blocks):
    """Run ONE arm over ``test_mc`` held-out CRN blocks (all blocks share)
    the same frozen thresholds); return pooled J, block J, QoS status."""
    acc = _empty_acc(len(bounds))
    pool_s, pool_n = np.zeros(len(bounds)), np.zeros(len(bounds))
    block_J = []
    for mc, tape in enumerate(exog_blocks):
        out = runner(bounds, n_runs, seed + mc, tape)
        _acc(out, acc)
        pool_s += np.asarray(out["pool"]["sum_h1_delay"], dtype=float)
        pool_n += np.asarray(out["pool"]["n_h1"], dtype=float)
        b_s = np.asarray(out["pool"]["sum_h1_delay"], dtype=float)
        b_n = np.asarray(out["pool"]["n_h1"], dtype=float)
        block_J.append(float(np.max(b_s / np.maximum(b_n, 1.0))))
    J = float(np.max(pool_s / np.maximum(pool_n, 1.0)))
    qos = anytime_qos_status(
        acc["n_H0"], acc["n_H1"], acc["n_FA"], acc["n_MD"],
        0.05, 0.05, delta_fam=0.05, n_streams=N_STREAMS,
        ret_bounds=False)
    return {"J": J, "block_J": block_J, "qos": qos}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/p5a_ablation_ladder.json")
    parser.add_argument("--geom", type=int, default=2,
                        help="the registered congested boundary cell geometry")
    parser.add_argument("--rho", type=float, default=1.8,
                        help="congested rho_target of the registered cell")
    parser.add_argument("--test-seed", type=int, default=400000,
                        help="FRESH held-out seed namespace (disjoint from "
                             "P4.2 cert 100000, P4.1b discovery 7, threshold "
                             "calibration 100, P4.2b cal 200000 / test 300000)")
    parser.add_argument("--test-cell-runs", type=int, default=1500)
    parser.add_argument("--test-mc", type=int, default=8,
                        help="held-out blocks = test_cell_runs * test_mc")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=SPEC)
    parser.add_argument("--beta", type=float, default=SPEC)
    parser.add_argument("--pi-bits", type=int, default=10)
    parser.add_argument("--lam-bits", type=int, default=10)
    parser.add_argument("--price-mode", default="global_simplex",
                        choices=("global_simplex", "owner_local"))
    parser.add_argument("--calib-seed", type=int, default=100)
    parser.add_argument("--calib-verify", type=int, default=1000)
    parser.add_argument("--calib-n-runs", type=int, default=300)
    args = parser.parse_args()
    t0 = time.time()

    k, q = SCALE
    sc = build_distributed_scenario(np.random.default_rng(args.geom),
                                    k_uavs=k, q_targets=q)
    bt = calibrate_target_bounds(sc, args.alpha, args.beta,
                                 n_runs=args.calib_n_runs,
                                 seed=args.calib_seed,
                                 llr_bits=TOKEN_LLR_BITS,
                                 verify_runs=args.calib_verify)
    am = build_airtime_model(sc, rho_target=args.rho)
    bounds = [[bt[qq][0], bt[qq][1] - 1.0] for qq in range(q)]

    # shared held-out CRN exogen blocks (same tapes drive every arm)
    exog_blocks = [
        build_exogenous_tape(args.test_seed + mc, args.test_cell_runs,
                             q, k, args.max_steps)
        for mc in range(args.test_mc)
    ]

    # --- arm runners ----------------------------------------------------
    def run_v2(bounds, n_runs, seed, tape):
        return simulate_frids_v2(sc, bounds, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode="local", airtime=am, exog=tape)

    def run_b00(bounds, n_runs, seed, tape):
        # owner-only routing: CA index WITHOUT task price and WITHOUT
        # receiver price (flat pi), neutral admission -- the isolated
        # owner-directed evidence architecture.
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 task_price=False, airtime_price=False,
                                 admission_policy="neutral", exog=tape)

    def run_b0(bounds, n_runs, seed, tape):
        # B00 + dynamic task price pi_q = y_q/(D_q+eps); no receiver price.
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 task_price=True, airtime_price=False,
                                 admission_policy="neutral", exog=tape)

    def run_b1(bounds, n_runs, seed, tape):
        # B0 + receiver airtime price lambda_j; still neutral admission.
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 task_price=True, airtime_price=True,
                                 admission_policy="neutral", exog=tape)

    def run_c(bounds, n_runs, seed, tape):
        # full CA-FRIDS: + density admission.
        return simulate_ca_frids(sc, bounds, am, n_runs, seed=seed,
                                 max_steps=args.max_steps, raw_counts=True,
                                 price_mode=args.price_mode,
                                 pi_bits=args.pi_bits, lam_bits=args.lam_bits,
                                 task_price=True, airtime_price=True,
                                 admission_policy="density", exog=tape)

    arms = {
        "A": (run_v2, "FRIDS-v2 (reference)"),
        "B00": (run_b00, "owner-only routing"),
        "B0": (run_b0, "owner routing + task price"),
        "B1": (run_b1, "owner routing + task price + receiver price"),
        "C": (run_c, "full CA-FRIDS (+ density admission)"),
    }
    results = {}
    for name, (runner, label) in arms.items():
        results[name] = _run_arm(runner, bounds, args.test_cell_runs,
                                 args.test_seed, args.max_steps, exog_blocks)
        print(f"arm {name:>3} ({label}): J={results[name]['J']:.4f} "
              f"qos={results[name]['qos']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # --- consecutive deltas (paired per-block bootstrap 95% CI) ---------
    def _delta(prev, cur):
        """Delta verbatim sign-aware (audit/p5-a item 2): ``d = J_prev -
        J_cur`` -- POSITIVE means the ``cur`` arm IMPROVES worst-target
        delay (a gain mechanism), NEGATIVE means ``cur`` HURTS it (a loss
        mechanism, e.g. lambda at the frozen point).  The wording must not
        print ``J_cur < J_prev by -x`` (self-contradictory): a loss is
        written ``J_cur > J_prev by |x| (lambda HURTS worst-target delay)``."""
        if cur is None:
            return None
        d, lo, hi = _paired_ci(results[prev]["block_J"],
                               results[cur]["block_J"])
        base = float(np.mean(results[prev]["block_J"]))
        rel = d / max(base, 1e-12)
        sign = "improves worst-target delay" if d >= 0.0 \
            else "HURTS worst-target delay (a loss mechanism)"
        return {"point": d, "ci95": [lo, hi], "rel": rel,
                "is_gain": bool(d >= 0.0),
                "wording": (
                    f"J_{cur} {'<' if d >= 0 else '>'} J_{prev} by "
                    f"{abs(d):.4f} ({abs(rel) * 100.0:.1f}%; "
                    f"95% CI [{lo:.4f}, {hi:.4f}]; the arm {sign})")}

    deltas = {
        "D_owner": _delta("A", "B00"),
        "D_pi": _delta("B00", "B0"),
        "D_lambda": _delta("B0", "B1"),
        "D_admission": _delta("B1", "C"),
    }

    # cumulative mechanism decomposition
    cum = results["A"]["J"] - results["C"]["J"]
    ladder = {
        "A": results["A"]["J"],
        "B00": results["B00"]["J"],
        "B0": results["B0"]["J"],
        "B1": results["B1"]["J"],
        "C": results["C"]["J"],
        "cumulative_owner_to_full": float(cum),
"decomposition": {
            "D_owner": deltas["D_owner"]["point"],
            "D_pi": deltas["D_pi"]["point"],
            "D_lambda": deltas["D_lambda"]["point"],
            "D_admission": deltas["D_admission"]["point"],
        },
    }

    # mechanism-dominant verdict (advice/017 section 13): the SINGLE largest
    # POSITIVE consecutive delta names the mechanism that explains most of
    # the CA-FRIDS gain at the frozen operating point (audit/017 13 #1: a
    # NEGATIVE delta -- e.g. D_lambda here -- is a HARM, not a gain, so it
    # must never be selected as dominant).  The ladder picks the largest
    # ``is_gain=True`` delta; if NO positive delta exists the verdict is
    # "no dominant positive mechanism" instead of silently picking a loss.
    gains = {k: d for k, d in deltas.items() if d["is_gain"]}
    if gains:
        dom = max(gains, key=lambda k: gains[k]["point"])
        dominant = dom
        dominant_point = gains[dom]["point"]
        dominant_rel = gains[dom]["rel"]
        c_total = results["A"]["J"] - results["C"]["J"]
        dom_share = dominant_point / max(c_total, 1e-12)
        if dom == "D_owner":
            dom_note = ("The owner-directed EVIDENCE ARCHITECTURE is the "
                        "dominant positive mechanism -- architecturally the "
                        "gain does NOT come from the deficit price alone "
                        "(advice/017 section 13).")
        elif dom == "D_pi":
            dom_note = ("The DETECTION-DEFICIT task coordination (the "
                        "detection-deficit price pi_q = y_q/(D_q+eps)) is the "
                        "dominant positive mechanism -- a first-tier "
                        "ALGORITHM contribution (advice/017 section 13).")
        elif dom == "D_lambda":
            dom_note = ("THE RECEIVER-CAPACITY (airtime) price is the "
                        "dominant positive mechanism -- the congestion "
                        "price is what pays for the delay gain here.")
        else:
            dom_note = ("the density admission is the dominant positive "
                        "mechanism at this cell.")
    else:
        dominant = None
        dominant_point = 0.0
        dominant_rel = 0.0
        c_total = results["A"]["J"] - results["C"]["J"]
        dom_share = 0.0
        dom_note = ("NO positive mechanism dominates: every consecutive "
                    "delta is <= 0 at this cell (all mechanisms either "
                    "harm or are ~neutral on worst-target delay) -- the "
                    "advice/017 section 11 ladder verdict does NOT apply.")
    gate = {
        "objective": "held-out matched-policy worst-target E[T|H1] pooled",
        "arms": results,
        "deltas": deltas,
        "ladder": ladder,
        "dominant_mechanism": {
            "key": dominant, "point": dominant_point,
            "rel": dominant_rel,
            "share_of_total_gain": float(dom_share),
            "note": dom_note,
        },
        "interpretation": (
            "The largest positive delta names the mechanism that explains "
            "most of the CA-FRIDS gain at the frozen operating point.  "
            "Whichever of {D_owner, D_pi, D_lambda, D_admission} dominates "
            "is the mechanism to lead with in the paper -- if D_owner "
            "dominates it is the ARCHITECTURE (owner-directed evidence), "
            "if D_pi it is the detection-deficit task coordination, if "
            "D_lambda it is the receiver-capacity steering, if "
            "D_admission it is the density admission; the dominant share "
            "is relative to the total CA-vs-v2 gain (advice/017 section "
            "13)."),
        "caveats": [
            "The ladder runs at the FROZEN policy-B operating point "
            "(delta=1, shared by every arm), so it answers the "
            "mechanism-attribution question ONLY at that operating point.  "
            "It is NOT the matched-QoS frontier (P4.2b answers that) and "
            "NOT an anytime-valid QoS certificate (P4.2/registered do "
            "that): every arm QoS is EXPECTED to be UNCERTAIN here.",
            "D_lambda may be negative (the receiver airtime price steering "
            "can HURT worst-target E[T|H1] at the frozen point): per "
            "advice/017 section 7, lambda is registered as a "
            "feasibility/capacity-steering mechanism (and P4.1b found "
            "airtime value in the feasibility/comm-efficiency layer, not "
            "the delay layer), so a negative delay delta at one cell is "
            "the expected honest reading -- the paper must NOT claim "
            "lambda reduces stopping delay.",
            "D_owner is NOT a pure architecture-only ablation (audit "
            "finding, advice/017 section 13): arm A (FRIDS-v2) is the "
            "full-mesh local-replica broadcast with its OWN local deficit "
            "price, while arm B00 is the CA owner-directed evidence plane "
            "WITH the deficit price removed (flat index).  Delta_owner "
            "bundles two changes: (i) topology full-mesh -> owner-point "
            "routing, (ii) dropping v2's local deficit price.  So the "
            "honest reading is \\\"owner-directed evidence plane (deficit "
            "price removed)\\\", NOT \\\"owner routing alone\\\".  If "
            "D_owner dominates, the paper must say: owner-directed "
            "architecture AND the removal of v2's local-deficit steering "
            "together explain the gain -- do NOT claim pure routing.",
        ],
    }
    params = {
        "gate_id_base": "p5a-ablation-ladder",
        "scale": [k, q], "geom": args.geom, "regime": "congested",
        "rho": args.rho,
        "test_episodes": args.test_cell_runs * args.test_mc,
        "max_steps": args.max_steps, "alpha": args.alpha, "beta": args.beta,
        "pi_bits": args.pi_bits, "lam_bits": args.lam_bits,
        "price_mode": args.price_mode,
        "git_commit": _git_sha(), "git_dirty": _git_dirty(),
        "config_hash": _config_hash(args),
        "seed_scheme": (
            "FRESH held-out seed namespace 400000+mc shared by ALL arms "
            "(disjoint from P4.2 cert 100000, P4.1b discovery 7, threshold "
            "calibration 100, P4.2b cal 200000 / test 300000), so every "
            "consecutive delta is a genuine PAIRED comparison"),
        "frozen": ["geom=2 congested rho=1.8 (registered boundary cell)",
                   "FRIDS-v2 / CA-FRIDS schedulers unchanged",
                   "FIXED calibrated policy-B stopping thresholds "
                   "(delta=1, same for every arm -- scheduler-only / "
                   "mechanism-only comparison)",
                   "no scheduler / price / geometry / seed tuning"],
    }
    payload = {
        "gate_id": "p5a-ablation-ladder",
        "params": params,
        "metrics": gate,
        "runtime_s": round(time.time() - t0, 1),
        "provenance": {
            "git_commit": _git_sha(), "git_tree": _git_tree(),
            "git_dirty": _git_dirty(), "config_hash": _config_hash(args),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("gate:", json.dumps(gate, indent=1))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()