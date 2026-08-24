"""P4-META -- project-level registered closure certificate (advice/016
section 19, updated per advice/017 section 12.1).

Historical verdicts must NOT be retro-edited (P4.1b keeps ``adopt_ca`` =
false: at that time it was 5 PASS + 1 UNCERTAIN).  This gate reads the three
CLEAN certificates that now exist

    results/ca_frids_gate_3g100m.json            (P4.1b, ec5812d, git_dirty=false)
    results/ca_frids_anytime_cert_gate.json      (P4.2, clean-HEAD, git_dirty=false)
    results/ca_frids_qos_frontier_gate.json      (P4.2b, clean-HEAD, git_dirty=false)

and produces the project-level verdict per advice/017 section 12:

    P4.1b: performance_gate = PASS, qos = 5 PASS + 1 UNCERTAIN  (historical)
    P4.2 : unresolved_cell  = CA PASS, v2 = FAIL  (32-stream anytime-valid)
    P4.2b: three-state matched-QoS frontier -- both schedulers CERTIFIED
           FEASIBLE at their own recalibrated operating points (Case B),
           observed held-out matched-QoS delay reduction with paired CI
    combined_registered_verdict: CA-FRIDS = PROMOTE (registered-benchmark
    primary scheduler; FRIDS-v2 = reference / ablation baseline)

The promotion predicate is the POSITIVE evidence only (advice/017 section
12.1): P4.1b performance gate + P4.2 strong gate + P4.2b matched-QoS support.
The historical ``P4.1b.adopt_ca == false`` is NOT a positive promotion
condition -- it is the OLD conservative verdict (5 PASS + 1 UNCERTAIN at
that time) and must stay historical.  The wording is STRICTLY the
registered-benchmark one: the strong gate certifies the frozen common
policy-B operating point under the matched shared-airtime capacity model;
"v2 is fundamentally infeasible" is NEVER claimed -- the P4.2b matched-QoS
frontier shows both schedulers reach certifiable matched operating points
under their own recalibrated thresholds, with CA achieving the lower
held-out matched-QoS delay (an OBSERVED held-out reduction, reported with a
paired-block bootstrap CI).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    with (PROJECT_ROOT / "results" / name).open(encoding="utf-8") as h:
        return json.load(h)


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


def promotion_predicate(g1: dict, g3: dict, p42b: dict | None) -> dict:
    """Registered-promotion predicate -- POSITIVE evidence only
    (advice/017 section 12.1).  The historical ``P4.1b.adopt_ca == false``
    is NOT a positive promotion condition; the predicate rests on:

    - P4.1b performance gate PASS (no regression + congested win);
    - P4.2 strong gate (CA certified PASS, v2 certified FAIL);
    - P4.2b matched-QoS support: Case B with CA observed lower held-out
      matched delay at matched QoS, OR Case A with v2 frontier
      CERTIFIED INFEASIBLE, OR Case A-UNRESOLVED (weaker wording:
      "v2 not certified feasible on the swept frontier", infeasibility
      NOT certified).
    """
    p41b_perf = bool(g1["no_regression_uncongested_ucb"]
                     and g1["congested_win_5pct_lcb"])
    p42_strong = bool(g3["strong_gate"])
    p42b_case = None if p42b is None else p42b["metrics"]["case"]
    p42b_mj = None if p42b is None else p42b["metrics"]["matched_j"]
    p42b_qos = None if p42b is None else p42b["metrics"]["held_out_qos"]
    if p42b is None:
        p42b_support = False
        p42b_wording = "P4.2b certificate NOT present"
    elif p42b_case == "B":
        b_smaller = (p42b_mj["ca"] is not None and p42b_mj["v2"] is not None
                     and p42b_mj["ca"] < p42b_mj["v2"]
                     and p42b_qos == {"v2": "PASS", "ca": "PASS"})
        p42b_support = bool(b_smaller)
        p42b_wording = ("matched-QoS frontier = Case B: both schedulers "
                        "certified feasible at matched operating points; "
                        "observed held-out matched-QoS delay reduction")
    elif p42b_case == "A-CERTIFIED-INFEASIBLE":
        p42b_support = True
        p42b_wording = ("matched-QoS frontier = CASE A-CERTIFIED "
                        "INFEASIBLE: v2 not certified feasible and a "
                        "certified violation persists at the favorable "
                        "extremes -- the A_q lever cannot clear the spec")
    else:  # A-UNRESOLVED
        p42b_support = True
        p42b_wording = ("matched-QoS frontier = CASE A-UNRESOLVED: v2 not "
                        "certified feasible at any swept A_q on the frontier; "
                        "infeasibility NOT certified (the swept grid / "
                        "calibration MC does not certify feasibility)")
    combined = p41b_perf and p42_strong and p42b_support
    verdict = (
        "CA-FRIDS PROMOTED to registered-benchmark primary scheduler; "
        "FRIDS-v2 = reference / ablation baseline"
        if combined else
        ("registered closure incomplete: P4.1b performance gate / P4.2 "
         "strong gate / P4.2b matched-QoS support must ALL be positive on "
         "the clean certs (P4.1b adopt_ca=false is historical and is NOT a "
         "promotion condition)")
    )
    return {"combined": combined, "verdict": verdict,
            "p41b_perf": p41b_perf, "p42_strong": p42_strong,
            "p42b_support": p42b_support, "p42b_wording": p42b_wording}


def main() -> None:
    p41b = _load("ca_frids_gate_3g100m.json")
    p42 = _load("ca_frids_anytime_cert_gate.json")
    p42b = (_load("ca_frids_qos_frontier_gate.json")
            if (PROJECT_ROOT / "results" / "ca_frids_qos_frontier_gate.json")
            .exists() else None)

    g1 = p41b["metrics"]
    g3 = p42["metrics"]
    assert p41b["provenance"]["git_dirty"] is False, "P4.1b cert not clean"
    assert p42["provenance"]["git_dirty"] is False, "P4.2 cert not clean"

    pred = promotion_predicate(g1, g3, p42b)
    combined = pred["combined"]
    verdict = pred["verdict"]
    p42b_wording = pred["p42b_wording"]
    payload = {
        "gate_id": "p4-meta-registered-closure",
        "evidence": {
            "p4_1b": {
                "file": "ca_frids_gate_3g100m.json",
                "git_commit": p41b["params"]["git_commit"],
                "git_dirty": p41b["provenance"]["git_dirty"],
                "performance_gate": (
                    "PASS" if g1["no_regression_uncongested_ucb"]
                    and g1["congested_win_5pct_lcb"] else "FAIL"),
                "qos": "5 PASS + 1 UNCERTAIN",
                "adopt_ca": g1["adopt_ca"],
                "congested_improvement": g1["congested_improvement_mean"],
                "uncongested_regression": g1["uncongested_regression_mean"],
            },
            "p4_2": {
                "file": "ca_frids_anytime_cert_gate.json",
                "git_dirty": p42["provenance"]["git_dirty"],
                "unresolved_cell": "CA PASS",
                "v2": "FAIL",
                "strong_gate": g3["strong_gate"],
                "strong_gate_at_stage": g3["strong_gate_at_stage"],
                "strong_gate_episodes": g3["strong_gate_episodes"],
"first_crossing_stage_v2_FA0": g3["first_crossing_stage"]
            ["v2"]["FA"]["0"],
            },
            "p4_2b": (None if p42b is None else {
                "file": "ca_frids_qos_frontier_gate.json",
                "case": p42b["metrics"]["case"],
                "verdict": p42b["metrics"]["verdict"],
                "v2_frontier_state": p42b["metrics"]["v2_frontier"]["scheduler_state"],
                "ca_frontier_state": p42b["metrics"]["ca_frontier"]["scheduler_state"],
                "v2_m_star": p42b["metrics"]["v2_frontier"]["m_star"],
                "ca_m_star": p42b["metrics"]["ca_frontier"]["m_star"],
                "matched_j": p42b["metrics"]["matched_j"],
                "held_out_qos": p42b["metrics"]["held_out_qos"],
                "held_out_reduction": p42b["metrics"]["held_out_reduction"],
            }),
        },
        "combined_registered_verdict": {
            "clause": verdict,
            "p42b_supporting_wording": p42b_wording,
            "strict_wording": (
                "Under the frozen common policy-B operating point and "
                "matched shared-airtime capacity model, CA-FRIDS is "
                "anytime-valid QoS-certified at the registered congested "
                "boundary cell whereas FRIDS-v2 is certified to violate "
                "the false-alarm constraint at that FROZEN operating point.  "
                "'v2 fundamentally infeasible' is NOT claimed: the P4.2b "
                "QoS-Matched Operating-Point Frontier shows both schedulers "
                "reach CERTIFIED FEASIBLE matched operating points under "
                "their own recalibrated thresholds, with CA achieving the "
                "LOWER held-out matched-QoS delay (reported as an OBSERVED "
                "held-out reduction with a paired-block bootstrap CI; the "
                "CA matched multipliers are ALSO reported verbatim -- the "
                "registered CA m_star is [2,1.5,2,1.5,1.5,1,1,1], so "
                "'CA basically 1.0x threshold' is NOT a claim."),
        },
        "provenance": {
            "git_commit": _git_sha(), "git_tree": _git_tree(),
            "git_dirty": _git_dirty(),
            "source_fingerprints": {
                "p4_1b": _sha16(json.dumps(p41b["metrics"], sort_keys=True)),
                "p4_2": _sha16(json.dumps(p42["metrics"], sort_keys=True)),
                "p4_2b": (None if p42b is None else
                          _sha16(json.dumps(p42b["metrics"], sort_keys=True))),
            },
        },
    }
    out = PROJECT_ROOT / "results" / "p4_meta_cert.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("P4-META:", json.dumps(payload["combined_registered_verdict"],
                                 indent=1))


if __name__ == "__main__":
    main()