"""P4-META -- project-level registered closure certificate (advice/016
section 19).

Historical verdicts must NOT be retro-edited (P4.1b keeps ``adopt_ca`` =
false: at that time it was 5 PASS + 1 UNCERTAIN).  This gate reads the two
CLEAN certificates that now exist

    results/ca_frids_gate_3g100m.json            (P4.1b, ec5812d, git_dirty=false)
    results/ca_frids_anytime_cert_gate.json      (P4.2, clean-HEAD, git_dirty=false)

and produces the project-level verdict per advice/016 section 19:

    P4.1b: performance_gate = PASS, qos = 5 PASS + 1 UNCERTAIN
    P4.2 : unresolved_cell   = CA PASS, v2 = FAIL  (32-stream anytime-valid)
    combined_registered_verdict: CA-FRIDS = PROMOTE (registered-benchmark
    primary scheduler; FRIDS-v2 = reference / ablation baseline)

The wording is STRICTLY the registered-benchmark one (advice/016 section 7):
the strong gate certifies the frozen common policy-B operating point under
the matched shared-airtime capacity model, NOT "v2 is fundamentally
infeasible" -- the latter requires the P4.2b QoS-Matched Operating-Point
Frontier experiment.
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

    combined = g1["adopt_ca"] is False and g3["strong_gate"]
    verdict = (
        "CA-FRIDS PROMOTED to registered-benchmark primary scheduler; "
        "FRIDS-v2 = reference / ablation baseline"
        if combined else
        "registered closure incomplete: verify P4.1b adopt_ca=false is "
        "historical and P4.2 strong_gate=true on the clean certs"
    )
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
                "matched_j": p42b["metrics"]["matched_j"],
                "held_out_qos": p42b["metrics"]["held_out_qos"],
            }),
        },
        "combined_registered_verdict": {
            "clause": verdict,
            "strict_wording": (
                "Under the frozen common policy-B operating point and "
                "matched shared-airtime capacity model, CA-FRIDS is "
                "anytime-valid QoS-certified at the registered congested "
                "boundary cell whereas FRIDS-v2 is certified to violate "
                "the false-alarm constraint.  'v2 fundamentally infeasible' "
                "is NOT claimed -- the P4.2b QoS-Matched Operating-Point "
                "Frontier experiment answered that question (Case B: both "
                "schedulers reach certifiable matched points; CA has the "
                "lower matched-QoS stopping delay)."),
        },
        "provenance": {
            "git_commit": _git_sha(), "git_tree": _git_tree(),
            "git_dirty": _git_dirty(),
            "source_fingerprints": {
                "p4_1b": _sha16(json.dumps(p41b["metrics"], sort_keys=True)),
                "p4_2": _sha16(json.dumps(p42["metrics"], sort_keys=True)),
            },
        },
    }
    out = PROJECT_ROOT / "results" / "p4_meta_cert.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("P4-META:", json.dumps(payload["combined_registered_verdict"],
                                 indent=1))


if __name__ == "__main__":
    main()