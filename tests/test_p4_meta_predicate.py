"""P4-META promotion-predicate tests (advice/017 section 12.1).

The registered promotion predicate ``promotion_predicate`` must rest on
POSITIVE evidence only:

- P4.1b performance gate PASS (no regression + congested win);
- P4.2 strong gate (CA PASS, v2 FAIL);
- P4.2b matched-QoS support (Case B with CA lower, or Case A-certified
  v2-infefeasible, or Case A-unresolved with the weaker wording).

The historical ``P4.1b.adopt_ca == false`` must NEVER be used as a positive
promotion condition (it was the OLD conservative verdict: 5 PASS + 1
UNCERTAIN at that time).
"""

from scripts.run_p4_meta_cert import promotion_predicate


def _g1_pass():
    return {"no_regression_uncongested_ucb": True,
            "congested_win_5pct_lcb": True}


def _g1_fail():
    return {"no_regression_uncongested_ucb": False,
            "congested_win_5pct_lcb": True}


def _g3_pass():
    return {"strong_gate": True}


def _g3_fail():
    return {"strong_gate": False}


def _p42b_case_b_lower():
    return {"metrics": {"case": "B",
                        "matched_j": {"ca": 4.5, "v2": 7.4},
                        "held_out_qos": {"v2": "PASS", "ca": "PASS"}}}


def _p42b_case_a_certified():
    return {"metrics": {"case": "A-CERTIFIED-INFEASIBLE",
                        "matched_j": {"ca": 4.5, "v2": None},
                        "held_out_qos": {"v2": None, "ca": "PASS"}}}


def _p42b_case_a_unresolved():
    return {"metrics": {"case": "A-UNRESOLVED",
                        "matched_j": {"ca": 4.5, "v2": None},
                        "held_out_qos": {"v2": None, "ca": "PASS"}}}


def _p42b_case_b_v2_lower():
    return {"metrics": {"case": "B",
                        "matched_j": {"ca": 7.4, "v2": 4.5},
                        "held_out_qos": {"v2": "PASS", "ca": "PASS"}}}


def test_promoted_on_three_positive_conditions():
    # P4.1b perf PASS + P4.2 strong PASS + P4.2b Case B (CA lower) -> PROMOTE.
    pred = promotion_predicate(_g1_pass(), _g3_pass(),
                               _p42b_case_b_lower())
    assert pred["combined"] is True
    assert "PROMOTED" in pred["verdict"]


def test_promotion_does_not_require_historical_adopt_ca_false():
    # The old predicate was ``g1["adopt_ca"] is False and strong_gate``, i.e.
    # it used the HISTORICAL conservative verdict as a positive condition.
    # The new predicate uses the P4.1b performance metrics, NOT ``adopt_ca``:
    # a dict WITHOUT the historical ``adopt_ca`` key must still promote when
    # the three positive conditions all hold.
    g1 = _g1_pass()
    assert "adopt_ca" not in g1
    pred = promotion_predicate(g1, _g3_pass(), _p42b_case_b_lower())
    assert pred["combined"] is True


def test_promotion_rejected_when_p41b_perf_fails():
    pred = promotion_predicate(_g1_fail(), _g3_pass(),
                               _p42b_case_b_lower())
    assert pred["combined"] is False
    assert "incomplete" in pred["verdict"]


def test_promotion_rejected_when_p42_strong_fails():
    pred = promotion_predicate(_g1_pass(), _g3_fail(),
                               _p42b_case_b_lower())
    assert pred["combined"] is False


def test_promotion_rejected_when_p42b_missing():
    pred = promotion_predicate(_g1_pass(), _g3_pass(), None)
    assert pred["combined"] is False
    assert pred["p42b_wording"] == "P4.2b certificate NOT present"


def test_promotion_rejected_when_case_b_v2_lower():
    # Case B but v2 lower matched delay -> CA lower condition fails.
    pred = promotion_predicate(_g1_pass(), _g3_pass(),
                               _p42b_case_b_v2_lower())
    assert pred["combined"] is False


def test_case_a_certified_supports_promotion():
    pred = promotion_predicate(_g1_pass(), _g3_pass(),
                               _p42b_case_a_certified())
    assert pred["combined"] is True
    assert "CERTIFIED INFEASIBLE" in pred["p42b_wording"]


def test_case_a_unresolved_supports_and_words_weakly():
    # A-UNRESOLVED is still support, but the wording must say v2 is
    # "not certified feasible", NOT "infeasible".
    pred = promotion_predicate(_g1_pass(), _g3_pass(),
                               _p42b_case_a_unresolved())
    assert pred["combined"] is True
    assert "not certified feasible" in pred["p42b_wording"]
    assert "INFEASIBLE" not in pred["p42b_wording"]


def test_historical_adopt_ca_false_still_flagged_as_not_condition():
    # Regression guard: even IF the P4.1b metrics still carry adopt_ca=false
    # (the historical value), the predicate must NOT use it -- the promotion
    # rests on the performance metrics only.
    g1 = _g1_pass()
    g1["adopt_ca"] = False
    pred = promotion_predicate(g1, _g3_pass(), _p42b_case_b_lower())
    assert pred["combined"] is True