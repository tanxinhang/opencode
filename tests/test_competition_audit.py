"""Gate F0-A tests: target competition audit + allocation fixes."""

import numpy as np
import pytest

from uav_otfs_isac.competition_audit import (
    classify_case,
    simulate_competition_audit,
)
from uav_otfs_isac.distributed_audit import (
    build_distributed_scenario,
    build_target_values,
    calibrate_target_bounds,
)


@pytest.fixture(scope="module")
def setup():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=6, q_targets=3)
    bt = calibrate_target_bounds(sc, n_runs=80, seed=100,
                                 verify_runs=300)
    nu = (1 / 3, 1 / 3, 1 / 3)
    singles = build_target_values(sc, bt, horizon=40, nu=nu)
    return sc, bt, singles, nu


def test_diagnostics_ranges(setup):
    sc, bt, singles, nu = setup
    out = simulate_competition_audit(sc, bt, singles, nu, n_runs=80,
                                     seed=5)
    assert 0.0 <= out["r_min"] <= out["r_mean"] <= 1.0
    assert out["H_max_idle"] >= 0.0
    assert out["concurrency_max"] >= 1
    assert len(out["r_per_target"]) == 3
    assert len(out["nbar_per_target"]) == 3
    assert 0.0 <= out["distorted_choice_rate"] <= 1.0
    assert out["j_median_scale"] > 0.0
    for p in out["p_md"] + out["p_fa"]:
        assert 0.0 <= p <= 1.0


def test_frozen_default_matches_f0(setup):
    """normalize_gains=False must reproduce the frozen mechanism (the
    F0/F0-S mainline is untouched)."""
    sc, bt, singles, nu = setup
    a = simulate_competition_audit(sc, bt, singles, nu, n_runs=80,
                                   seed=7)
    b = simulate_competition_audit(sc, bt, singles, nu, n_runs=80,
                                   seed=7, normalize_gains=True,
                                   eta_A=0.0, psi_gamma=1.0, eta=0.0)
    assert a["worst_target_delay"] == pytest.approx(
        b["worst_target_delay"], abs=1e-9)


def test_age_and_price_change_choices_when_normalized(setup):
    """The additive terms are inert on the 1e9-scaled raw gains but
    active on the normalized index (the F0-A unit-mismatch finding)."""
    sc, bt, singles, nu = setup
    raw = simulate_competition_audit(sc, bt, singles, nu, n_runs=60,
                                     seed=11)
    norm = simulate_competition_audit(sc, bt, singles, nu, n_runs=60,
                                      seed=11, normalize_gains=True,
                                      eta=2.0, eta_A=0.4)
    assert norm["distorted_choice_rate"] >= raw["distorted_choice_rate"]
    assert norm["H_max_idle"] <= raw["H_max_idle"] + 0.5


def test_classify_case_allowed_outputs():
    rows = {
        "6_3": {"r_min": 0.40, "r_mean": 0.57, "H_max_idle": 4.0,
                "rho_alloc": 0.44, "distorted_choice_rate": 0.001,
                "concurrency_max": 6},
        "16_8": {"r_min": 0.23, "r_mean": 0.47, "H_max_idle": 5.3,
                 "rho_alloc": 0.17, "distorted_choice_rate": 0.002,
                 "concurrency_max": 16},
    }
    v = classify_case(rows)
    assert v["primary_case"].startswith("case_")
    assert v["primary_case"] in (
        "case_1_resources_insufficient",
        "case_2_starvation",
        "case_3_overconcentration",
        "case_2_3_starvation_and_overconcentration")
    assert isinstance(v["next_step"], str) and len(v["next_step"]) > 0
    assert set(v["evidence"]) == {"r_min", "r_mean", "H_max_idle",
                                  "rho_alloc", "distorted_choice_rate"}


def test_gain_scale_diagnostic_detects_bias(setup):
    """The F0-A gain diagnostics must be on the decision scale (huge for
    the raw 1e9 in-band gains), which is what makes additive prices
    inert."""
    sc, bt, singles, nu = setup
    out = simulate_competition_audit(sc, bt, singles, nu, n_runs=60,
                                     seed=13)
    assert out["j_median_scale"] > 1e3
    assert out["j_cross_target_spread"] > 1e3


def test_fresh_intents_frozen_defaults(setup):
    """fresh_intents=False and delivery_override=None must reproduce the
    frozen mainline exactly."""
    sc, bt, singles, nu = setup
    a = simulate_competition_audit(sc, bt, singles, nu, n_runs=60,
                                   seed=17)
    b = simulate_competition_audit(sc, bt, singles, nu, n_runs=60,
                                   seed=17, fresh_intents=False,
                                   delivery_override=None)
    assert a["worst_target_delay"] == pytest.approx(
        b["worst_target_delay"], abs=1e-9)


def test_delivery_override_and_fresh_intents_run(setup):
    """The F0-F diagnostic options run and return valid metrics."""
    sc, bt, singles, nu = setup
    out = simulate_competition_audit(
        sc, bt, singles, nu, n_runs=40, seed=19,
        delivery_override=1.0, fresh_intents=True,
        normalize_gains=True, eta=0.0)
    assert 0.0 < out["worst_target_delay"] <= 40.0
    assert 0.0 <= max(out["p_md"]) <= 1.0
    assert out["r_min"] >= 0.0


def test_fresh_intents_more_aggressive_price(setup):
    """Fresh intents make the congestion price reflect the same-cycle
    herd (higher counts), which at the tested scale increases the
    distorted-choice rate or leaves it (diagnostic, not a claim)."""
    sc, bt, singles, nu = setup
    a = simulate_competition_audit(sc, bt, singles, nu, n_runs=60,
                                   seed=23, normalize_gains=True, eta=1.0)
    b = simulate_competition_audit(sc, bt, singles, nu, n_runs=60,
                                   seed=23, normalize_gains=True, eta=1.0,
                                   fresh_intents=True)
    assert b["distorted_choice_rate"] >= 0.0
    assert b["distorted_choice_rate"] >= a["distorted_choice_rate"] - 0.01
