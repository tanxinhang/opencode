"""Gate F0-D tests: target difficulty decomposition (advice/008)."""

import numpy as np
import pytest

from uav_otfs_isac.difficulty_decomposition import (
    d_kl_binary,
    difficulty_fingerprint,
    isolated_scenario,
    run_decomposition,
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
    return sc, bt, singles


def test_d_kl_binary_known_values():
    assert d_kl_binary(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)
    # d(1-beta || alpha) with alpha=beta=0.05
    v = d_kl_binary(0.95, 0.05)
    assert v == pytest.approx(
        0.95 * np.log(19.0) + 0.05 * np.log(0.05 / 0.95), rel=1e-9)
    assert v > 0.0


def test_isolated_scenario_preserves_realization(setup):
    sc, bt, singles = setup
    iso = isolated_scenario(scenario=sc, q=1)
    assert iso["k"] == sc["k"] and iso["q"] == 1
    assert iso["owner_of"] == [1]
    assert np.array_equal(iso["u2u_success"], sc["u2u_success"])
    # same kernels as target 1 of the full scenario
    for i in range(sc["k"]):
        full = sc["by_host"][(i, 1)]
        alone = iso["by_host"][(i, 0)]
        assert len(full) == len(alone)
        for fa, ia in zip(full, alone):
            assert np.array_equal(fa["llr"], ia["llr"])
            assert np.array_equal(fa["p0"], ia["p0"])
            assert np.array_equal(fa["p1"], ia["p1"])
            assert ia["target"] == 0
    # the original scenario kernels are not mutated
    assert sc["by_host"][(0, 1)][0]["target"] == 1


def test_difficulty_fingerprint(setup):
    sc, bt, singles = setup
    fp = difficulty_fingerprint(sc, 0)
    assert 0.0 < fp["i_plus_max"] < 10.0
    assert 0.0 < fp["chernoff_max"] < fp["i_plus_max"]
    assert 1 <= fp["n_useful_uavs"] <= sc["k"]
    assert fp["t_lb_per_obs"] > 0.0


def test_decomposition_structure_and_consistency(setup):
    sc, bt, singles = setup
    dec = run_decomposition(sc, bt, singles, n_runs=60, seeds=2)
    q = sc["q"]
    assert len(dec["j_iso"]) == q
    assert len(dec["j_cent"]) == q
    assert len(dec["j_dist"]) == q
    assert len(dec["j_fullmsg"]) == q
    assert dec["q_star"] == int(np.argmax(dec["j_dist"]))
    d = dec["iso_distribution"]
    assert d["median"] <= d["p90"] <= d["max"]
    h = dec["hardest"]
    s = h["shares"]
    assert abs(s["J_iso_share"] + s["comp_share"] + s["dec_share"]
               - 1.0) < 1e-6
    assert dec["case"].startswith("case_") or dec["case"] == "mixed"
    assert isinstance(dec["next_step"], str) and len(dec["next_step"]) > 0
    # dec split sums to the dec share on the hardest target (within MC)
    qq = h["q"]
    total = dec["dec_split"]["quantization_share"][qq] \
        + dec["dec_split"]["delivery_local_share"][qq]
    assert abs(total - s["dec_share"]) < 0.05


def test_decomposition_deterministic(setup):
    sc, bt, singles = setup
    a = run_decomposition(sc, bt, singles, n_runs=40, seeds=1)
    b = run_decomposition(sc, bt, singles, n_runs=40, seeds=1)
    assert a["j_dist"] == b["j_dist"]
    assert a["case"] == b["case"]
