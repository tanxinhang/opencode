"""F0-G6 tests: feasibility envelope machinery (advice/012)."""

import numpy as np
import pytest

from uav_otfs_isac.distributed_audit import (
    build_distributed_scenario,
    calibrate_target_bounds,
    token_bits,
)
from uav_otfs_isac.feasibility import (
    communication_load,
    f_submodular_oracle,
    fw_minimum_norm_point,
    rho_bruteforce,
    strongest_load_cut,
    submodular_minimize,
)
from uav_otfs_isac.frids import simulate_frids_v2


@pytest.fixture(scope="module")
def scenario():
    return build_distributed_scenario(np.random.default_rng(0),
                                      k_uavs=6, q_targets=3)


def test_rho_star_matches_bruteforce(scenario):
    """The strongest cut rho* via submodular minimization + binary
    search equals the exhaustive value at small Q (Bottleneck-Subset
    Feasibility Law)."""
    owner = scenario["owner_of"]
    rho_sm = strongest_load_cut(scenario, owner, horizon=40)
    rho_bf = rho_bruteforce(scenario, owner, horizon=40)
    assert rho_sm["rho_star"] == pytest.approx(rho_bf, abs=1e-3)
    assert rho_sm["feasible_info"] is True  # rho* < 1 in the family


def test_submodular_minimize_bruteforce(scenario):
    owner = scenario["owner_of"]
    from uav_otfs_isac.difficulty_decomposition import d_kl_binary
    info = d_kl_binary(0.95, 0.05)
    f = f_submodular_oracle(scenario, owner, 0.03, 40, info)
    S, val = submodular_minimize(f, 3)
    best = min(
        (f(frozenset(i for i in range(3) if mask & (1 << i))), mask)
        for mask in range(1, 8))
    assert val == pytest.approx(best[0], abs=1e-6)
    # S* is a nonempty subset (the empty set gives 0, not the min here)
    assert len(S) > 0


def test_fw_minimum_norm_point_basic(scenario):
    owner = scenario["owner_of"]
    from uav_otfs_isac.difficulty_decomposition import d_kl_binary
    f = f_submodular_oracle(scenario, owner, 0.03, 40,
                            d_kl_binary(0.95, 0.05))
    x = fw_minimum_norm_point(f, 3)
    assert len(x) == 3
    assert np.all(np.isfinite(x))


def test_communication_load():
    assert communication_load(16, 19, 400) == pytest.approx(0.7125)
    assert communication_load(16, 19, 100) > 1.0  # budget-limited regime


def test_cyclic_owner_for_q_gt_k():
    """Q > K needs a cyclic owner assignment (targets share owners)."""
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=8, q_targets=12)
    assert len(sc["owner_of"]) == 12
    assert all(0 <= o < 8 for o in sc["owner_of"])
    # the owner index is valid for every target
    assert sc["owner_of"][11] == 11 % 8


def test_token_cascade_layout():
    for q in (8, 12, 16, 24, 32):
        tb = token_bits(q)
        assert tb["total"] <= 19
        assert tb["q"] == int(np.ceil(np.log2(max(q, 2))))
        assert tb["intent"] == tb["q"]
        assert set(tb["dropped"]).issubset({"u", "r", "chi", "stamp"})


def test_frids_runs_at_q_gt_k():
    sc = build_distributed_scenario(np.random.default_rng(0),
                                    k_uavs=8, q_targets=12)
    bt = calibrate_target_bounds(sc, n_runs=40, seed=100,
                                 verify_runs=0)
    out = simulate_frids_v2(sc, bt, n_runs=40, seed=7, max_steps=40)
    assert 0.0 < out["worst_target_delay"] <= 40.0
    assert len(out["e1_delays"]) == 12
