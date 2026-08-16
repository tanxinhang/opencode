import numpy as np

from uav_otfs_isac.robust_joint_power_bit import (
    communication_pd_with_upper_bound,
    count_conditional_upper_bound,
    erasure_deterministic_upper_bound,
    hoeffding_upper_bound,
    per_report_communication_target_pd,
)


def _instance(seed=0, reports=5):
    rng = np.random.default_rng(seed)
    return (
        float(rng.uniform(0.3, 1.0)),
        rng.uniform(0.5, 2.0, reports),
        rng.uniform(0.0, 2.0, reports),
        rng.integers(1, 3, reports),
        rng.uniform(0.01, 0.05, reports),
        rng.uniform(0.7, 0.95, reports),
    )


def test_hoeffding_bound_structure():
    ub = hoeffding_upper_bound(0.7, 2048, delta=0.01)
    assert ub > 0.7
    assert abs(ub - 0.7) < 0.05
    tighter = hoeffding_upper_bound(0.7, 8192, delta=0.01)
    assert tighter < ub


def test_hoeffding_bound_requires_positive_samples():
    try:
        hoeffding_upper_bound(0.5, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_deterministic_upper_bound_dominates_exact_expectation():
    # R 小 → 精确枚举分支，确定性上界必须 ≥ 精确期望
    for seed in range(4):
        owner, deltas, powers, bits, flips, successes = _instance(seed, reports=4)
        exact = per_report_communication_target_pd(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8,
        )
        ub = erasure_deterministic_upper_bound(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8,
        )
        assert ub >= exact - 1e-9, (ub, exact)


def test_deterministic_upper_bound_dominates_mc_estimate():
    # R 大 → MC 分支，确定性上界仍必须 ≥ 估计值
    for seed in range(3):
        owner, deltas, powers, bits, flips, successes = _instance(seed, reports=10)
        mc = per_report_communication_target_pd(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8, samples=512,
        )
        ub = erasure_deterministic_upper_bound(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8, samples=512,
        )
        assert ub >= mc - 1e-6, (ub, mc)


def test_communication_pd_with_upper_bound_mc_branch():
    owner, deltas, powers, bits, flips, successes = _instance(seed=9, reports=10)
    result = communication_pd_with_upper_bound(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8, samples=512,
    )
    assert result["deterministic_ub"] >= result["estimate"] - 1e-6
    assert result["hoeffding_ub"] >= result["estimate"]
    assert result["deterministic_ub"] <= 1.0 + 1e-12
    assert 0.0 <= result["estimate"] <= 1.0


def test_communication_pd_with_upper_bound_exact_branch():
    owner, deltas, powers, bits, flips, successes = _instance(seed=4, reports=4)
    result = communication_pd_with_upper_bound(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8, samples=2048,
    )
    assert result["deterministic_ub"] >= result["estimate"] - 1e-9
    assert result["hoeffding_ub"] >= result["estimate"]


def test_deterministic_upper_bound_is_no_erasure_value():
    owner, deltas, powers, bits, flips, successes = _instance(seed=2, reports=5)
    ub = erasure_deterministic_upper_bound(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8,
    )
    no_erasure = per_report_communication_target_pd(
        owner, deltas, powers, bits, flips,
        np.ones_like(successes),
        grid=32, max_exact_reports=8,
    )
    assert abs(ub - no_erasure) < 1e-12


def _heterogeneous_instance(seed=0, reports=4):
    rng = np.random.default_rng(seed)
    return (
        float(rng.uniform(0.3, 1.0)),
        rng.uniform(0.3, 2.0, reports),
        rng.uniform(0.0, 2.0, reports),
        rng.integers(1, 3, reports),
        rng.uniform(0.01, 0.05, reports),
        rng.uniform(0.3, 0.9, reports),
    )


def test_count_conditional_bound_valid_and_tighter_exact_branch():
    # R 小 → 精确枚举: 界必须夹在精确期望与 no-erasure 值之间, 且
    # erasure 非平凡时严格更紧.
    for seed in range(4):
        owner, deltas, powers, bits, flips, successes = _instance(seed, reports=4)
        exact = per_report_communication_target_pd(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8,
        )
        det_ub = erasure_deterministic_upper_bound(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8,
        )
        cc_ub = count_conditional_upper_bound(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8,
        )
        assert cc_ub >= exact - 1e-9, (cc_ub, exact)
        assert cc_ub <= det_ub + 1e-12, (cc_ub, det_ub)
        if float(successes.min()) < 1.0 and float(successes.max()) < 1.0:
            assert cc_ub < det_ub - 1e-6, (cc_ub, det_ub)


def test_count_conditional_bound_valid_mc_branch():
    # R 大 → MC 分支: 界仍须支配估计, 且不劣于 no-erasure 值.
    for seed in range(3):
        owner, deltas, powers, bits, flips, successes = _instance(seed, reports=10)
        mc = per_report_communication_target_pd(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8, samples=512,
        )
        det_ub = erasure_deterministic_upper_bound(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8, samples=512,
        )
        cc_ub = count_conditional_upper_bound(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=8, samples=512,
        )
        assert cc_ub >= mc - 1e-6, (cc_ub, mc)
        assert cc_ub <= det_ub + 1e-12, (cc_ub, det_ub)


def test_count_conditional_bound_heterogeneous_success():
    # 每报告 erasure 不同: 泊松-二项计数律必须与精确枚举一致.
    owner, deltas, powers, bits, flips, successes = _heterogeneous_instance(seed=7)
    exact = per_report_communication_target_pd(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8,
    )
    det_ub = erasure_deterministic_upper_bound(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8,
    )
    cc_ub = count_conditional_upper_bound(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8,
    )
    assert cc_ub >= exact - 1e-9
    assert cc_ub <= det_ub + 1e-12
    assert cc_ub < det_ub - 1e-6


def test_count_conditional_bound_no_erasure_equals_deterministic():
    # success=1 → 计数律退化为 n=R, 界必须等于 no-erasure 值.
    owner, deltas, powers, bits, flips, successes = _instance(seed=2, reports=5)
    det_ub = erasure_deterministic_upper_bound(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8,
    )
    cc_ub = count_conditional_upper_bound(
        owner, deltas, powers, bits, flips,
        np.ones_like(successes),
        grid=32, max_exact_reports=8,
    )
    assert abs(cc_ub - det_ub) < 1e-12


def test_communication_pd_with_upper_bound_reports_count_conditional():
    owner, deltas, powers, bits, flips, successes = _instance(seed=4, reports=4)
    result = communication_pd_with_upper_bound(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8, samples=2048,
    )
    assert result["count_conditional_ub"] >= result["estimate"] - 1e-9
    assert result["count_conditional_ub"] <= result["deterministic_ub"] + 1e-12


def test_poisson_binomial_count_law_is_exact():
    from uav_otfs_isac.robust_joint_power_bit import _poisson_binomial
    rng = np.random.default_rng(3)
    for reports in [3, 7, 10]:
        successes = rng.uniform(0.2, 0.95, reports)
        law = _poisson_binomial(successes)
        assert abs(law.sum() - 1.0) < 1e-12
        # 与直接枚举 2^R 个 mask 的计数律逐位一致
        expected = np.zeros(reports + 1)
        for mask in range(1 << reports):
            probability = 1.0
            for j in range(reports):
                probability *= successes[j] if (mask >> j) & 1 else 1.0 - successes[j]
            expected[mask.bit_count()] += probability
        assert np.allclose(law, expected, atol=1e-12)


def test_count_stratified_mc_unbiased_and_tight():
    # R=8, max_exact_reports=7 强制 MC 分支: 分层估计在 8 个独立种子上的
    # 均值必须接近精确边缘化 (无偏), 且单次估计的误差远小于普通 MC 的
    # 典型量级 (计数维已精确, 仅剩组内子集不确定性).
    owner, deltas, powers, bits, flips, successes = _instance(seed=11, reports=8)
    exact = per_report_communication_target_pd(
        owner, deltas, powers, bits, flips, successes,
        grid=32, max_exact_reports=8, samples=2048,
    )
    estimates = [
        per_report_communication_target_pd(
            owner, deltas, powers, bits, flips, successes,
            grid=32, max_exact_reports=7, samples=2048, rng=np.random.default_rng(seed),
        )
        for seed in range(8)
    ]
    mean_estimate = float(np.mean(estimates))
    # 单次估计的采样噪声 ~ 8e-4, 8 个种子的均值噪声 ~ 3e-4: 无偏性在
    # 噪声水平内成立, 而计数维已精确, 单次误差远小于普通 MC 的典型量级
    # (普通 MC 在计数维的波动 ~ 0.02).
    assert abs(mean_estimate - exact) < 8e-4, (mean_estimate, exact)
    assert max(abs(value - exact) for value in estimates) < 4e-3