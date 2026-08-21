"""Tests for detection-aware quantization and information gradients.

Covers the provable monotonicity facts (bits refinement + data processing,
BSC cascade, erasure), the greedy-vs-exhaustive allocation quality bound,
the design-metric hierarchy (I+ mean drift misranks designs, Chernoff
tracks exact P_D), and the 1-bit LLR structure classification.
"""

import itertools

import numpy as np
import pytest

from uav_otfs_isac.detection_quantization import (
    information_waterfilling,
    link_information_vs_bits,
    llr_1bit_structure,
    llr_quadratic,
    maxmin_pd_allocation,
    one_bit_kl_scan,
    optimal_span,
    option_metric_vs_bits,
    verify_bits_concavity,
    verify_bits_monotonicity,
    verify_flip_monotonicity,
    verify_pd_bits_monotonicity,
    verify_success_monotonicity,
)
from uav_otfs_isac.detection_information import (
    chernoff_information,
    post_communication_likelihoods,
    sequential_pd,
)
from uav_otfs_isac.reporting import quantizer_from_gaussian_range


def _random_instances(n, seed, snr_range=(-3.0, 8.0)):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        l_acc = 4
        snr_db = float(rng.uniform(*snr_range))
        noncentrality = l_acc * 10 ** (snr_db / 10.0)
        mu0, var0 = float(l_acc), float(l_acc)
        mu1 = mu0 + noncentrality
        var1 = var0 + 2.0 * noncentrality
        flip = float(rng.uniform(0.01, 0.15))
        success = float(rng.uniform(0.5, 0.98))
        out.append((mu0, var0, mu1, var1, flip, success))
    return out


def test_bits_monotonicity_is_provable_and_holds():
    instances = _random_instances(25, 10)
    result = verify_bits_monotonicity(instances, bits_max=6)
    assert result["passed"]
    assert result["violations"] == []


def test_flip_and_success_monotonicity_hold():
    instances = _random_instances(15, 11)
    assert verify_flip_monotonicity(instances)["passed"]
    assert verify_success_monotonicity(instances)["passed"]


def test_bits_concavity_is_not_assumed():
    # I+(b) margins may jump when the step resolves the H0 bulk; the
    # concavity check exists to keep that fact visible, not to pass.
    # Measured on the test instance family: violations in 30/40 instances.
    instances = _random_instances(40, 12)
    result = verify_bits_concavity(instances, bits_max=6)
    assert not result["passed"]
    assert result["violations"]


def test_waterfilling_is_near_exact_vs_bruteforce():
    rng = np.random.default_rng(13)
    worst_gap = 0.0
    for _ in range(15):
        links = [_random_instances(1, 0)[0] for _ in range(3)]
        profiles = [link_information_vs_bits(*l, 5) for l in links]
        budget = 8
        bits, total = information_waterfilling(profiles, budget)
        best = 0.0
        for combo in itertools.product(range(6), repeat=3):
            if sum(combo) != budget:
                continue
            value = sum(profiles[k][combo[k]] for k in range(3))
            best = max(best, value)
        gap = (best - total) / max(best, 1e-12)
        worst_gap = max(worst_gap, gap)
    assert worst_gap <= 0.05


def test_span_design_metric_hierarchy():
    """Chernoff must track the exact-P_D span ranking far better than the
    KL mean drift (design metric lesson)."""
    from uav_otfs_isac.detection_quantization import quantizer_edges

    instances = _random_instances(12, 14)
    spans = [3.0, 4.0, 6.0, 8.0, 12.0, 20.0]
    i_plus_fail = 0
    chernoff_fail = 0
    for mu0, var0, mu1, var1, flip, success in instances:
        pd_by_span = {}
        i_by_span = {}
        c_by_span = {}
        for span in spans:
            edges, values = quantizer_edges(
                mu0, var0, mu1, var1, 3, span,
            )
            info = post_communication_likelihoods(
                mu0, var0, mu1, var1, edges, values, 3, flip, success,
            )
            pd_by_span[span] = float(sequential_pd(
                info["p1_y"], info["p0_y"], 4, 0.05, 0.05,
            )["pd"])
            i_by_span[span] = float(info["kl_plus"])
            c_by_span[span] = float(chernoff_information(
                info["p1_y"], info["p0_y"],
            ))
        best_span = max(spans, key=lambda s: pd_by_span[s])
        best_pd = pd_by_span[best_span]
        i_span = max(spans, key=lambda s: i_by_span[s])
        c_span = max(spans, key=lambda s: c_by_span[s])
        if pd_by_span[i_span] < 0.99 * best_pd:
            i_plus_fail += 1
        if pd_by_span[c_span] < 0.99 * best_pd:
            chernoff_fail += 1
    assert i_plus_fail > chernoff_fail
    assert chernoff_fail <= 3


def test_one_bit_window_never_worse_than_single():
    instances = _random_instances(6, 15)
    for mu0, var0, mu1, var1, flip, success in instances:
        scan = one_bit_kl_scan(mu0, var0, mu1, var1, flip, success, grid=151)
        single = scan["best_single_threshold"]
        window = scan["best_two_sided_window"]
        assert window[2] >= single[1] - 1e-12


def test_llr_structure_classification_by_variance_ratio():
    assert llr_1bit_structure(4.0, 4.0, 13.6, 23.2)["kind"] == "two_sided_window"
    assert llr_1bit_structure(4.0, 23.2, 13.6, 4.0)["kind"] == "single_interval"
    assert llr_1bit_structure(4.0, 4.0, 13.6, 4.0)["kind"] == "degenerate_equal_variance"


def test_llr_quadratic_matches_direct_log_ratio():
    x = np.linspace(-10, 30, 5)
    from scipy.stats import norm
    direct = np.log(
        norm.pdf((x - 13.6) / np.sqrt(23.2)) / np.sqrt(23.2)
        / (norm.pdf((x - 4.0) / np.sqrt(4.0)) / np.sqrt(4.0))
    )
    assert np.allclose(llr_quadratic(4.0, 4.0, 13.6, 23.2, x), direct,
                       atol=1e-9)


def test_optimal_span_reports_default_loss():
    mu0, var0, mu1, var1, flip, success = _random_instances(1, 16)[0]
    result = optimal_span(mu0, var0, mu1, var1, 3, flip, success)
    assert result["metric"] == "chernoff"
    assert 1.0 <= result["span_opt"] <= 12.0
    assert result["relative_gain_default"] >= 0.0
    assert result["chernoff_opt"] >= result["chernoff_default"] - 1e-12
    assert result["i_plus_opt"] >= 0.0
    # the I+ variant is exposed for comparison and never claims optimality
    result_i = optimal_span(mu0, var0, mu1, var1, 3, flip, success,
                            metric="i_plus")
    assert result_i["metric"] == "i_plus"
    assert result_i["metric_opt"] >= result_i["metric_default"] - 1e-12


def test_llr_structure_roots_satisfy_level():
    mu0, var0, mu1, var1 = 4.0, 4.0, 13.6, 23.2
    structure = llr_1bit_structure(mu0, var0, mu1, var1)
    assert structure["kind"] == "two_sided_window"
    l_star = float(llr_quadratic(
        mu0, var0, mu1, var1, np.array([structure["x_star"]])
    )[0])
    for (root_a, root_b), delta in zip(structure["windows"], (0.5, 1.0, 2.0)):
        residual = llr_quadratic(
            mu0, var0, mu1, var1, np.array([root_a, root_b]),
        ) - (l_star + delta)
        assert np.allclose(residual, 0.0, atol=1e-9)
        assert root_a < structure["x_star"] < root_b


def test_single_interval_family_beats_threshold_for_var1_lt_var0():
    # when var1 < var0 the canonical 1-bit region is a single interval;
    # the scan must find it (at least as good as any threshold/window)
    mu0, var0, mu1, var1 = 4.0, 23.2, 13.6, 4.0
    flip, success = 0.05, 0.9
    scan = one_bit_kl_scan(mu0, var0, mu1, var1, flip, success, grid=151)
    assert scan["llr_structure"]["kind"] == "single_interval"
    interval = scan["best_single_interval"]
    assert interval is not None
    assert interval[2] >= scan["best_single_threshold"][1] - 1e-12
    assert interval[2] >= scan["best_two_sided_window"][2] - 1e-12


def test_bits_zero_has_no_information():
    mu0, var0, mu1, var1, flip, success = _random_instances(1, 17)[0]
    profile = link_information_vs_bits(mu0, var0, mu1, var1, flip, success, 5)
    assert profile[0] == 0.0


def test_pd_bits_monotonicity_diagnostic_honest():
    # Theorem: exact monotonicity for the true LLR statistic (NP
    # admissibility of the refinement chain).  The grid statistic is
    # slightly suboptimal, so the diagnostic reports violations rather
    # than claiming exact monotonicity; measured violations are small
    # (<= 0.008) and never flip the allocation ground truth.
    instances = _random_instances(12, 19)
    result = verify_pd_bits_monotonicity(instances, bits_max=5, n=4)
    assert "passed" in result and "violations" in result
    assert result["max_violation"] <= 0.01


def test_pd_curve_baseline_at_zero_bits():
    mu0, var0, mu1, var1, flip, success = _random_instances(1, 20)[0]
    curve = option_metric_vs_bits(mu0, var0, mu1, var1, flip, success, 3,
                                  metric="pd")
    assert curve[0] == pytest.approx(0.05)
    assert 0.5 <= curve[3] <= 1.0


def _link_option(rng, bits_max=5):
    l_acc = 4
    snr_db = float(rng.uniform(-3.0, 8.0))
    nc = l_acc * 10 ** (snr_db / 10.0)
    return {
        "mu0": float(l_acc), "var0": float(l_acc),
        "mu1": float(l_acc + nc), "var1": float(l_acc + 2.0 * nc),
        "flip": float(rng.uniform(0.01, 0.15)),
        "success": float(rng.uniform(0.5, 0.98)),
        "bits_max": bits_max,
    }


def test_maxmin_allocation_matches_bruteforce_exactly():
    """The floor-cover allocation must reproduce the exhaustive max-min
    worst-target P_D(4) on small instances where enumeration is possible."""
    rng = np.random.default_rng(21)
    for _ in range(6):
        targets = [[_link_option(rng, bits_max=3) for _ in range(2)]
                   for _ in range(3)]
        budget = 8
        result = maxmin_pd_allocation(targets, budget, metric="pd")
        # option_metric_vs_bits is deterministic, so precompute each link's
        # curve once instead of re-evaluating it for every enumeration combo.
        curves = [
            [
                option_metric_vs_bits(
                    targets[t][r]["mu0"], targets[t][r]["var0"],
                    targets[t][r]["mu1"], targets[t][r]["var1"],
                    targets[t][r]["flip"], targets[t][r]["success"],
                    3, metric="pd",
                )
                for r in range(2)
            ]
            for t in range(3)
        ]
        best_worst = 0.0
        for combo in itertools.product(range(4), repeat=6):
            if sum(combo) > budget:
                continue
            worst = 1.0
            for t in range(3):
                best = 0.0
                for r in range(2):
                    best = max(best, float(curves[t][r][combo[t * 2 + r]]))
                worst = min(worst, best)
            best_worst = max(best_worst, worst)
        assert result["worst_metric"] == pytest.approx(best_worst, abs=1e-9)
        assert result["worst_metric"] == pytest.approx(result["levels"], abs=1e-9)


def test_maxmin_allocation_beats_waterfilling():
    """The exact max-min P_D allocation dominates both I+ water-fillers in
    the worst-target metric (this is the point of the floor-cover)."""
    rng = np.random.default_rng(22)
    for _ in range(4):
        targets = [[_link_option(rng, bits_max=5) for _ in range(2)]
                   for _ in range(3)]
        budget = 8
        result = maxmin_pd_allocation(targets, budget, metric="pd")
        # water-filling on flat I+ profiles
        flat = []
        for t in targets:
            for o in t:
                flat.append(option_metric_vs_bits(
                    o["mu0"], o["var0"], o["mu1"], o["var1"],
                    o["flip"], o["success"], 5, metric="i_plus",
                ))
        bits_sum, _ = information_waterfilling(flat, budget)
        bits_min, _ = information_waterfilling(flat, budget, max_min=True)
        for bits in (bits_sum, bits_min):
            worst = 1.0
            for t in range(3):
                best = 0.0
                for r in range(2):
                    curve = option_metric_vs_bits(
                        targets[t][r]["mu0"], targets[t][r]["var0"],
                        targets[t][r]["mu1"], targets[t][r]["var1"],
                        targets[t][r]["flip"], targets[t][r]["success"],
                        5, metric="pd",
                    )
                    best = max(best, float(curve[bits[t * 2 + r]]))
                worst = min(worst, best)
            assert result["worst_metric"] >= worst - 1e-9


def test_maxmin_allocation_scales_and_respects_budget():
    rng = np.random.default_rng(23)
    targets = [[_link_option(rng, bits_max=5) for _ in range(3)]
               for _ in range(5)]
    result = maxmin_pd_allocation(targets, 14, metric="pd")
    assert sum(sum(row) for row in result["bits"]) <= 14
    assert result["worst_metric"] >= 0.05
    assert all(a >= result["levels"] - 1e-9 for a in result["achieved"])


def test_maxmin_allocation_chernoff_proxy_reasonable():
    """Chernoff-floor allocation should be close to the exact-P_D allocation
    in the worst-target P_D (the metric hierarchy of the gate)."""
    rng = np.random.default_rng(24)
    for _ in range(3):
        targets = [[_link_option(rng, bits_max=5) for _ in range(2)]
                   for _ in range(3)]
        budget = 8
        exact = maxmin_pd_allocation(targets, budget, metric="pd")
        proxy = maxmin_pd_allocation(targets, budget, metric="chernoff")
        # realized worst P_D of the Chernoff allocation
        worst = 1.0
        for t in range(3):
            best = 0.0
            for r in range(2):
                curve = option_metric_vs_bits(
                    targets[t][r]["mu0"], targets[t][r]["var0"],
                    targets[t][r]["mu1"], targets[t][r]["var1"],
                    targets[t][r]["flip"], targets[t][r]["success"],
                    5, metric="pd",
                )
                best = max(best, float(curve[proxy["bits"][t][r]]))
            worst = min(worst, best)
        assert worst >= 0.5 * exact["worst_metric"]


def test_waterfilling_respects_budget_and_saturation():
    rng = np.random.default_rng(18)
    links = [_random_instances(1, 0)[0] for _ in range(3)]
    profiles = [link_information_vs_bits(*l, 4) for l in links]
    bits, _ = information_waterfilling(profiles, 100)
    assert sum(bits) <= 3 * 4
    assert all(bits[k] <= 4 for k in range(3))
    bits2, _ = information_waterfilling(profiles, 3)
    assert sum(bits2) == 3