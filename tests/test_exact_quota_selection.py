import numpy as np

from uav_otfs_isac.controlled import symmetric_diversity_model
from uav_otfs_isac.exact_quota_selection import (
    _pareto_dominated_options,
    best_per_size,
    exact_budget_select,
    exact_maxmin_select,
    exact_quota_select,
    subset_expected_pd_map,
)
from uav_otfs_isac.expected_pd import expected_pd_greedy_select
from uav_otfs_isac.models import TargetEvidenceModel
from uav_otfs_isac.scalable_selection import (
    minimum_cost_to_threshold,
    scaled_maxmin_select,
)


def _models():
    return [
        symmetric_diversity_model(
            np.array([1.6, 1.3, 1.1, 0.9]), success_probability=0.8
        )
        for _ in range(3)
    ]


def _nonuniform_models(rng):
    models = []
    for _ in range(3):
        models.append(symmetric_diversity_model(
            np.sort(rng.uniform(0.6, 1.8, size=4))[::-1],
            success_probability=float(rng.uniform(0.5, 0.95)),
            report_bits=np.array([1, 2, 3, 5]),
        ))
    return models


def _objective_score(values, qos, qos_w, perf_w):
    values = np.asarray(values, dtype=float)
    gap = float(np.sum(
        qos_w * np.maximum(qos - values, 0.0) / np.maximum(qos, 1e-12)
    ))
    return (
        -gap,
        float(np.sum(perf_w * values)),
        float(np.min(values)),
    )


def _exhaustive_oracle(
    models, budget_bits, false_alarm_rate, qos, qos_w, perf_w, grid
):
    subsets = [
        subset_expected_pd_map(model, false_alarm_rate, grid=grid)
        for model in models
    ]
    best = None

    def recurse(target_index, groups, used):
        nonlocal best
        if target_index == len(models):
            values = [
                next(
                    value for scheduled, value in subsets[q].items()
                    if scheduled == groups[q]
                )
                for q in range(len(models))
            ]
            score = _objective_score(values, qos, qos_w, perf_w)
            if best is None or score > best[0]:
                best = (
                    score,
                    tuple(groups),
                    np.asarray(values, dtype=float),
                    used,
                )
            return
        model = models[target_index]
        for scheduled, _ in subsets[target_index].items():
            cost = sum(
                int(model.report_bits[i])
                for i in scheduled
                if i != model.owner
            )
            if used + cost > budget_bits:
                continue
            recurse(target_index + 1, groups + [scheduled], used + cost)

    recurse(0, [], 0)
    assert best is not None
    return best


def _exhaustive_maxmin_oracle(models, budget_bits, false_alarm_rate, grid):
    subsets = [
        subset_expected_pd_map(model, false_alarm_rate, grid=grid)
        for model in models
    ]
    best = None

    def recurse(index, groups, used):
        nonlocal best
        if index == len(models):
            values = np.asarray([
                subsets[q][groups[q]] for q in range(len(models))
            ], dtype=float)
            score = float(np.min(values))
            if best is None or score > best[0]:
                best = (score, tuple(groups), values, used)
            return
        model = models[index]
        for scheduled, _ in subsets[index].items():
            cost = sum(
                int(model.report_bits[i])
                for i in scheduled
                if i != model.owner
            )
            if used + cost <= budget_bits:
                recurse(index + 1, groups + [scheduled], used + cost)

    recurse(0, [], 0)
    assert best is not None
    return best


def _exhaustive_maxmin_secondary_oracle(
    models, budget_bits, false_alarm_rate, qos, qos_w, perf_w, grid
):
    subsets = [
        subset_expected_pd_map(model, false_alarm_rate, grid=grid)
        for model in models
    ]
    best = None

    def recurse(index, groups, used):
        nonlocal best
        if index == len(models):
            values = np.asarray([
                subsets[q][groups[q]] for q in range(len(models))
            ], dtype=float)
            gap = float(np.sum(
                qos_w * np.maximum(qos - values, 0.0) / np.maximum(qos, 1e-12)
            ))
            score = (
                float(np.min(values)),
                -gap,
                float(np.sum(perf_w * values)),
            )
            if best is None or score > best[0]:
                best = (score, tuple(groups), values, used)
            return
        model = models[index]
        for scheduled, _ in subsets[index].items():
            cost = sum(
                int(model.report_bits[i])
                for i in scheduled
                if i != model.owner
            )
            if used + cost <= budget_bits:
                recurse(index + 1, groups + [scheduled], used + cost)

    recurse(0, [], 0)
    assert best is not None
    return best


def test_best_per_size_returns_maximum_subset():
    model = _models()[0]
    values = subset_expected_pd_map(model, 0.05, grid=256)
    best = best_per_size(model, values)
    for size, (scheduled, value) in enumerate(best):
        assert len(scheduled) == size + 1
        candidates = [
            candidate for candidate, candidate_value in values.items()
            if len(candidate) == size + 1
        ]
        assert value == max(values[candidate] for candidate in candidates)


def test_pareto_dominated_options_keeps_only_cost_value_frontier():
    options = [
        (1, frozenset({0}), 0.7),
        (2, frozenset({0, 1}), 0.9),
        (2, frozenset({0, 2}), 0.85),
        (3, frozenset({0, 1, 2}), 0.8),
    ]
    kept = _pareto_dominated_options(options)
    assert kept == [
        (1, frozenset({0}), 0.7),
        (2, frozenset({0, 1}), 0.9),
    ]


def test_exact_quota_respects_budget_and_owner():
    models = _models()
    result = exact_quota_select(
        models, budget_bits=10, false_alarm_rate=0.05, grid=256
    )
    assert result.used_bits <= 10
    for q, model in enumerate(models):
        assert model.owner in result.scheduled[q]


def test_exact_quota_never_worse_than_greedy():
    models = _models()
    greedy = expected_pd_greedy_select(
        models, budget_bits=10, false_alarm_rate=0.05, grid=256
    )
    exact = exact_quota_select(
        models, budget_bits=10, false_alarm_rate=0.05, grid=256
    )
    assert np.mean(exact.expected_pd) >= np.mean(greedy.expected_pd) - 1e-12
    assert exact.normalized_qos_gap <= greedy.normalized_qos_gap + 1e-12


def test_exact_budget_matches_exhaustive_oracle_nonuniform_costs():
    rng = np.random.default_rng(20260806)
    qos = np.array([0.85, 0.82, 0.88])
    qos_w = np.array([0.4, 0.3, 0.3])
    perf_w = np.array([1.0, 0.8, 0.9])
    for budget in (3, 5, 7, 9, 11):
        models = _nonuniform_models(rng)
        exact = exact_budget_select(
            models, budget, 0.05, qos_pd=qos, qos_weights=qos_w,
            performance_weights=perf_w, grid=256,
        )
        score, scheduled, values, used = _exhaustive_oracle(
            models, budget, 0.05, qos, qos_w, perf_w, grid=256
        )
        assert exact.used_bits <= budget
        assert _objective_score(
            exact.expected_pd, qos, qos_w, perf_w
        ) == score
        assert np.allclose(exact.expected_pd, values)


def test_exact_budget_respects_heterogeneous_costs():
    models = _nonuniform_models(np.random.default_rng(7))
    result = exact_budget_select(
        models, budget_bits=4, false_alarm_rate=0.05, grid=256
    )
    assert result.used_bits <= 4
    for q, model in enumerate(models):
        assert model.owner in result.scheduled[q]


def test_exact_budget_never_worse_than_greedy_nonuniform_costs():
    rng = np.random.default_rng(20260806)
    for budget in (5, 8, 11):
        models = _nonuniform_models(rng)
        greedy = expected_pd_greedy_select(
            models, budget, 0.05, grid=256
        )
        exact = exact_budget_select(
            models, budget, 0.05, grid=256
        )
        assert np.mean(exact.expected_pd) >= np.mean(greedy.expected_pd) - 1e-12
        assert exact.normalized_qos_gap <= greedy.normalized_qos_gap + 1e-12


def test_exact_budget_lexicographically_dominates_greedy_under_qos():
    rng = np.random.default_rng(99)
    qos = np.array([0.80, 0.85, 0.90])
    qos_w = np.array([0.5, 0.3, 0.2])
    perf_w = np.array([1.0, 1.2, 0.8])
    for budget in (6, 9):
        models = _nonuniform_models(rng)
        greedy = expected_pd_greedy_select(
            models, budget, 0.05, qos_pd=qos, qos_weights=qos_w,
            performance_weights=perf_w, grid=256,
        )
        exact = exact_budget_select(
            models, budget, 0.05, qos_pd=qos, qos_weights=qos_w,
            performance_weights=perf_w, grid=256,
        )
        greedy_score = _objective_score(
            greedy.expected_pd, qos, qos_w, perf_w
        )
        exact_score = _objective_score(
            exact.expected_pd, qos, qos_w, perf_w
        )
        assert exact_score >= greedy_score


def test_exact_budget_reduces_to_quota_for_equal_costs():
    models = _models()
    quota = exact_quota_select(
        models, budget_bits=10, false_alarm_rate=0.05, grid=256
    )
    budget = exact_budget_select(
        models, budget_bits=10, false_alarm_rate=0.05, grid=256
    )
    assert np.allclose(quota.expected_pd, budget.expected_pd)
    assert quota.normalized_qos_gap == budget.normalized_qos_gap


def test_exact_maxmin_matches_exhaustive_oracle_nonuniform_costs():
    rng = np.random.default_rng(20260807)
    for budget in (3, 5, 7, 9, 11):
        models = _nonuniform_models(rng)
        exact = exact_maxmin_select(
            models, budget, 0.05, grid=256
        )
        score, scheduled, values, used = _exhaustive_maxmin_oracle(
            models, budget, 0.05, grid=256
        )
        assert exact.used_bits <= budget
        assert np.isclose(np.min(exact.expected_pd), score)
        assert np.min(exact.expected_pd) >= np.min(values) - 1e-12


def test_exact_maxmin_never_worse_than_greedy_worst_target():
    rng = np.random.default_rng(20260807)
    for budget in (5, 8, 11):
        models = _nonuniform_models(rng)
        greedy = expected_pd_greedy_select(
            models, budget, 0.05, grid=256
        )
        exact = exact_maxmin_select(
            models, budget, 0.05, grid=256
        )
        assert np.min(exact.expected_pd) >= np.min(greedy.expected_pd) - 1e-12


def test_exact_maxmin_secondary_matches_exhaustive_oracle():
    rng = np.random.default_rng(20260812)
    qos = np.array([0.80, 0.85, 0.90])
    qos_w = np.array([0.5, 0.3, 0.2])
    perf_w = np.array([1.0, 1.2, 0.8])
    for budget in (5, 9):
        models = _nonuniform_models(rng)
        exact = exact_maxmin_select(
            models, budget, 0.05, qos_pd=qos, qos_weights=qos_w,
            performance_weights=perf_w, grid=256,
        )
        score, scheduled, values, used = _exhaustive_maxmin_secondary_oracle(
            models, budget, 0.05, qos, qos_w, perf_w, grid=256
        )
        gap = float(np.sum(
            qos_w * np.maximum(qos - exact.expected_pd, 0.0)
            / np.maximum(qos, 1e-12)
        ))
        exact_score = (
            float(np.min(exact.expected_pd)),
            -gap,
            float(np.sum(perf_w * exact.expected_pd)),
        )
        assert exact_score == score


def test_minimum_cost_to_threshold_matches_bruteforce():
    model = _nonuniform_models(np.random.default_rng(3))[0]
    values = subset_expected_pd_map(model, 0.05, grid=256)
    thresholds = sorted({float(value) for value in values.values()})
    for threshold in thresholds[::2]:
        exact_result = minimum_cost_to_threshold(
            model, threshold, 0.05, grid=256
        )
        brute_cost = min(
            sum(
                int(model.report_bits[i])
                for i in scheduled
                if i != model.owner
            )
            for scheduled, value in values.items()
            if value >= threshold - 1e-12
        )
        assert exact_result is not None
        assert exact_result[0] == brute_cost
        assert values[exact_result[1]] >= threshold - 1e-12


def test_scaled_maxmin_close_to_exhaustive_oracle_nonuniform_costs():
    rng = np.random.default_rng(20260808)
    for budget in (3, 5, 7, 9, 11):
        models = _nonuniform_models(rng)
        exact = scaled_maxmin_select(
            models, budget, 0.05, grid=256, tolerance=1e-7
        )
        score, scheduled, values, used = _exhaustive_maxmin_oracle(
            models, budget, 0.05, grid=256
        )
        assert exact.used_bits <= budget
        assert np.min(exact.expected_pd) <= score + 1e-6
        assert np.min(exact.expected_pd) >= score - 1e-6
        assert exact.certificate_upper_bound is not None
        assert exact.certificate_upper_bound >= np.min(exact.expected_pd) - 1e-6


def test_scaled_maxmin_respects_budget_and_owner():
    models = _nonuniform_models(np.random.default_rng(8))
    result = scaled_maxmin_select(
        models, budget_bits=5, false_alarm_rate=0.05, grid=256
    )
    assert result.used_bits <= 5
    for q, model in enumerate(models):
        assert model.owner in result.scheduled[q]


def test_minimum_cost_low_pd_matches_bruteforce_when_all_set_is_not_max():
    rng = np.random.default_rng(1)
    n = 8
    raw0 = rng.normal(size=(n, n))
    sigma0 = raw0 @ raw0.T + 0.3 * np.eye(n)
    scale = rng.uniform(0.4, 2.0, n)
    raw1 = rng.normal(size=(n, n))
    sigma1 = (raw1 @ raw1.T + 0.3 * np.eye(n)) * (
        scale[:, None] * scale[None, :]
    )
    mu0 = rng.normal(size=n) * 0.1
    mu1 = mu0 + rng.normal(size=n)
    report_bits = np.array([0] + [1 + (i % 3) for i in range(1, n)], dtype=int)
    model = TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=mu0,
        mu1=mu1,
        sigma0=sigma0,
        sigma1=sigma1,
        success_prob=np.ones(n),
        report_bits=report_bits,
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )
    values = subset_expected_pd_map(model, 0.05, grid=128)
    all_set = frozenset(range(n))
    all_value = values[all_set]
    counterexample = next(
        (
            (scheduled, value)
            for scheduled, value in values.items()
            if value > all_value + 1e-9 and value < 0.5
        ),
        None,
    )
    assert counterexample is not None
    scheduled, subset_value = counterexample
    threshold = 0.5 * (subset_value + all_value)
    brute = minimum_cost_to_threshold(
        model, threshold, 0.05, grid=128, max_exhaustive_reports=14
    )
    branch = minimum_cost_to_threshold(
        model, threshold, 0.05, grid=128, max_exhaustive_reports=0
    )
    assert brute is not None
    assert branch == brute


def test_exact_selectors_raise_when_greedy_fallback_disabled():
    n = 12
    model = TargetEvidenceModel(
        target_id=0,
        owner=0,
        mu0=np.zeros(n),
        mu1=np.linspace(0.2, 2.0, n),
        sigma0=np.eye(n),
        sigma1=np.eye(n),
        success_prob=np.ones(n),
        report_bits=np.array([0] + [1] * (n - 1), dtype=int),
        bit_flip_prob=np.zeros(n),
        quantizer_edges=np.array([-np.inf, 0.0, np.inf]),
        quantizer_values=np.array([-1.0, 1.0]),
    )
    for selector in (exact_budget_select, exact_maxmin_select):
        raised = False
        try:
            selector(
                [model], 5, 0.05, max_exhaustive_reports=5,
                allow_greedy_fallback=False, grid=64,
            )
        except ValueError:
            raised = True
        assert raised
