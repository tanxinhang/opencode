import numpy as np
from scipy.stats import norm

from uav_otfs_isac.fusion import (
    conditional_marginal_deflection,
    gaussian_pd_closed_form,
    gaussian_detection_probability,
    optimal_gaussian_detection_probability,
    optimal_gaussian_weights,
    optimal_deflection,
    optimal_weights,
    pd_shift_upper_bound,
)


def test_schur_gain_matches_direct_difference():
    delta = np.array([1.0, 0.8, 0.4])
    sigma = np.array([[1.0, 0.4, 0.1], [0.4, 1.2, 0.25], [0.1, 0.25, 0.9]])
    selected = {0, 1}
    direct = optimal_deflection(delta, sigma, {0, 1, 2}) - optimal_deflection(delta, sigma, selected)
    schur = conditional_marginal_deflection(delta, sigma, selected, 2)
    assert np.isclose(direct, schur, rtol=1e-10, atol=1e-10)
    assert schur >= 0


def test_optimal_weights_have_unit_null_variance():
    delta = np.array([1.0, 0.3])
    sigma = np.array([[2.0, 0.2], [0.2, 1.0]])
    weights = optimal_weights(delta, sigma, {0, 1})
    assert np.isclose(weights @ sigma @ weights, 1.0)


def test_correlation_reduces_redundant_gain():
    delta = np.array([1.0, 1.0, 0.6])
    sigma = np.array([[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 1.0]])
    redundant = conditional_marginal_deflection(delta, sigma, {0}, 1)
    independent = conditional_marginal_deflection(delta, sigma, {0}, 2)
    assert independent > redundant


def test_gaussian_pd_reduces_to_equal_covariance_formula():
    mu0 = np.zeros(2); mu1 = np.array([1.0, 0.5]); covariance = np.eye(2)
    alpha = 0.05
    pd = gaussian_detection_probability(
        mu0, mu1, covariance, covariance, {0, 1}, alpha
    )
    deflection = optimal_deflection(mu1 - mu0, covariance, {0, 1})
    expected = norm.sf(norm.ppf(1.0 - alpha) - np.sqrt(deflection))
    assert np.isclose(pd, expected)


def test_optimal_pd_matches_closed_form_in_proportional_regime():
    rng = np.random.default_rng(20260805)
    for _ in range(12):
        n = 5
        raw = rng.normal(size=(n, n))
        sigma0 = raw @ raw.T + 0.2 * np.eye(n)
        delta = rng.normal(size=n)
        ratio = float(rng.uniform(0.3, 2.5))
        sigma1 = ratio * sigma0
        mu0 = rng.normal(size=n) * 0.1
        mu1 = mu0 + delta
        for indices in ({0}, {0, 2}, {0, 1, 3, 4}):
            deflection = optimal_deflection(delta, sigma0, indices)
            closed = gaussian_pd_closed_form(deflection, ratio, 0.05)
            pd = gaussian_detection_probability(
                mu0, mu1, sigma0, sigma1, indices, 0.05
            )
            optimal = optimal_gaussian_detection_probability(
                mu0, mu1, sigma0, sigma1, indices, 0.05
            )
            assert np.isclose(pd, closed, rtol=1e-10, atol=1e-10)
            assert np.isclose(optimal, closed, rtol=1e-9, atol=1e-9)


def test_optimal_pd_is_set_monotone_at_operating_points():
    rng = np.random.default_rng(20260805)
    tested_edges = 0
    for _ in range(25):
        n = 5
        raw0 = rng.normal(size=(n, n))
        sigma0 = raw0 @ raw0.T + 0.4 * np.eye(n)
        raw1 = rng.normal(size=(n, n))
        scale = rng.uniform(0.4, 2.0, n)
        sigma1 = (raw1 @ raw1.T + 0.4 * np.eye(n)) * (
            scale[:, None] * scale[None, :]
        )
        mu0 = rng.normal(size=n) * 0.1
        mu1 = mu0 + rng.normal(size=n) * 1.5
        for mask in range(1 << n):
            base = {i for i in range(n) if mask & (1 << i)}
            if not base:
                continue
            base_optimal = optimal_gaussian_detection_probability(
                mu0, mu1, sigma0, sigma1, base, 0.05, grid=1024
            )
            if base_optimal < 0.5:
                continue
            for candidate in range(n):
                if candidate in base:
                    continue
                new_optimal = optimal_gaussian_detection_probability(
                    mu0, mu1, sigma0, sigma1, base | {candidate}, 0.05,
                    grid=1024,
                )
                assert new_optimal >= base_optimal - 1e-9
                tested_edges += 1
    assert tested_edges >= 50


def test_optimal_pd_never_below_deflection_pd():
    rng = np.random.default_rng(20260805)
    for _ in range(10):
        n = 5
        raw0 = rng.normal(size=(n, n))
        sigma0 = raw0 @ raw0.T + 0.3 * np.eye(n)
        raw1 = rng.normal(size=(n, n))
        scale = rng.uniform(0.4, 2.0, n)
        sigma1 = (raw1 @ raw1.T + 0.3 * np.eye(n)) * (
            scale[:, None] * scale[None, :]
        )
        mu0 = rng.normal(size=n) * 0.1
        mu1 = mu0 + rng.normal(size=n)
        indices = {0, 2, 3}
        deflection_pd = gaussian_detection_probability(
            mu0, mu1, sigma0, sigma1, indices, 0.05
        )
        optimal_pd = optimal_gaussian_detection_probability(
            mu0, mu1, sigma0, sigma1, indices, 0.05
        )
        assert optimal_pd >= deflection_pd - 1e-12


def test_optimal_gaussian_weights_have_unit_null_variance():
    rng = np.random.default_rng(20260805)
    n = 4
    raw = rng.normal(size=(n, n))
    sigma0 = raw @ raw.T + 0.2 * np.eye(n)
    sigma1 = 1.7 * sigma0
    delta = rng.normal(size=n)
    weights = optimal_gaussian_weights(
        np.zeros(n), delta, sigma0, sigma1, {0, 1, 2, 3}, 0.05
    )
    assert np.isclose(weights @ sigma0 @ weights, 1.0, rtol=1e-8, atol=1e-8)


def test_pd_shift_upper_bound_covers_optimal_linear_score():
    rng = np.random.default_rng(20260809)
    for _ in range(20):
        n = 6
        raw0 = rng.normal(size=(n, n))
        sigma0 = raw0 @ raw0.T + 0.3 * np.eye(n)
        scale = rng.uniform(0.4, 2.0, n)
        raw1 = rng.normal(size=(n, n))
        sigma1 = (raw1 @ raw1.T + 0.3 * np.eye(n)) * (
            scale[:, None] * scale[None, :]
        )
        mu0 = rng.normal(size=n) * 0.1
        mu1 = mu0 + rng.normal(size=n)
        for mask in (0b101, 0b1001, 0b1111, 0b10101):
            indices = {i for i in range(n) if mask & (1 << i)}
            if not indices:
                continue
            pd = optimal_gaussian_detection_probability(
                mu0, mu1, sigma0, sigma1, indices, 0.05, grid=1024
            )
            shift = norm.ppf(np.clip(pd, 1e-12, 1.0 - 1e-12))
            bound = pd_shift_upper_bound(
                mu0, mu1, sigma0, sigma1, indices, 0.05
            )
            assert shift <= bound + 1e-8


def test_pd_shift_upper_bound_covers_high_false_alarm_rate():
    rng = np.random.default_rng(20260813)
    n = 5
    raw0 = rng.normal(size=(n, n))
    sigma0 = raw0 @ raw0.T + 0.4 * np.eye(n)
    scale = rng.uniform(0.4, 2.0, n)
    raw1 = rng.normal(size=(n, n))
    sigma1 = (raw1 @ raw1.T + 0.4 * np.eye(n)) * (
        scale[:, None] * scale[None, :]
    )
    mu0 = rng.normal(size=n) * 0.1
    mu1 = mu0 + rng.normal(size=n)
    indices = {0, 2, 3}
    for false_alarm_rate in (0.55, 0.7, 0.9):
        pd = optimal_gaussian_detection_probability(
            mu0, mu1, sigma0, sigma1, indices, false_alarm_rate, grid=1024
        )
        shift = norm.ppf(np.clip(pd, 1e-12, 1.0 - 1e-12))
        bound = pd_shift_upper_bound(
            mu0, mu1, sigma0, sigma1, indices, false_alarm_rate
        )
        assert shift <= bound + 1e-8


def test_dual_upper_bound_tightens_cauchy_on_correlated_evidence():
    """The mu-parameterized bound must beat the mu=0 Cauchy member."""
    rng = np.random.default_rng(20260818)
    for correlation in (0.8, 0.9, 0.95):
        n = 11
        index = np.arange(n)
        sigma1 = correlation ** np.abs(index[:, None] - index[None, :])
        mu0 = np.zeros(n)
        mu1 = np.concatenate(([0.8], np.linspace(1.5, 0.5, n - 1)))
        sigma0 = np.eye(n)
        indices = set(range(n))
        cholesky0 = np.linalg.cholesky(sigma0)
        inverse0 = np.linalg.inv(cholesky0)
        a = inverse0 @ (mu1 - mu0)
        q = inverse0 @ sigma1 @ inverse0.T
        z = norm.ppf(0.95)
        eigenvalues = np.linalg.eigvalsh(q)
        cauchy = float(np.sqrt(max(a @ np.linalg.solve(q, a), 0.0)))
        cauchy -= z / float(np.sqrt(eigenvalues.max()))
        dual = pd_shift_upper_bound(
            mu0, mu1, sigma0, sigma1, indices, 0.05
        )
        pd = optimal_gaussian_detection_probability(
            mu0, mu1, sigma0, sigma1, indices, 0.05, grid=128
        )
        actual = norm.ppf(np.clip(pd, 1e-12, 1.0 - 1e-12))
        assert dual <= cauchy - 0.1
        assert actual <= dual + 1e-6
