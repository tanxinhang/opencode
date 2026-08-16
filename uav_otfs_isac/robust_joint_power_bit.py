"""Robust joint sensing-power and communication-bit allocation.

Each ``(power, bit)`` option is evaluated both at the clean communication
point and at the worst endpoint ``(flip_hi, success_lo)``.  By Lemma 4.70,
the endpoint is the worst case over the communication ambiguity rectangle,
so the exact DP over robust options is the exact worst-case allocation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from math import log, sqrt

import numpy as np

from .fusion import optimal_gaussian_detection_probability
from .joint_allocation import moments


@dataclass(frozen=True)
class JointPowerBitOption:
    cost_bits: int
    powers: tuple[float, ...]
    bits: tuple[int, ...]
    clean_pd: float
    robust_pd: float


def _target_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probability: float,
    success_probability: float,
    grid: int,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng=None,
) -> float:
    return _expected_communication_pd(
        owner_delta,
        report_deltas,
        powers,
        bits,
        np.full(len(report_deltas), float(flip_probability)),
        np.full(len(report_deltas), float(success_probability)),
        grid,
        max_exact_reports=max_exact_reports,
        samples=samples,
        rng=rng,
    )


def communication_target_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probability: float,
    success_probability: float,
    grid: int = 32,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng=None,
) -> float:
    """P_D of one target with a shared communication channel state."""
    return _target_pd(
        owner_delta,
        report_deltas,
        powers,
        bits,
        flip_probability,
        success_probability,
        grid,
        max_exact_reports=max_exact_reports,
        samples=samples,
        rng=rng,
    )


def _pd_for_mask(owner_delta, entries, mask, grid):
    return _pd_for_mask_cached(
        float(owner_delta), tuple(entries), int(mask), int(grid)
    )


@lru_cache(maxsize=1 << 16)
def _pd_for_mask_cached(owner_delta, entries, mask, grid):
    mu0 = [0.0]
    mu1 = [float(owner_delta)]
    var0 = [1.0]
    var1 = [1.0]
    for j, (m0, m1, v0, v1, _) in enumerate(entries):
        if mask >> j & 1:
            mu0.append(m0)
            mu1.append(m1)
            var0.append(v0)
            var1.append(v1)
    return float(optimal_gaussian_detection_probability(
        np.asarray(mu0),
        np.asarray(mu1),
        np.diag(var0),
        np.diag(var1),
        set(range(len(mu0))),
        0.05,
        grid=grid,
    ))


def _build_entries(
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probabilities: np.ndarray,
    success_probabilities: np.ndarray,
) -> list[tuple[float, float, float, float, float]]:
    """Post-BSC (m0, m1, v0, v1, success) tuples of the active reports."""
    deltas = np.asarray(report_deltas, dtype=float)
    powers = np.asarray(powers, dtype=float)
    bits = np.asarray(bits, dtype=int)
    flips = np.asarray(flip_probabilities, dtype=float)
    successes = np.asarray(success_probabilities, dtype=float)
    entries = []
    for i in range(deltas.size):
        if bits[i] <= 0:
            continue
        scaled = float(deltas[i]) * np.sqrt(max(float(powers[i]), 0.0))
        m0, m1, v0, v1 = moments(scaled, int(bits[i]), float(flips[i]))
        entries.append((m0, m1, v0, v1, float(successes[i])))
    return entries


def _exact_erasure_marginalization(
    owner_delta: float,
    entries: list[tuple[float, float, float, float, float]],
    grid: int,
) -> tuple[float, list[float], np.ndarray]:
    """Exact erasure expectation plus the per-count maxima and count law.

    Returns ``(total, per_count_max, count_probabilities)`` where
    ``total = sum_mask P(mask) P_D(mask)`` is the exact expectation over the
    independent erasure law, ``per_count_max[n]`` is the largest P_D over
    all received sets of size ``n``, and ``count_probabilities[n]`` is the
    exact probability that exactly ``n`` reports are received (the
    Poisson-binomial law of the per-report success probabilities).  All
    three are read off from one pass over the ``2^R`` masks.
    """
    report_count = len(entries)
    total = 0.0
    per_count_max = [-1.0] * (report_count + 1)
    count_probabilities = np.zeros(report_count + 1, dtype=float)
    for mask in range(1 << report_count):
        probability = 1.0
        for j, (_, _, _, _, success) in enumerate(entries):
            if mask >> j & 1:
                probability *= success
            else:
                probability *= 1.0 - success
        received = mask.bit_count()
        count_probabilities[received] += probability
        if probability <= 0.0:
            continue
        value = _pd_for_mask(owner_delta, entries, mask, grid)
        total += probability * value
        if value > per_count_max[received]:
            per_count_max[received] = value
    return float(total), per_count_max, count_probabilities


def _poisson_binomial(
    successes: np.ndarray,
) -> np.ndarray:
    """Exact count law ``c_n = P(N = n)`` of independent Bernoulli draws.

    The received count ``N = sum_j B_j`` of independent per-report success
    indicators follows the Poisson-binomial law, computed exactly by the
    standard O(R^2) dynamic program over the per-report success
    probabilities.  The count marginal is exact even in the Monte Carlo
    regime, so the MC variance is confined to the within-count subset
    uncertainty only.
    """
    probs = np.asarray([1.0], dtype=float)
    for success in successes:
        success = float(success)
        next_probs = np.zeros(probs.size + 1, dtype=float)
        next_probs[:-1] += probs * (1.0 - success)
        next_probs[1:] += probs * success
        probs = next_probs
    return probs


def _size_biased_table(
    weights: np.ndarray,
    size: int,
) -> np.ndarray:
    """Elementary-symmetric-sum table of the conditional Poisson design.

    ``table[j][m] = sum over m-subsets T of {j..R-1} of prod_{k in T} w_k``
    (the elementary symmetric polynomial of degree ``m`` of the suffix
    weights), built by the O(R^2) dynamic program
    ``table[j][m] = table[j+1][m] + w_j * table[j+1][m-1]``.  The table
    gives the exact conditional inclusion probabilities of the design with
    ``P(S) proportional to prod_{j in S} w_j`` restricted to ``|S| = size``.
    """
    stochastic = np.asarray([
        float(w) for w in weights if np.isfinite(w)
    ])
    report_count = int(stochastic.size)
    table = np.zeros((report_count + 1, int(size) + 1), dtype=float)
    table[report_count, 0] = 1.0
    for j in range(report_count - 1, -1, -1):
        table[j, 0] = 1.0
        table[j, 1:] = (
            table[j + 1, 1:] + stochastic[j] * table[j + 1, :-1]
        )
    return table


def _draw_size_biased_subset(
    rng: np.random.Generator,
    weights: np.ndarray,
    size: int,
) -> int:
    """One exact size-biased subset of the given size, as a bit mask.

    Conditional Poisson sampling with inclusion probabilities derived from
    the elementary-symmetric-sum table: ``P(S) = prod_{j in S} w_j / (sum
    over size-``size`` subsets of prod w)``, which is exactly the erasure
    law conditional on ``N = size`` with weights ``w_j = s_j / (1 - s_j)``.
    Reports with ``s_j = 1`` get infinite weight and are always included;
    reports with ``s_j = 0`` get zero weight and are never included.
    """
    if size <= 0:
        return 0
    finite = np.isfinite(weights)
    mandatory = int(np.count_nonzero(~finite))
    if mandatory >= size:
        mask = 0
        for j in range(weights.size):
            if not finite[j]:
                mask |= 1 << int(j)
        return mask
    need = size - mandatory
    table = _size_biased_table(weights, need)
    mask = 0
    stochastic_index = 0
    for j in range(weights.size):
        if not finite[j]:
            mask |= 1 << int(j)
            continue
        if need <= 0:
            break
        denominator = table[stochastic_index, need]
        probability = (
            0.0
            if denominator <= 0.0
            else (
                float(weights[j])
                * table[stochastic_index + 1, need - 1]
                / denominator
            )
        )
        if rng.random() < probability:
            mask |= 1 << int(j)
            need -= 1
        stochastic_index += 1
    return mask


def _expected_communication_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probabilities: np.ndarray,
    success_probabilities: np.ndarray,
    grid: int,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng=None,
) -> float:
    """Expected P_D over independent report erasures.

    Below ``max_exact_reports`` the erasure law is marginalized exactly over
    all ``2^R`` received subsets.  Above it, the received count ``N`` is
    marginalized exactly (Poisson-binomial) and each count stratum is
    estimated by size-biased subset draws, the Rao-Blackwell form of Monte
    Carlo: ``E[P_D] = sum_n c_n E[P_D | N = n]`` with exact ``c_n``.  By the
    law of total variance the stratified estimator has variance
    ``E[Var(P_D | N)] / samples <= Var(P_D) / samples``, so it never loses
    to plain Monte Carlo and is exact whenever the fused P_D depends on the
    received set only through its size.
    """
    entries = _build_entries(
        report_deltas, powers, bits,
        flip_probabilities, success_probabilities,
    )
    if len(entries) <= max_exact_reports:
        return _exact_erasure_marginalization(
            owner_delta, entries, grid
        )[0]
    if rng is None:
        rng = np.random.default_rng(0)
    successes = np.asarray([
        float(success) for _, _, _, _, success in entries
    ])
    count_probabilities = _poisson_binomial(successes)
    weights = np.where(
        successes < 1.0,
        successes / np.maximum(1.0 - successes, 1e-300),
        np.inf,
    )
    estimate = 0.0
    for n in range(len(entries) + 1):
        if count_probabilities[n] <= 0.0:
            continue
        draws = max(1, int(round(samples * count_probabilities[n])))
        inner = 0.0
        for _ in range(draws):
            mask = _draw_size_biased_subset(rng, weights, n)
            inner += _pd_for_mask(owner_delta, entries, mask, grid)
        estimate += count_probabilities[n] * (inner / draws)
    return float(estimate)


def hoeffding_upper_bound(
    estimate: float,
    samples: int,
    delta: float = 0.01,
) -> float:
    """Hoeffding one-sided confidence upper bound on the MC estimate.

    Each Monte Carlo draw of P_D lies in [0, 1], so with probability at
    least ``1 - delta`` the true expectation satisfies
    ``E[P_D] <= estimate + sqrt(ln(1/delta) / (2 * samples))``.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    return float(estimate) + sqrt(log(1.0 / delta) / (2.0 * samples))


def erasure_deterministic_upper_bound(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probabilities: np.ndarray,
    success_probabilities: np.ndarray,
    grid: int = 32,
    max_exact_reports: int = 8,
    samples: int = 2048,
) -> float:
    """Deterministic upper bound on the erasure expectation.

    By set-monotonicity of the fused P_D (reports add evidence, never
    remove it), the no-erasure value P_D(all reports received) dominates
    the value of every received subset and hence dominates the expectation
    over the independent erasure law.  The bound requires no Monte Carlo
    draws and holds with probability one.
    """
    success_probabilities = np.asarray(success_probabilities, dtype=float)
    all_received = np.ones(success_probabilities.size, dtype=float)
    return per_report_communication_target_pd(
        owner_delta,
        report_deltas,
        powers,
        bits,
        flip_probabilities,
        all_received,
        grid,
        max_exact_reports,
        samples,
    )


def count_conditional_upper_bound(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probabilities: np.ndarray,
    success_probabilities: np.ndarray,
    grid: int = 32,
    max_exact_reports: int = 8,
    samples: int = 2048,
) -> float:
    """Deterministic count-conditional upper bound on the erasure expectation.

    Write ``R`` for the number of active reports, ``P_D(S)`` for the fused
    detection probability of the received set ``S``, and ``N = |S|`` for the
    received count.  The erasure law is independent per report, so the count
    ``N`` follows the Poisson-binomial law ``c_n = P(N = n)`` of the
    per-report success probabilities, and

    ``E[P_D] = sum_n c_n E[P_D | N = n] <= sum_n c_n max_{|S| = n} P_D(S)``.

    The maximum over the ``n``-subsets is attained at the set of the ``n``
    strongest reports.  Proof: for independent reports (diagonal covariance,
    the model used throughout this module) with positive per-report shifts
    ``delta_j > 0`` (every report carries a target-present hypothesis),
    ``P_D(w) = Phi((w'delta - z sqrt(w'Sigma0 w)) / sqrt(w'Sigma1 w))`` is
    non-decreasing in each ``delta_j`` for every fixed direction ``w`` with
    nonnegative components, and the P_D-optimal direction has all-positive
    components (its whitened form ``y(mu) = (Q + mu I)^-1 a`` has ``a_j > 0``
    and diagonal ``Q``).  Hence the optimum ``P_D*(delta) = max_w P_D(w)``
    is non-decreasing in every report's strength, and the largest ``n``-set
    is the set of the ``n`` reports with the largest single-report
    deflection ``(m1 - m0)^2 / v0``.  Consequently the bound above is the
    tightest deterministic upper bound that depends only on the received
    count statistics, and it dominates the no-erasure value pointwise:

    ``count-conditional UB <= P_D(all reports)``

    with equality only when erasure is impossible or the fused P_D is
    already flat.  Below ``max_exact_reports`` all ``2^R`` masks are
    enumerated once, so the bound is computed at zero extra fusion cost;
    above it the ``R + 1`` count-ordered sets are evaluated directly.
    """
    entries = _build_entries(
        report_deltas, powers, bits,
        flip_probabilities, success_probabilities,
    )
    if len(entries) <= max_exact_reports:
        _, per_count_max, count_probabilities = _exact_erasure_marginalization(
            owner_delta, entries, grid
        )
        return float(sum(
            count_probabilities[n] * per_count_max[n]
            for n in range(len(entries) + 1)
        ))
    ordered = sorted(
        entries,
        key=lambda entry: (float(entry[1]) - float(entry[0])) ** 2
        / max(float(entry[2]), 1e-30),
        reverse=True,
    )
    count_probabilities = _poisson_binomial(np.asarray([
        float(success) for _, _, _, _, success in ordered
    ]))
    total = 0.0
    for n in range(len(ordered) + 1):
        if count_probabilities[n] <= 0.0:
            continue
        mask = (1 << n) - 1
        total += count_probabilities[n] * _pd_for_mask(
            owner_delta, ordered, mask, grid
        )
    return float(total)


def communication_pd_with_upper_bound(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probabilities: np.ndarray,
    success_probabilities: np.ndarray,
    grid: int = 32,
    max_exact_reports: int = 8,
    samples: int = 2048,
    delta: float = 0.01,
    rng=None,
) -> dict:
    """Expected P_D with deterministic and Hoeffding upper bounds.

    Returns ``estimate`` (the exact marginalization below
    ``max_exact_reports``, the count-stratified Rao-Blackwell mean above
    it), ``deterministic_ub`` (the no-erasure value, a.s. valid),
    ``count_conditional_ub`` (the count-conditional bound of
    :func:`count_conditional_upper_bound`, also a.s. valid and pointwise
    no looser than ``deterministic_ub``), and ``hoeffding_ub`` (valid with
    probability at least ``1 - delta``).  When the exact branch runs the
    Hoeffding bound is exact by construction (zero estimation error).
    """
    estimate = per_report_communication_target_pd(
        owner_delta,
        report_deltas,
        powers,
        bits,
        flip_probabilities,
        success_probabilities,
        grid,
        max_exact_reports,
        samples,
        rng,
    )
    deterministic_ub = erasure_deterministic_upper_bound(
        owner_delta,
        report_deltas,
        powers,
        bits,
        flip_probabilities,
        success_probabilities,
        grid,
        max_exact_reports,
        samples,
    )
    count_conditional_ub = count_conditional_upper_bound(
        owner_delta,
        report_deltas,
        powers,
        bits,
        flip_probabilities,
        success_probabilities,
        grid,
        max_exact_reports,
        samples,
    )
    return {
        "estimate": float(estimate),
        "deterministic_ub": float(deterministic_ub),
        "count_conditional_ub": float(count_conditional_ub),
        "hoeffding_ub": float(hoeffding_upper_bound(
            estimate, samples, delta
        )),
    }


def per_report_communication_target_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probabilities: np.ndarray,
    success_probabilities: np.ndarray,
    grid: int = 32,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng=None,
) -> float:
    """Expected P_D with per-report BSC flip and erasure."""
    return _expected_communication_pd(
        owner_delta,
        report_deltas,
        powers,
        bits,
        np.asarray(flip_probabilities, dtype=float),
        np.asarray(success_probabilities, dtype=float),
        grid,
        max_exact_reports=max_exact_reports,
        samples=samples,
        rng=rng,
    )


def enumerate_robust_power_bit_options(
    owner_delta: float,
    report_deltas: np.ndarray,
    *,
    power_levels: np.ndarray,
    bit_options: np.ndarray,
    budget: int,
    flip_interval: tuple[float, float],
    success_interval: tuple[float, float],
    power_cost: float = 1.0,
    bit_cost: float = 1.0,
    grid: int = 32,
) -> list[JointPowerBitOption]:
    """Enumerate joint options with clean and worst-endpoint P_D."""
    deltas = np.asarray(report_deltas, dtype=float)
    flip_lo, flip_hi = flip_interval
    success_lo, success_hi = success_interval
    per_report_choices = list(itertools.product(power_levels, bit_options))
    options = []
    for combo in itertools.product(
        per_report_choices, repeat=deltas.size
    ):
        powers = np.asarray([item[0] for item in combo], dtype=float)
        bits = np.asarray([item[1] for item in combo], dtype=int)
        cost = int(round(
            power_cost * float(powers.sum()) + bit_cost * float(bits.sum())
        ))
        if cost > budget:
            continue
        clean_pd = _target_pd(
            owner_delta, deltas, powers, bits, flip_lo, success_hi, grid
        )
        robust_pd = _target_pd(
            owner_delta, deltas, powers, bits, flip_hi, success_lo, grid
        )
        options.append(JointPowerBitOption(
            cost_bits=cost,
            powers=tuple(float(value) for value in powers),
            bits=tuple(int(value) for value in bits),
            clean_pd=float(clean_pd),
            robust_pd=float(robust_pd),
        ))
    return options


def enumerate_heterogeneous_robust_power_bit_options(
    owner_delta: float,
    report_deltas: np.ndarray,
    flip_intervals: list[tuple[float, float]],
    success_intervals: list[tuple[float, float]],
    *,
    power_levels: np.ndarray,
    bit_options: np.ndarray,
    budget: int,
    power_cost: float = 1.0,
    bit_cost: float = 1.0,
    grid: int = 32,
) -> list[JointPowerBitOption]:
    """Enumerate joint options with per-report communication intervals."""
    deltas = np.asarray(report_deltas, dtype=float)
    flip_lo = np.asarray([item[0] for item in flip_intervals], dtype=float)
    flip_hi = np.asarray([item[1] for item in flip_intervals], dtype=float)
    success_lo = np.asarray(
        [item[0] for item in success_intervals], dtype=float
    )
    success_hi = np.asarray(
        [item[1] for item in success_intervals], dtype=float
    )
    per_report_choices = list(itertools.product(power_levels, bit_options))
    options = []
    for combo in itertools.product(
        per_report_choices, repeat=deltas.size
    ):
        powers = np.asarray([item[0] for item in combo], dtype=float)
        bits = np.asarray([item[1] for item in combo], dtype=int)
        cost = int(round(
            power_cost * float(powers.sum()) + bit_cost * float(bits.sum())
        ))
        if cost > budget:
            continue
        clean_pd = per_report_communication_target_pd(
            owner_delta,
            deltas,
            powers,
            bits,
            flip_lo,
            success_hi,
            grid,
        )
        robust_pd = per_report_communication_target_pd(
            owner_delta,
            deltas,
            powers,
            bits,
            flip_hi,
            success_lo,
            grid,
        )
        options.append(JointPowerBitOption(
            cost_bits=cost,
            powers=tuple(float(value) for value in powers),
            bits=tuple(int(value) for value in bits),
            clean_pd=float(clean_pd),
            robust_pd=float(robust_pd),
        ))
    return options


def pareto_options(
    options: list[JointPowerBitOption],
    value_field: str,
) -> list[tuple[int, float]]:
    result = []
    best_value = -1.0
    last_cost = None
    for option in sorted(options, key=lambda item: (item.cost_bits, -getattr(item, value_field))):
        value = float(getattr(option, value_field))
        if option.cost_bits == last_cost:
            continue
        last_cost = option.cost_bits
        if value > best_value + 1e-12:
            result.append((option.cost_bits, value))
            best_value = value
    return result
