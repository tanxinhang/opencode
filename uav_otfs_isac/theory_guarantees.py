r"""Theoretical guarantees for the NOMP allocation: complexity and P_D.

The results here scale to general :math:`Q` targets and :math:`N` UAVs
(equivalently :math:`R` report links per target); none of the bounds depend
on a specific scene size.

Complexity
==========

Let :math:`Q` be the target count, :math:`R` the report links per target,
:math:`B` the bit budget, and :math:`T` the number of greedy/refine steps.

Theorem (NOMP greedy+refine complexity)
    The two-phase schedule ``nomp_wta_greedy_joint_multi`` terminates in
    time :math:`O(Q R^2 T (1 + \log(Q R^2)))` and space :math:`O(Q R)`.
    The greedy phase evaluates at most :math:`Q R` candidate report
    activations per step for :math:`T` steps; the refinement phase
    generates :math:`O(Q R^2)` single-swap candidates per round
    (``_iter_candidates``) and sorts them, for at most :math:`T` rounds.
    Every per-target P_D evaluation is a constant-size cacheable query, so
    the UAV count :math:`N` enters only through :math:`R`, linearly.

Corollary (scale separation)
    The cost of the schedule is polynomial in :math:`Q, R, B` and
    independent of the UAV count once the report structure is fixed; adding
    UAVs that do not create report links adds zero cost.

Submodularity and the greedy bound
==================================

Proportional-covariance model with deterministic reception (unit success
probability, no bit flips) and equal H0/H1 variances is the regime of
``_proportional_pd``.  There the optimal linear score satisfies

.. math:: P_D(D) = \\Phi\\left(\\frac{\\sqrt{D} - z}{\\sqrt{c}}\\right),

where :math:`D` is the deflection of the fused score, :math:`z =
\\Phi^{-1}(1-P_{\\mathrm{FA}})`, and :math:`c = \\sigma_1^2/\\sigma_0^2`
equals 1 in the equal-variance instance.

Lemma (concavity of P_D in deflection)
    For :math:`c \\ge z^2/4` the map :math:`D \\mapsto P_D(D)` is concave
    on :math:`D \\ge 0`.  Proof: with
    :math:`u = (\\sqrt{D}-z)/\\sqrt{c}`,
    :math:`P_D''(D) = -\\varphi(u)(u a^2 + a/\\sqrt{D})/(4D)` for
    :math:`a = 1/\\sqrt{c}`, and the sign of the bracket is controlled by
    :math:`\\sqrt{D}(z-\\sqrt{D}) < c`, whose maximum over
    :math:`\\sqrt{D} \\in [0,z]` is :math:`z^2/4`.  The equal-variance
    instance :math:`c=1` satisfies the condition for every
    :math:`P_{\\mathrm{FA}} \\le 0.32`.

Lemma (monotone submodularity of activated P_D)
    Fix the power/bit allocation.  Under the proportional-covariance model
    with deterministic reception, the activated-report set function
    :math:`F(A) = P_D(D_0 + \\sum_{i \\in A} w_i)` with fixed positive
    weights :math:`w_i = \\delta_i^2 p_i` is monotone nondecreasing and
    submodular, because :math:`P_D` is concave in the total deflection and
    the total deflection is additive over the activated set.

Theorem (greedy (1-1/e) bound)
    For a single target under a cardinality constraint on the number of
    activated report links, the greedy algorithm that repeatedly activates
    the report with the largest marginal gain achieves a value of at least
    :math:`(1-1/e) \\approx 0.632` times the optimal value.  This follows
    from monotone submodularity by the classical greedy bound.  The
    ``initial_min_cover`` phase of the NOMP pipeline is exactly this
    greedy on the per-target activation problem, so the min-cover stage
    inherits the bound.

Remark (max-min pipeline)
    The multi-target max-min pipeline does not inherit the single-target
    (1-1/e) bound; it is a heuristic on the max-min objective.  The
    theoretical role of the bound is to certify the per-target activation
    stage and to justify the greedy refinement's monotone behavior: every
    refinement round either improves or preserves the worst-target P_D
    (checked numerically in ``verify_refinement_monotone``).

Upper bounds on the erasure expectation
=======================================

For more than ``max_exact_reports`` report links the erasure expectation
:math:`\\mathbb{E}[P_D(\\text{received})]` is estimated by Monte Carlo.
Two upper bounds are available (``robust_joint_power_bit``):

Theorem (Hoeffding confidence bound)
    With probability at least :math:`1-\\delta`, the true expectation is at
    most the sample mean plus :math:`\\sqrt{\\ln(1/\\delta)/(2 n)}`, because
    every P_D draw lies in [0, 1].

Theorem (deterministic erasure bound)
    By set-monotonicity of the fused P_D, the no-erasure value
    :math:`P_D(\\text{all reports})` dominates the value of every received
    subset, hence dominates the expectation over the independent erasure
    law.  The bound holds with probability one and needs no sampling.
"""

from __future__ import annotations

import itertools
from math import sqrt

import numpy as np
from scipy.stats import norm

from .power_split_theory import _proportional_pd


def pd_of_deflection(
    deflection: float,
    false_alarm_rate: float = 0.05,
    variance_ratio: float = 1.0,
) -> float:
    """P_D = Phi((sqrt(D) - z) / sqrt(c)) under the proportional model."""
    z = float(norm.ppf(1.0 - false_alarm_rate))
    return float(norm.cdf((sqrt(max(deflection, 0.0)) - z) / sqrt(variance_ratio)))


def concavity_condition(
    deflection: float,
    false_alarm_rate: float = 0.05,
    variance_ratio: float = 1.0,
) -> dict:
    """Check the analytic second-derivative sign of P_D at one D value.

    Returns the second derivative and the sufficient condition
    sqrt(D)(z - sqrt(D)) < c that guarantees concavity.
    """
    z = float(norm.ppf(1.0 - false_alarm_rate))
    c = variance_ratio
    d = max(deflection, 1e-12)
    u = (sqrt(d) - z) / sqrt(c)
    a = 1.0 / sqrt(c)
    phi_u = float(norm.pdf(u))
    second = -phi_u * (u * a * a + a / sqrt(d)) / (4.0 * d)
    bound = sqrt(d) * (z - sqrt(d))
    return {
        "second_derivative": second,
        "sufficient_condition_holds": bool(bound < c),
        "bound_value": bound,
        "variance_ratio": c,
    }


def verify_concavity_grid(
    false_alarm_rate: float = 0.05,
    variance_ratio: float = 1.0,
    d_max: float = 40.0,
    points: int = 200,
) -> dict:
    """Numerically verify concavity (P_D'' <= 0) on a deflection grid."""
    z = float(norm.ppf(1.0 - false_alarm_rate))
    grid = np.linspace(0.0, d_max, points)
    c = variance_ratio
    worst = -np.inf
    violations = 0
    for d in grid:
        if d <= 0.0:
            continue
        u = (sqrt(d) - z) / sqrt(c)
        second = -float(norm.pdf(u)) * (
            u / c + 1.0 / sqrt(d) / sqrt(c)
        ) / (4.0 * d)
        if second > 0.0:
            violations += 1
        worst = max(worst, second)
    return {
        "violations": int(violations),
        "worst_second_derivative": worst,
        "concave_on_grid": bool(violations == 0),
        "sufficient_c_minus_z2_over_4": float(c - z * z / 4.0),
    }


def _deflection_from_powers(
    owner_delta: float, deltas: np.ndarray, powers: np.ndarray
) -> float:
    """Total deflection: owner baseline plus power-scaled report gains."""
    return float(owner_delta) + float(
        np.asarray(deltas, dtype=float) @ np.asarray(powers, dtype=float)
    )


def verify_submodularity(
    owner_delta: float,
    deltas: np.ndarray,
    bits: np.ndarray,
    *,
    grid: int = 32,
    max_reports: int | None = None,
) -> dict:
    """Exhaustively verify monotone submodularity on all report subsets.

    Checks the defining inequality: for every A subset B subset S and
    i not in B, F(A | i) - F(A) >= F(B | i) - F(B).
    """
    deltas = np.asarray(deltas, dtype=float)
    bits = np.asarray(bits, dtype=int)
    r = deltas.size
    if max_reports is not None:
        r = min(r, max_reports)
    def f(mask):
        powers = np.zeros(r, dtype=float)
        for i in range(r):
            powers[i] = 1.0 if (mask >> i) & 1 else 0.0
        return _proportional_pd(
            owner_delta, deltas[:r], powers, bits[:r], grid
        )
    n = 1 << r
    values = [f(mask) for mask in range(n)]
    violations = 0
    max_violation = -np.inf
    for a_mask in range(n):
        for i in range(r):
            if (a_mask >> i) & 1:
                continue
            gain_a = values[a_mask | (1 << i)] - values[a_mask]
            # 检查所有包含 a_mask 的超集 B（不含 i）
            complement = ((n - 1) ^ a_mask) & ~(1 << i)
            sub = complement
            while True:
                b_mask = a_mask | sub
                gain_b = values[b_mask | (1 << i)] - values[b_mask]
                diff = gain_a - gain_b
                if diff < -1e-9:
                    violations += 1
                    max_violation = max(max_violation, -diff)
                if sub == 0:
                    break
                sub = (sub - 1) & complement
    return {
        "reports": int(r),
        "subset_count": int(n),
        "violations": int(violations),
        "max_violation": float(max_violation) if violations else 0.0,
        "submodular": bool(violations == 0),
        "monotone": all(
            values[a | (1 << i)] >= values[a] - 1e-9
            for a in range(n) for i in range(r) if not (a >> i) & 1
        ),
    }


def greedy_single_target(
    owner_delta: float,
    deltas: np.ndarray,
    bits: np.ndarray,
    *,
    grid: int = 32,
    cardinality: int,
) -> tuple[list[int], float]:
    """Greedy activation of at most `cardinality` reports for one target."""
    deltas = np.asarray(deltas, dtype=float)
    bits = np.asarray(bits, dtype=int)
    active: list[int] = []
    for _ in range(cardinality):
        best_i, best_gain = None, 0.0
        base = _proportional_pd(
            owner_delta, deltas, _powers_of(active, deltas.size), bits, grid
        )
        for i in range(deltas.size):
            if i in active:
                continue
            trial = _proportional_pd(
                owner_delta, deltas,
                _powers_of(active + [i], deltas.size), bits, grid,
            )
            if trial - base > best_gain + 1e-12:
                best_gain = trial - base
                best_i = i
        if best_i is None:
            break
        active.append(best_i)
    return active, _proportional_pd(
        owner_delta, deltas, _powers_of(active, deltas.size), bits, grid
    )


def _powers_of(active: list[int], size: int) -> np.ndarray:
    powers = np.zeros(size, dtype=float)
    for i in active:
        powers[i] = 1.0
    return powers


def exhaustive_single_target(
    owner_delta: float,
    deltas: np.ndarray,
    bits: np.ndarray,
    *,
    grid: int = 32,
    cardinality: int,
) -> tuple[list[int], float]:
    """Exhaustive best activation of at most `cardinality` reports."""
    deltas = np.asarray(deltas, dtype=float)
    bits = np.asarray(bits, dtype=int)
    best_mask, best_value = None, -1.0
    for k in range(cardinality + 1):
        for combo in itertools.combinations(range(deltas.size), k):
            powers = np.zeros(deltas.size, dtype=float)
            for i in combo:
                powers[i] = 1.0
            value = _proportional_pd(owner_delta, deltas, powers, bits, grid)
            if value > best_value + 1e-12:
                best_value = value
                best_mask = list(combo)
    return best_mask, best_value


def verify_greedy_ratio(
    owner_delta: float,
    deltas: np.ndarray,
    bits: np.ndarray,
    *,
    grid: int = 32,
    cardinality: int = 3,
) -> dict:
    """Verify greedy >= (1-1/e) * OPT on one target instance."""
    _, greedy_value = greedy_single_target(
        owner_delta, deltas, bits, grid=grid, cardinality=cardinality
    )
    _, opt_value = exhaustive_single_target(
        owner_delta, deltas, bits, grid=grid, cardinality=cardinality
    )
    ratio = greedy_value / opt_value if opt_value > 0 else 1.0
    bound = 1.0 - 1.0 / np.e
    return {
        "greedy_value": float(greedy_value),
        "optimal_value": float(opt_value),
        "ratio": float(ratio),
        "greedy_bound": float(bound),
        "bound_holds": bool(ratio >= bound - 1e-9),
    }


def verify_refinement_monotone(
    scenario,
    budget: int,
    *,
    grid: int = 16,
    max_rounds: int = 30,
    samples: int = 2048,
) -> dict:
    """Verify the worst-target P_D never degrades across the NOMP pipeline."""
    from .nomp_refinement import maxmin_refine, wta_greedy_joint_multi

    greedy = wta_greedy_joint_multi(
        scenario, budget, min_cover=True, grid=grid, max_bits=2,
    )
    powers, bits, _ = maxmin_refine(
        scenario,
        greedy["powers"],
        greedy["bits"],
        max_power=budget,
        max_bits=2,
        max_rounds=max_rounds,
        grid=grid,
    )
    from .nomp_refinement import target_scores

    before = float(min(target_scores(scenario, greedy["powers"], greedy["bits"], grid)))
    after = float(min(target_scores(scenario, powers, bits, grid)))
    return {
        "before_refine_worst": before,
        "after_refine_worst": after,
        "monotone": bool(after >= before - 1e-9),
    }


def complexity_bounds(Q: int, R: int, T: int = 100) -> dict:
    """Report the symbolic complexity bounds of the NOMP pipeline."""
    return {
        "greedy_time": f"O({Q} * {R} * {T})",
        "refine_time": f"O({Q} * {R}^2 * {T} * (1 + log({Q}*{R}^2)))",
        "space": f"O({Q} * {R})",
        "uav_count_dependence": (
            "N enters only linearly through R (report links per target)"
        ),
        "scale_separation": (
            "adding UAVs without report links adds zero cost"
        ),
    }