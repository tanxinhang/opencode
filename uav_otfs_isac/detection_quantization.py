"""Detection-aware quantization and information-gradient allocation.

This module turns the detection information of ``detection_information.py``
into *design tools* for the two remaining advice items: the detection-aware
quantizer ``Q_b* = argmax KL`` and the information gradient
``dI+/d(bit)`` used for budget allocation.

Provable facts used throughout (all restated with their argument):

1. **Information monotone in bits.**  The system's quantizer family refines
   uniformly with ``bits`` (same ``[low, high]`` range, halved steps), so the
   received level of the ``b``-bit link is the ``b+1``-bit received level with
   its LSB dropped -- a deterministic function.  By data processing,

       I+(b) <= I+(b+1),   b = 0, 1, 2, ...

2. **Information monotone in channel quality.**  ``BSC(p2)`` is
   ``BSC(p1)`` followed by ``BSC((p2-p1)/(1-2p1))`` (Theorem 4.59), and the
   erasure operation is a deterministic function of the success event, so
   ``I+`` is nonincreasing in the flip probability and nondecreasing in the
   success probability.

3. **LLR structure of the 1-bit optimum.**  With ``var1 > var0`` (the
   physical case here: variance grows with the noncentrality), the Gaussian
   log-likelihood ratio is strictly convex and tends to ``-inf`` at both
   ends, so its superlevel sets are *two-sided* windows ``{x < a} U {x > b}``
   with ``a < b``; with ``var1 < var0`` they are single intervals.  The
   two-sided window is therefore the canonical 1-bit detection region for
   the system's physics, and it is compared against the single-threshold
   family numerically (exact KL).

4. **Design metric hierarchy (verified here, not assumed).**  ``I+`` is the
   *mean* LLR drift; across *different quantizer designs* the mean drift
   ranks designs wrongly, because it ignores the LLR atom structure that
   governs threshold crossing (a coarse quantizer concentrates H0 mass and
   inflates KL while degrading the actual error probability).  The Chernoff
   information -- the full-distribution error exponent -- tracks the exact
   ``P_D(n)`` ranking, and the exact ``P_D(n)`` is the ground truth.  The
   gate therefore compares all three metrics on the span design knob.

5. **Greedy quality without concavity.**  ``I+(b)`` is *not* concave in
   general (margins can jump when the step size resolves the H0 bulk), so
   water-filling is not claimed exact; the gate measures the greedy gap
   against exhaustive search (small instances, <= 5% in testing).

Everything is exact: KL divergences and BSC propagation use the same
machinery as ``detection_information.py``, and the bit/span/threshold
sweeps are dense on the design axis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .detection_information import (
    post_communication_likelihoods,
    sequential_pd,
)
from .reporting import bsc_transition, quantizer_from_gaussian_range


def quantizer_edges(mu0, var0, mu1, var1, bits, span_std=4.0):
    """Uniform-refinement quantizer of the system's design family."""
    low = min(mu0 - span_std * np.sqrt(var0), mu1 - span_std * np.sqrt(var1))
    high = max(mu0 + span_std * np.sqrt(var0), mu1 + span_std * np.sqrt(var1))
    boundaries = np.linspace(low, high, 2**bits + 1)
    edges = np.concatenate(([-np.inf], boundaries[1:-1], [np.inf]))
    values = 0.5 * (boundaries[:-1] + boundaries[1:])
    return edges, values


def link_information_vs_bits(
    mu0: float,
    var0: float,
    mu1: float,
    var1: float,
    flip_probability: float,
    success_probability: float,
    bits_max: int = 6,
    span_std: float = 4.0,
) -> NDArray[np.float64]:
    """Exact ``I+(b)`` for ``b = 0..bits_max`` of the uniform-refinement
    quantizer family (the system's design family: same range, halved steps).
    ``I+(0) = 0`` (a single bin carries no discrimination power).
    """
    bits_max = int(bits_max)
    if bits_max < 0:
        raise ValueError("bits_max must be nonnegative")
    out = np.zeros(bits_max + 1, dtype=float)
    for bits in range(1, bits_max + 1):
        edges, values = quantizer_edges(
            mu0, var0, mu1, var1, bits, span_std,
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values,
            bits, flip_probability, success_probability,
        )
        out[bits] = float(info["kl_plus"])
    return out


def verify_bits_monotonicity(
    instances: list[tuple[float, float, float, float, float, float]],
    bits_max: int = 6,
) -> dict:
    """Checks ``I+(b) <= I+(b+1)`` (data-processing argument, fact 1)."""
    violations = []
    for idx, (mu0, var0, mu1, var1, flip, success) in enumerate(instances):
        profile = link_information_vs_bits(
            mu0, var0, mu1, var1, flip, success, bits_max,
        )
        for b in range(bits_max):
            if profile[b] > profile[b + 1] + 1e-12:
                violations.append({
                    "instance": idx, "bits": b,
                    "i_plus_b": float(profile[b]),
                    "i_plus_b1": float(profile[b + 1]),
                })
    return {"passed": not violations, "violations": violations}


def verify_flip_monotonicity(
    instances: list[tuple[float, float, float, float, float, float]],
    bits: int = 3,
    flips: tuple[float, float] = (0.02, 0.15),
    grid: int = 7,
) -> dict:
    """Checks ``I+`` nonincreasing in the BSC flip probability
    (cascade identity Theorem 4.59 + data processing, fact 2)."""
    violations = []
    flip_values = np.linspace(flips[0], flips[1], grid)
    for idx, (mu0, var0, mu1, var1, _, success) in enumerate(instances):
        values = [
            link_information_vs_bits(mu0, var0, mu1, var1, f, success, bits)[bits]
            for f in flip_values
        ]
        for a, b in zip(values, values[1:]):
            if a < b - 1e-12:
                violations.append({"instance": idx, "flip_from": float(a),
                                   "flip_to": float(b)})
    return {"passed": not violations, "violations": violations}


def verify_success_monotonicity(
    instances: list[tuple[float, float, float, float, float, float]],
    bits: int = 3,
    grid: int = 7,
) -> dict:
    """Checks ``I+`` nondecreasing in the success probability."""
    violations = []
    success_values = np.linspace(0.5, 0.99, grid)
    for idx, (mu0, var0, mu1, var1, flip, _) in enumerate(instances):
        values = [
            link_information_vs_bits(mu0, var0, mu1, var1, flip, s, bits)[bits]
            for s in success_values
        ]
        for a, b in zip(values, values[1:]):
            if a > b + 1e-12:
                violations.append({"instance": idx, "success_from": float(a),
                                   "success_to": float(b)})
    return {"passed": not violations, "violations": violations}


def option_metric_vs_bits(
    mu0: float,
    var0: float,
    mu1: float,
    var1: float,
    flip_probability: float,
    success_probability: float,
    bits_max: int = 6,
    span_std: float = 4.0,
    metric: str = "pd",
    n: int = 4,
    alpha: float = 0.05,
    grid_step: float = 0.05,
) -> NDArray[np.float64]:
    """Exact per-option metric curve over ``bits = 0..bits_max``.

    ``metric="pd"`` returns the exact terminal ``P_D(n)`` of the NP test
    (the allocation ground truth), ``"chernoff"`` the Chernoff information
    (the verified proxy), ``"i_plus"`` the KL mean drift.  ``bits = 0``
    (option unused) is ``alpha`` for the PD curve (no observation means the
    trivial test) and ``0.0`` for the information metrics.
    """
    if metric not in ("pd", "chernoff", "i_plus"):
        raise ValueError("metric must be 'pd', 'chernoff' or 'i_plus'")
    bits_max = int(bits_max)
    if bits_max < 0:
        raise ValueError("bits_max must be nonnegative")
    out = np.zeros(bits_max + 1, dtype=float)
    out[0] = alpha if metric == "pd" else 0.0
    for bits in range(1, bits_max + 1):
        edges, values = quantizer_edges(
            mu0, var0, mu1, var1, bits, span_std,
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values,
            bits, flip_probability, success_probability,
        )
        if metric == "pd":
            out[bits] = float(sequential_pd(
                info["p1_y"], info["p0_y"], n, alpha, grid_step,
            )["pd"])
        elif metric == "chernoff":
            out[bits] = float(info["chernoff"])
        else:
            out[bits] = float(info["kl_plus"])
    return out


def verify_pd_bits_monotonicity(
    instances: list[tuple[float, float, float, float, float, float]],
    bits_max: int = 6,
    n: int = 4,
    alpha: float = 0.05,
    grid_step: float = 0.05,
) -> dict:
    """Checks ``P_D(b) <= P_D(b+1)`` for the refinement chain.

    The theorem is exact for the true likelihood-ratio statistic: the
    ``b``-bit observation is a deterministic function (LSB drop) of the
    ``b+1``-bit one, so any size-``alpha`` test on ``b`` bits is a valid
    test on ``b+1`` bits, and the NP likelihood-ratio test on the finer
    alphabet is most powerful among size-``alpha`` tests (admissibility).

    The grid implementation evaluates the accumulated LLR on a binned
    grid (``rint`` at the atom and accumulation level), a slightly
    suboptimal statistic, so small violations can occur (measured
    ``<= 0.008`` on the test family, always with the finer test running a
    more conservative P_FA).  This diagnostic therefore reports the
    violations and their magnitude instead of claiming exact monotonicity
    for the binned statistic.  The floor-cover allocation
    (``maxmin_pd_allocation``) does not rely on monotonicity and stays
    exact.
    """
    violations = []
    max_violation = 0.0
    for idx, (mu0, var0, mu1, var1, flip, success) in enumerate(instances):
        curve = option_metric_vs_bits(
            mu0, var0, mu1, var1, flip, success, bits_max,
            metric="pd", n=n, alpha=alpha, grid_step=grid_step,
        )
        for b in range(bits_max):
            if curve[b] > curve[b + 1] + 1e-12:
                max_violation = max(max_violation, curve[b] - curve[b + 1])
                violations.append({
                    "instance": idx, "bits": b,
                    "pd_b": float(curve[b]), "pd_b1": float(curve[b + 1]),
                })
    return {"passed": not violations, "violations": violations,
            "max_violation": float(max_violation)}


def maxmin_pd_allocation(
    target_options: list[list[dict]],
    budget: int,
    metric: str = "pd",
    n: int = 4,
    alpha: float = 0.05,
    grid_step: float = 0.05,
) -> dict:
    """Exact max-min allocation over the worst-target terminal ``P_D(n)``.

    ``target_options`` is one list per target, each entry a report/power
    option dict with ``(mu0, var0, mu1, var1, flip, success)`` (and optional
    ``bits_max``, ``span_std``).  ``metric`` selects the curve the floor is
    raised on: ``"pd"`` (exact ground truth), ``"chernoff"`` (verified
    proxy), ``"i_plus"`` (comparison only).

    Floor-cover theorem (no concavity assumption): for candidate level
    ``L``, the minimal budget target ``t`` needs to reach ``L`` on at least
    one of its options is

        c_t(L) = min_o min{ b >= 1 : curve_{t,o}(b) >= L },

    and ``max_{b : sum b <= B} min_t max_o curve_{t,o}(b)`` equals the
    largest ``L`` in the union of all curve values with ``sum_t c_t(L) <= B``
    (each target must spend ``c_t(L)`` to reach ``L``, and spending exactly
    ``c_t(L)`` reaches it).  The optimum is unique in value; leftover budget
    is given to the currently lowest target (monotone curves only raise
    floors further).  The result is exact even when curves are not monotone.
    """
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    curves = []
    for t, options in enumerate(target_options):
        t_curves = []
        for o in options:
            t_curves.append(option_metric_vs_bits(
                o["mu0"], o["var0"], o["mu1"], o["var1"],
                o["flip"], o["success"], int(o.get("bits_max", budget)),
                float(o.get("span_std", 4.0)),
                metric=metric, n=n, alpha=alpha, grid_step=grid_step,
            ))
        curves.append(t_curves)

    def cost(level: float) -> list[int]:
        per_target = []
        for t_curves in curves:
            best = None
            for curve in t_curves:
                # min over ALL b (curves are monotone in the verified
                # family, but the characterization is exact regardless)
                for b in range(1, len(curve)):
                    if curve[b] >= level:
                        if best is None or b < best:
                            best = b
            per_target.append(best)
        return per_target

    levels = sorted({float(v) for t_curves in curves for curve in t_curves
                     for v in curve[1:]}, reverse=True)
    best_level = alpha
    best_cost = None
    for level in levels:
        c = cost(level)
        if all(x is not None for x in c) and sum(c) <= budget:
            best_level = level
            best_cost = c
            break
    bits = [[0] * len(t_curves) for t_curves in curves]
    achieved = []
    for t, t_curves in enumerate(curves):
        if best_cost is None:
            achieved.append(max(float(curve.max()) for curve in t_curves))
            continue
        found = False
        for oi, curve in enumerate(t_curves):
            if best_cost[t] is not None and best_cost[t] < len(curve) \
                    and curve[best_cost[t]] >= best_level:
                bits[t][oi] = best_cost[t]
                found = True
                break
        if not found:
            oi = int(np.argmax([float(curve.max()) for curve in t_curves]))
            bits[t][oi] = int(np.argmax(t_curves[oi]))
        achieved.append(float(t_curves[oi][bits[t][oi]]))
    used = sum(sum(row) for row in bits)
    while used < budget:
        t = int(np.argmin(achieved))
        candidates = [(ti, oi) for ti in range(len(curves))
                      for oi in range(len(curves[ti]))
                      if bits[ti][oi] < len(curves[ti][oi]) - 1]
        if not candidates:
            break
        ti, oi = min(candidates, key=lambda p: float(curves[p[0]][p[1]][bits[p[0]][p[1]]]))
        bits[ti][oi] += 1
        achieved[ti] = max(achieved[ti],
                           float(curves[ti][oi][bits[ti][oi]]))
        used += 1
    return {
        "levels": best_level,
        "worst_metric": float(min(achieved)),
        "bits": bits,
        "achieved": achieved,
        "cost": sum(sum(row) for row in bits),
    }


def verify_bits_concavity(
    instances: list[tuple[float, float, float, float, float, float]],
    bits_max: int = 6,
) -> dict:
    """Checks diminishing returns ``I+(b) - I+(b-1)`` nonincreasing.

    This is the hypothesis of the water-filling theorem (fact 5); it is
    verified exactly here because the objective is discrete and exact.
    The check is expected to FAIL on this quantizer family (margins jump
    when the step size resolves the H0 bulk), which is why the greedy
    allocation is only claimed within a measured gap of the exhaustive
    optimum.
    """
    violations = []
    for idx, (mu0, var0, mu1, var1, flip, success) in enumerate(instances):
        profile = link_information_vs_bits(
            mu0, var0, mu1, var1, flip, success, bits_max,
        )
        margins = np.diff(profile)
        for b in range(1, len(margins)):
            if margins[b] > margins[b - 1] + 1e-12:
                violations.append({
                    "instance": idx, "bits": b,
                    "marginal_prev": float(margins[b - 1]),
                    "marginal": float(margins[b]),
                })
    return {"passed": not violations, "violations": violations}


def information_waterfilling(
    profiles: list[NDArray[np.float64]],
    budget: int,
    max_min: bool = False,
) -> tuple[list[int], float]:
    """Greedy bit allocation by marginal information.

    With ``max_min=False`` this maximizes ``sum_k I_k(b_k)`` (exact optimal
    by the separable-concave water-filling theorem when concavity holds).
    With ``max_min=True`` each unit goes to the currently smallest ``I_k``
    (egalitarian floor raising; the max-min-I+ allocation heuristic).
    """
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    n = len(profiles)
    bits = [0] * n
    i_plus = np.zeros(n, dtype=float)
    used = 0
    while used < budget:
        if max_min:
            k = int(np.argmin(i_plus))
            if bits[k] >= len(profiles[k]) - 1:
                # saturated: raise the next-smallest instead
                candidates = [i for i in range(n)
                              if bits[i] < len(profiles[i]) - 1]
                if not candidates:
                    break
                k = min(candidates, key=lambda i: i_plus[i])
        else:
            candidates = [i for i in range(n)
                          if bits[i] < len(profiles[i]) - 1]
            if not candidates:
                break
            k = max(candidates, key=lambda i: (
                profiles[i][bits[i] + 1] - profiles[i][bits[i]]
            ))
        bits[k] += 1
        i_plus[k] = profiles[k][bits[k]]
        used += 1
    return bits, float(i_plus.sum())


def information_vs_span(
    mu0: float,
    var0: float,
    mu1: float,
    var1: float,
    bits: int,
    flip_probability: float,
    success_probability: float,
    spans: NDArray[np.float64],
    metric: str = "chernoff",
) -> NDArray[np.float64]:
    """Design metric of the uniform quantizer vs its ``span_std`` knob.

    ``metric="chernoff"`` returns the Chernoff information (the design
    metric that tracks the exact ``P_D`` ranking; fact 4), ``"i_plus"``
    returns the KL mean drift (shown to misrank designs).  The system
    default span is ``4.0``.
    """
    if metric not in ("chernoff", "i_plus"):
        raise ValueError("metric must be 'chernoff' or 'i_plus'")
    out = np.zeros(len(spans), dtype=float)
    for i, span in enumerate(spans):
        edges, values = quantizer_edges(
            mu0, var0, mu1, var1, bits, float(span),
        )
        info = post_communication_likelihoods(
            mu0, var0, mu1, var1, edges, values,
            bits, flip_probability, success_probability,
        )
        out[i] = info["chernoff"] if metric == "chernoff" else info["kl_plus"]
    return out


def optimal_span(
    mu0: float,
    var0: float,
    mu1: float,
    var1: float,
    bits: int,
    flip_probability: float,
    success_probability: float,
    metric: str = "chernoff",
    span_grid: int = 81,
    span_range: tuple[float, float] = (1.0, 12.0),
) -> dict:
    """Optimal ``span_std`` for the uniform quantizer family.

    ``metric="chernoff"`` (default) is the verified design metric of the
    gate (fact 4: Chernoff tracks the exact ``P_D`` ranking, ``I+`` does
    not); ``metric="i_plus"`` returns the advice-style KL-optimal span for
    comparison only.  Returns both metric values at the optimum and at the
    4.0 system default, plus the relative gain over the default.
    """
    spans = np.linspace(span_range[0], span_range[1], span_grid)
    chernoff = information_vs_span(
        mu0, var0, mu1, var1, bits, flip_probability,
        success_probability, spans, metric="chernoff",
    )
    i_plus = information_vs_span(
        mu0, var0, mu1, var1, bits, flip_probability,
        success_probability, spans, metric="i_plus",
    )
    values = chernoff if metric == "chernoff" else i_plus
    best = int(np.argmax(values))
    j_default = int(np.argmin(np.abs(spans - 4.0)))
    return {
        "span_opt": float(spans[best]),
        "metric": metric,
        "metric_opt": float(values[best]),
        "metric_default": float(values[j_default]),
        "relative_gain_default": float(values[best] / max(values[j_default], 1e-12)
                                       - 1.0),
        "i_plus_opt": float(i_plus[best]),
        "i_plus_default": float(i_plus[j_default]),
        "chernoff_opt": float(chernoff[best]),
        "chernoff_default": float(chernoff[j_default]),
    }


def llr_quadratic(
    mu0: float, var0: float, mu1: float, var1: float, x
) -> NDArray[np.float64]:
    """Gaussian log-likelihood ratio as a function of ``x`` (quadratic).

    ``LLR(x) = log[p1(x)/p0(x)] = -0.5 (x-mu1)^2/var1 + 0.5 (x-mu0)^2/var0
    + 0.5 log(var0/var1)``.
    """
    x = np.asarray(x, dtype=float)
    return (
        -0.5 * ((x - mu1) ** 2) / var1
        + 0.5 * ((x - mu0) ** 2) / var0
        + 0.5 * np.log(var0 / var1)
    )


def llr_1bit_structure(
    mu0: float, var0: float, mu1: float, var1: float,
) -> dict:
    """Structural classification of the 1-bit LLR superlevel set.

    With ``var1 > var0`` the LLR is strictly convex and the superlevel set
    ``{x : LLR(x) >= l}`` is the two-sided window ``{x < a} U {x > b}``;
    with ``var1 < var0`` it is a single interval.  Returns the window or
    interval edges for the level ``l`` that makes the window symmetric
    around the LLR minimum/maximum (balanced operating point).
    """
    if var1 > var0:
        # convex, minimum at x*; balanced level l* = LLR(x*) + delta.
        # Superlevel set {LLR >= level}: roots of a x^2 + b x + (c - level) = 0
        # with a = 0.5/var0 - 0.5/var1 > 0, b = mu1/var1 - mu0/var0,
        # c = -0.5 mu1^2/var1 + 0.5 mu0^2/var0 + 0.5 log(var0/var1).
        x_star = (mu1 * var0 - mu0 * var1) / (var0 - var1)
        l_star = float(llr_quadratic(mu0, var0, mu1, var1, np.array([x_star]))[0])
        a = 0.5 / var0 - 0.5 / var1
        b = mu1 / var1 - mu0 / var0
        c = -0.5 * mu1**2 / var1 + 0.5 * mu0**2 / var0 + 0.5 * np.log(var0 / var1)
        roots = []
        for delta in (0.5, 1.0, 2.0):
            level = l_star + delta
            disc = b**2 - 4 * a * (c - level)
            if disc < 0:
                continue
            root = np.sqrt(disc)
            roots.append((float((-b - root) / (2 * a)),
                          float((-b + root) / (2 * a))))
        return {"kind": "two_sided_window", "x_star": float(x_star),
                "windows": roots}
    if var1 < var0:
        x_star = (mu1 * var0 - mu0 * var1) / (var0 - var1)
        return {"kind": "single_interval", "x_star": float(x_star)}
    return {"kind": "degenerate_equal_variance", "x_star": float((mu0 + mu1) / 2.0)}


def one_bit_kl_scan(
    mu0: float,
    var0: float,
    mu1: float,
    var1: float,
    flip_probability: float,
    success_probability: float,
    grid: int = 401,
) -> dict:
    """Exact 1-bit KL maximization over the single-threshold, one-sided,
    two-sided window and single-interval families (dense scan, exact
    discrete KL each).

    The LLR structure (``llr_1bit_structure``) identifies which family is
    canonical for the physics: ``var1 > var0`` -> two-sided window,
    ``var1 < var0`` -> single interval; both families are scanned so the
    comparison is never restricted to the wrong one.
    """
    low = min(mu0 - 8.0 * np.sqrt(var0), mu1 - 8.0 * np.sqrt(var1))
    high = max(mu0 + 8.0 * np.sqrt(var0), mu1 + 8.0 * np.sqrt(var1))
    xs = np.linspace(low, high, grid)
    n = len(xs)
    transition = bsc_transition(1, float(flip_probability))
    s = float(success_probability)

    def kl_of_bin_masses(m0: float, m1: float) -> float:
        # masses over the H1 bin; bin1 = (1-m0, 1-m1)
        p0_q = np.array([m0, 1.0 - m0])
        p1_q = np.array([m1, 1.0 - m1])
        p0_rec = p0_q @ transition
        p1_rec = p1_q @ transition
        p0_y = np.concatenate((s * p0_rec, np.array([1.0 - s])))
        p1_y = np.concatenate((s * p1_rec, np.array([1.0 - s])))
        return float(np.sum(p1_y * np.log(p1_y / p0_y)))

    std0 = np.sqrt(var0)
    std1 = np.sqrt(var1)
    # H0/H1 mass to the LEFT of each candidate threshold
    z0 = (xs - mu0) / std0
    z1 = (xs - mu1) / std1
    from scipy.stats import norm
    cdf0 = norm.cdf(z0)
    cdf1 = norm.cdf(z1)

    # single-threshold family: H1 bin = {x > tau}
    best_single = None
    best_value = -np.inf
    for i in range(n):
        m0 = cdf0[i]          # mass left of tau under H0
        m1 = cdf1[i]
        value = kl_of_bin_masses(m0, m1)
        if value > best_value:
            best_value = value
            best_single = (float(xs[i]), float(value))

    # two-sided window family {x < a} U {x > b}, plus the one-sided
    # windows {x > a} and {x < b} (the single-threshold family), so the
    # window optimum dominates the single-threshold optimum by construction.
    best_window = None
    best_window_value = -np.inf
    for i in range(n):
        for j in range(i + 2, n):
            m0 = cdf0[i] + (1.0 - cdf0[j])   # window mass under H0
            m1 = cdf1[i] + (1.0 - cdf1[j])
            value = kl_of_bin_masses(m0, m1)
            if value > best_window_value:
                best_window_value = value
                best_window = (float(xs[i]), float(xs[j]), float(value))
    for i in range(n):  # one-sided right window {x > a}
        m0 = 1.0 - cdf0[i]
        m1 = 1.0 - cdf1[i]
        value = kl_of_bin_masses(m0, m1)
        if value > best_window_value:
            best_window_value = value
            best_window = (float(xs[i]), float("inf"), float(value))
    for j in range(n):  # one-sided left window {x < b}
        m0 = cdf0[j]
        m1 = cdf1[j]
        value = kl_of_bin_masses(m0, m1)
        if value > best_window_value:
            best_window_value = value
            best_window = (float("-inf"), float(xs[j]), float(value))

    # single-interval family {a < x < b} (canonical when var1 < var0)
    best_interval = None
    best_interval_value = -np.inf
    for i in range(n):
        for j in range(i + 2, n):
            m0 = cdf0[j] - cdf0[i]
            m1 = cdf1[j] - cdf1[i]
            value = kl_of_bin_masses(m0, m1)
            if value > best_interval_value:
                best_interval_value = value
                best_interval = (float(xs[i]), float(xs[j]), float(value))

    structure = llr_1bit_structure(mu0, var0, mu1, var1)
    return {
        "best_single_threshold": best_single,
        "best_two_sided_window": best_window,
        "best_single_interval": best_interval,
        "window_gain_over_single": (
            best_window_value - best_value
        ),
        "llr_structure": structure,
    }


def allocate_by_information(
    links: list[dict],
    budget: int,
    max_min: bool = False,
) -> dict:
    """Per-target report-bit allocation from per-link ``I+`` profiles.

    ``links`` is a list of dicts with ``(mu0, var0, mu1, var1, flip,
    success, bits_max)``.  Returns the allocated bits per link, the
    achieved per-link ``I+`` and the totals.
    """
    profiles = [
        link_information_vs_bits(
            link["mu0"], link["var0"], link["mu1"], link["var1"],
            link["flip"], link["success"], link.get("bits_max", 6),
        )
        for link in links
    ]
    bits, total = information_waterfilling(profiles, budget, max_min)
    return {
        "bits": bits,
        "i_plus": [float(profiles[k][bits[k]]) for k in range(len(links))],
        "total_i_plus": float(total),
    }