"""BSC degradation ordering and exact likelihood-ratio ROC closure.

For ``0 <= p1 <= p2 <= 0.5``, the binary symmetric channel with crossover
``p2`` is a cascade of the channel with crossover ``p1`` followed by another
BSC with crossover ``q = (p2 - p1) / (1 - 2 p1)``.  Any decision rule on the
``p2`` output is therefore a randomized decision rule on the ``p1`` output,
so the achievable ROC under ``p1`` dominates the ROC under ``p2``.  The
functions here verify that ordering on the exact quantized likelihood-ratio
statistic used by the post-communication model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .reporting import (
    bsc_transition,
    gaussian_bin_probabilities,
    quantizer_from_gaussian_range,
)


def bsc_cascade_flip(lo: float, hi: float) -> float:
    """Return the second-stage flip q with BSC(hi) = BSC(lo) then BSC(q)."""
    lo = float(lo)
    hi = float(hi)
    if not 0.0 <= lo <= hi <= 0.5:
        raise ValueError("flip probabilities must satisfy 0 <= lo <= hi <= 0.5")
    if lo == 0.5:
        return 0.5
    return float((hi - lo) / (1.0 - 2.0 * lo))


def bsc_cascade_transition(bits: int, lo: float, hi: float) -> np.ndarray:
    """Transition matrix of BSC(lo) followed by BSC(q)."""
    q = bsc_cascade_flip(lo, hi)
    return bsc_transition(bits, lo) @ bsc_transition(bits, q)


def exact_lrt_roc_point(
    null_distribution: np.ndarray,
    alternative_distribution: np.ndarray,
    false_alarm_rate: float,
) -> tuple[float, float]:
    """Exact randomized-LRT P_D at a fixed false-alarm rate.

    The likelihood ratio is sorted descending and a single boundary atom is
    randomized to hit ``false_alarm_rate`` exactly, so the returned point
    lies on the exact ROC rather than a conservative threshold approximation.
    """
    null = np.asarray(null_distribution, dtype=float)
    alternative = np.asarray(alternative_distribution, dtype=float)
    if null.shape != alternative.shape or null.ndim != 1:
        raise ValueError("null and alternative distributions must be equal vectors")
    if np.any(null < 0.0) or np.any(alternative < 0.0):
        raise ValueError("distributions must be nonnegative")
    if not np.isclose(null.sum(), 1.0, atol=1e-10):
        raise ValueError("null distribution must sum to one")
    if not np.isclose(alternative.sum(), 1.0, atol=1e-10):
        raise ValueError("alternative distribution must sum to one")
    if not 0.0 <= false_alarm_rate <= 1.0:
        raise ValueError("false_alarm_rate must lie in [0, 1]")
    if false_alarm_rate == 0.0:
        return 0.0, 0.0
    if false_alarm_rate == 1.0:
        return 1.0, 1.0
    ratio = alternative / np.maximum(null, 1e-300)
    order = np.argsort(-ratio, kind="stable")
    null_sorted = null[order]
    alternative_sorted = alternative[order]
    cumulative_null = np.cumsum(null_sorted)
    cumulative_alternative = np.cumsum(alternative_sorted)
    index = int(np.searchsorted(cumulative_null, false_alarm_rate, side="right"))
    if index >= cumulative_null.size:
        return 1.0, 1.0
    if index == 0:
        previous_fa = 0.0
        previous_pd = 0.0
    else:
        previous_fa = float(cumulative_null[index - 1])
        previous_pd = float(cumulative_alternative[index - 1])
    next_fa = float(cumulative_null[index])
    next_pd = float(cumulative_alternative[index])
    mass = max(next_fa - previous_fa, 0.0)
    if mass <= 1e-15:
        return previous_pd, previous_fa
    weight = min(max((false_alarm_rate - previous_fa) / mass, 0.0), 1.0)
    return (
        previous_pd + weight * (next_pd - previous_pd),
        false_alarm_rate,
    )


def bsc_lrt_roc_point(
    mu0: float,
    variance0: float,
    mu1: float,
    variance1: float,
    bits: int,
    bit_flip_probability: float,
    false_alarm_rate: float,
) -> tuple[float, float]:
    """Exact quantized-Gaussian LRT point after a BSC."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    edges, _ = quantizer_from_gaussian_range(
        [mu0], [variance0], [mu1], [variance1], bits,
    )
    null = gaussian_bin_probabilities(mu0, variance0, edges)
    alternative = gaussian_bin_probabilities(mu1, variance1, edges)
    transition = bsc_transition(bits, bit_flip_probability)
    return exact_lrt_roc_point(
        null @ transition,
        alternative @ transition,
        false_alarm_rate,
    )


def verify_bsc_roc_dominance(
    *,
    bits_options: Sequence[int],
    mu0: float = 0.0,
    variance0: float = 1.0,
    mu1_options: Sequence[float],
    variance1: float = 1.0,
    lo_options: Sequence[float],
    hi_options: Sequence[float],
    false_alarm_grid: Sequence[float],
) -> dict:
    """Check exact-LRT P_D is nondecreasing as the BSC becomes cleaner."""
    if not (
        bits_options
        and mu1_options
        and lo_options
        and hi_options
        and false_alarm_grid
    ):
        raise ValueError("all BSC ROC option sequences must be nonempty")
    violations = []
    minimum_gap = float("inf")
    cells = 0
    for bits in bits_options:
        for mu1 in mu1_options:
            for lo in lo_options:
                for hi in hi_options:
                    if hi < lo:
                        continue
                    for pfa in false_alarm_grid:
                        pd_low, _ = bsc_lrt_roc_point(
                            mu0, variance0, mu1, variance1,
                            bits, lo, pfa,
                        )
                        pd_high, _ = bsc_lrt_roc_point(
                            mu0, variance0, mu1, variance1,
                            bits, hi, pfa,
                        )
                        cells += 1
                        minimum_gap = min(minimum_gap, pd_low - pd_high)
                        if pd_low < pd_high - 1e-10:
                            violations.append({
                                "bits": bits,
                                "mu1": mu1,
                                "lo": lo,
                                "hi": hi,
                                "pfa": pfa,
                                "pd_low": float(pd_low),
                                "pd_high": float(pd_high),
                            })
    return {
        "cells": cells,
        "minimum_pd_gap_clean_minus_degraded": float(minimum_gap),
        "violations": violations,
        "passed": not violations,
    }
