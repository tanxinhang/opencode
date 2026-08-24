"""P4.2 anytime-valid QoS certification tests (advice/015 section 2).

Tests the Beta-mixture time-uniform Bernoulli confidence sequence
(``uav_otfs_isac.qos.beta_mixture_cs``) and the 32-stream familywise
certification decisions (``anytime_qos_status``): interval semantics,
endpoint limits, tightening with information, Monte-Carlo coverage, and
the PASS/FAIL/UNCERTAIN decision rules pre-registered in advice/015
section 4 item 4.
"""

import numpy as np
from scipy.special import betaln

from uav_otfs_isac.qos import (
    anytime_qos_status,
    beta_mixture_cs,
    beta_mixture_log_evalue,
)

SPEC = 0.05
DELTA_S = SPEC / 32.0


def _log_evalue_increment(k, n, x, p, a=0.5, b=0.5):
    """Per-trial log-likelihood-ratio increment of the Beta-mixture
    e-process ``M_n(p)`` when trial ``n+1`` returns ``x`` (lets the test
    walk ``log M_n`` cheaply without per-step root finding): ``x=1`` adds
    ``ln B(k+1+a, n-k+b) - ln B(k+a, n-k+b) - ln p``; ``x=0`` adds
    ``ln B(k+a, n+1-k+b) - ln B(k+a, n-k+b) - ln(1-p)``."""
    if x == 1:
        return (betaln(k + 1 + a, n - k + b) - betaln(k + a, n - k + b)
                - np.log(p))
    return (betaln(k + a, n + 1 - k + b) - betaln(k + a, n - k + b)
            - np.log1p(-p))


def test_log_evalue_boundary_limits():
    # ``log M_n(p)`` tends to a FINITE limit as ``p -> 0+`` for ``k=0``
    # (and as ``p -> 1-`` for ``k=n``): the -k ln p (resp. (n-k) ln(1-p))
    # term vanishes and only the Beta-mixture predictive remains.
    n = 300
    a = b = 0.5
    lim = betaln(a, n + b) - betaln(a, b)
    assert np.isclose(beta_mixture_log_evalue(0, n, 0.0, a, b), lim)
    assert np.isclose(beta_mixture_log_evalue(0, n, 1e-15, a, b), lim,
                      atol=1e-6)
    assert np.isclose(beta_mixture_log_evalue(n, n, 1.0, a, b),
                      betaln(n + a, b) - betaln(a, b))


def test_cs_contains_empirical_rate():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 4000))
        k = int(rng.integers(0, n + 1))
        delta_s = float(rng.choice([0.05, 0.05 / 32]))
        lo, hi = beta_mixture_cs(k, n, delta_s)
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= k / n <= hi


def test_cs_shrinks_with_information():
    # same empirical rate, more trials -> tighter ANYTIME-valid bounds
    lo_small, hi_small = beta_mixture_cs(2, 100, 0.05 / 32)
    lo_big, hi_big = beta_mixture_cs(40, 2000, 0.05 / 32)
    assert hi_big < hi_small
    assert lo_big > lo_small


def test_cs_monotone_in_level():
    # smaller delta_s -> wider interval (more conservative)
    lo_loose, hi_loose = beta_mixture_cs(20, 500, 0.05)
    lo_tight, hi_tight = beta_mixture_cs(20, 500, 0.05 / 32)
    assert hi_tight > hi_loose
    assert lo_tight < lo_loose


def test_cs_coverage_simulation():
    # the ANYTIME guarantee implies a valid fixed-time interval too: at
    # the final look the true p must be covered with frequency ~ 1-delta_s
    true_p = 0.02
    n = 2000
    delta_s = 0.05
    paths = 300
    rng = np.random.default_rng(1)
    covered = 0
    for _ in range(paths):
        k = int(np.sum(rng.random(n) < true_p))
        lo, hi = beta_mixture_cs(k, n, delta_s)
        if lo <= true_p <= hi:
            covered += 1
    assert covered / paths >= 0.90   # true coverage 0.95; MC margin


def test_anytime_qos_status_pass():
    # certified PASS: small certified upper bounds on both error axes
    status = anytime_qos_status([700, 700], [800, 800], [2, 2], [3, 3],
                                SPEC, SPEC, delta_fam=0.05, n_streams=32)
    assert status == "PASS"


def test_anytime_qos_status_fail():
    # certified violation on the lower bound exceeds the spec
    status = anytime_qos_status([150, 150], [150, 150], [30, 30], [60, 60],
                                SPEC, SPEC, delta_fam=0.05, n_streams=32)
    assert status == "FAIL"


def test_anytime_qos_status_uncertain():
    # too few trials: neither certified pass nor certified violation
    status = anytime_qos_status([10, 10], [10, 10], [2, 2], [2, 2],
                                SPEC, SPEC, delta_fam=0.05, n_streams=32)
    assert status == "UNCERTAIN"


def test_anytime_qos_fail_not_masked_by_uncertain():
    # P3.5-A rule preserved: an earlier UNCERTAIN target must not mask a
    # later certified FAIL (target 1: 60 of 100 H1 missed -> certified
    # violation)
    status = anytime_qos_status([10, 100], [10, 100], [2, 60], [2, 60],
                                SPEC, SPEC, delta_fam=0.05, n_streams=32)
    assert status == "FAIL"


def test_anytime_qos_returns_bounds():
    status, bnd = anytime_qos_status([700, 700], [800, 800], [2, 2], [3, 3],
                                     SPEC, SPEC, delta_fam=0.05, n_streams=32,
                                     ret_bounds=True)
    assert status == "PASS"
    assert len(bnd["FA_lo"]) == 2 and len(bnd["FA_hi"]) == 2
    assert len(bnd["MD_lo"]) == 2 and len(bnd["MD_hi"]) == 2
    assert np.all(np.asarray(bnd["FA_hi"]) <= SPEC)\
        and np.all(np.asarray(bnd["MD_hi"]) <= SPEC)


def test_anytime_qos_familywise_level_per_stream():
    # n_streams=32 -> delta_s = 0.05/32 per stream; the PASS decision on
    # the two-error/two-target block consumes the SAME per-stream budget
    # as the 32-stream protocol (union bound)
    status = anytime_qos_status([700, 700], [800, 800], [2, 2], [3, 3],
                                0.05, 0.05, delta_fam=0.05, n_streams=32)
    assert status == "PASS"
    _, bnd = anytime_qos_status([700, 700], [800, 800], [2, 2], [3, 3],
                                0.05, 0.05, delta_fam=0.05, n_streams=32,
                                ret_bounds=True)
    # each per-stream bound uses exactly delta_s = 0.05/32 (the 32-stream
    # familywise budget), independently of how many targets are passed
    assert np.isclose(bnd["FA_hi"][0], beta_mixture_cs(2, 700, 0.05 / 32)[1])
    assert np.isclose(bnd["MD_hi"][1], beta_mixture_cs(3, 800, 0.05 / 32)[1])


def test_anytime_pathwise_never_escape():
    """advice/016 section 16: direct pathwise anytime test of the
    EVERYSTING property ``Pr(forall n<=N: p in C_n) >= 1 - delta``, i.e.
    ``Pr(exists n<=N: p outside C_n) <= delta`` (Ville bound at the
    e-process level), without needing per-step root finding.  For each
    ``p`` we walk ``log M_n(p)`` via the per-trial increments and flag a
    crossing exactly when ``log M_n(p) >= -ln delta`` -- this is ``p not
    in C_n`` by the definition ``C_n = {p : M_n(p) < 1/delta}``.  We also
    cross-check a few sample points against the actual ``beta_mixture_cs``
    interval inversion, i.e. that the root-finder implements exactly the
    same set.
    """
    delta = 0.05
    c_thr = -np.log(delta)
    N = 600
    paths = 500
    for p in (0.01, 0.05, 0.1, 0.5):
        rng = np.random.default_rng(11 + int(round(p * 100)))
        crossed = 0
        for _ in range(paths):
            logM = 0.0          # M_0(p) == 1
            k = 0
            ever = False
            for n in range(1, N + 1):
                x = int(rng.random() < p)
                logM += _log_evalue_increment(k, n - 1, x, p)
                k += x
                if logM >= c_thr:
                    ever = True
                    break       # once escaped, this path may stay escaped
            if ever:
                crossed += 1
        rate = crossed / paths
        # Ville: the escape probability is <= delta for EVERY p and n
        # (finite N can make it well BELOW delta, so only the upper bound
        # is a hard check); the lower sanity bound only ensures paths do
        # actually cross (the MC test is not vacuous).
        assert rate <= delta + 0.035        # hard anytime upper bound + MC margin
        assert rate >= 0.004                # not vacuous: crossings do occur
    # cross-check the interval root-finder against the direct e-value set
    # at a random sample of (k, n):  p inside [lo, hi]  <==>  log M_n(p) < c
    rng = np.random.default_rng(7)
    a = b = 0.5
    for _ in range(300):
        n = int(rng.integers(1, 2000))
        k = int(rng.integers(0, n + 1))
        p = float(rng.uniform(0.0, 1.0))
        lo, hi = beta_mixture_cs(k, n, delta)
        logM = (betaln(k + a, n - k + b) - betaln(a, b)
                - k * np.log(p) - (n - k) * np.log1p(-p)
                if 0.0 < p < 1.0 else 0.0)
        assert (logM < c_thr) == (lo < p < hi)