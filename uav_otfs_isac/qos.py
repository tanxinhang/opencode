"""P2.1a QoS certificate (advice/008 section 13): simultaneous-confidence
Dual-QoS status on the RAW conditional counts.

The frozen runner previously turned the already-computed conditional
error probabilities into ``p * n_runs`` successes -- the inferred count
has the WRONG denominator (per-target ``N_H0``/``N_H1`` are random, not
``n_runs``), which the advice/008 section 13 audit listed as a P0 flaw.
This module certifies the Dual QoS

    P_FA,q <= alpha   AND   P_MD,q <= beta       (every target)

from the raw counts ``(N_H0,q, N_H1,q, N_FA,q, N_MD,q)`` with the
simultaneous confidence ``delta_q = delta_cell / (2Q)`` (Bonferroni over
the two error axes of every target, Clopper-Pearson exact two-sided):
PASS iff the certified upper bounds meet both specs on every target;
FAIL iff the certified LOWER bound of some error probability exceeds
its spec (a certified violation -- only then is the cell qualified);
UNCERTAIN otherwise, and UNCERTAIN is UNRESOLVED by construction (it is
never re-labelled by necessary-condition reasoning).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln
from scipy.stats import beta


def clopper_pearson(k: int, n: int, a2: float) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval of a binomial
    proportion at confidence ``1 - a2`` (probability ``k`` of ``n``)."""
    if n <= 0:
        return 0.0, 1.0
    k = int(np.clip(k, 0, n))
    lo = 0.0 if k == 0 else float(beta.ppf(a2 / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a2 / 2.0, k + 1, n - k))
    return float(lo), float(hi)


def raw_qos_status(n_H0, n_H1, n_FA, n_MD, alpha: float, beta: float,
                   delta_cell: float = 0.05) -> str:
    """Simultaneous Dual-QoS status from the RAW per-target conditional
    counts (advice/008 section 13).  Returns PASS / FAIL / UNCERTAIN.

    ``delta_q = delta_cell/(2Q)``; a two-sided Clopper-Pearson interval
    of confidence ``1 - delta_q`` is drawn around every P_FA,q and
    P_MD,q.  The cell is PASS if every certified UPPER bound clears its
    spec, FAIL if some certified LOWER bound exceeds its spec (a
    certified QoS violation), UNCERTAIN otherwise -- and UNCERTAIN is
    never relabelled (the protocol keeps it unresolved)."""
    q = len(n_H0)
    delta_q = delta_cell / (2.0 * max(q, 1))
    # P3.5-A (advice/009 P0-1): scan ALL targets BEFORE deciding.  A
    # per-target early return lets an earlier UNCERTAIN target mask a
    # later certified FAIL, contradicting the definition
    # ``exists q: LCB(P_err,q) > p_max  => FAIL``.  FAIL is the strongest
    # decision the data can make, so it is checked across every target
    # first; UNCERTAIN is returned only if some target is unresolved and
    # none is a certified violation.
    has_uncertain = False
    for qq in range(q):
        fa_lo, fa_hi = clopper_pearson(int(n_FA[qq]), int(n_H0[qq]),
                                       delta_q)
        md_lo, md_hi = clopper_pearson(int(n_MD[qq]), int(n_H1[qq]),
                                       delta_q)
        if fa_lo > alpha or md_lo > beta:
            return "FAIL"
        if fa_hi > alpha or md_hi > beta:
            has_uncertain = True
    return "UNCERTAIN" if has_uncertain else "PASS"


def pool_raw_counts(rows: list[dict]) -> dict:
    """Aggregate the per-run ``raw_counts`` blocks over geoms and MC
    seeds into the pooled per-target counts the cell certificate uses."""
    q = len(rows[0]["raw_counts"]["n_H0"])
    out = {key: [0] * q for key in ("n_H0", "n_H1", "n_FA", "n_MD")}
    for row in rows:
        rc = row["raw_counts"]
        for key in out:
            out[key] = [a + b for a, b in zip(out[key], rc[key])]
    return out


# ---------------------------------------------------------------------------
# P4.2 anytime-valid QoS certification (advice/015 section 2)
# ---------------------------------------------------------------------------
# For every target/error Bernoulli stream ``X_t in {0,1}`` with
# ``S_n = sum_{t<=n} X_t``, the Beta mixture ``a = b = 1/2`` defines the
# e-value / non-negative martingale
#
#     M_n(p) = B(S_n+a, n-S_n+b) / (B(a,b) p^{S_n} (1-p)^{n-S_n})
#
# under any fixed true ``p``, so by Ville's inequality the confidence set
#
#     C_n = { p : M_n(p) < 1/delta_s }
#
# satisfies ``Pr(forall n, p in C_n) >= 1 - delta_s`` -- the ANYTIME-VALID
# guarantee that makes the four pre-registered looks
# (0:7.5k -> 0:15k -> 0:30k -> 0:60k) and even data-driven stopping legal
# without alpha-spending.  The stopping time may be decided by the data.
# ``log M_n(p)`` is strictly convex in ``p`` with its minimum at the
# empirical rate ``k/n``, so ``C_n`` is an interval ``[lo, hi]`` whose
# endpoints are the two roots of ``log M_n(p) = - log delta_s``.

_LOG_EPS = 1e-9


def beta_mixture_log_evalue(k: int, n: int, p: float,
                            a: float = 0.5, b: float = 0.5) -> float:
    """Log of the Beta-mixture e-value ``M_n(p)`` at ``(k successes of
    n trials)``.  Under a fixed true ``p``, ``M_n(p)`` is a non-negative
    martingale starting at 1 (the ratio of the Beta(a,b)-mixture
    predictive marginal to the point-null binomial likelihood), so Ville
    gives the time-uniform bound.  The boundary limits for ``k=0`` /
    ``k=n`` return the finite endpoint value (``log M_n`` tends to it as
    ``p -> 0+`` / ``p -> 1-``)."""
    n = int(n)
    k = int(np.clip(int(k), 0, n))
    log_pred = float(betaln(k + a, n - k + b) - betaln(a, b))
    log_den = 0.0
    if k > 0:
        if p <= 0.0:
            return float("inf")
        log_den -= k * np.log(p)
    if n - k > 0:
        if p >= 1.0:
            return float("inf")
        log_den -= (n - k) * np.log1p(-p)
    return log_pred + log_den


def beta_mixture_cs(k: int, n: int, delta_s: float,
                    a: float = 0.5, b: float = 0.5) -> tuple[float, float]:
    """ANYTIME-valid confidence interval ``[lo, hi]`` of a Bernoulli
    proportion from ``(k successes, n trials)`` at per-stream level
    ``delta_s``: the Beta-mixture e-process set ``{p : M_n(p) <
    1/delta_s}`` (advice/015 section 2).  Coverage ``1 - delta_s`` holds
    SIMULTANEOUSLY at every stopping time, so the four pre-registered
    prefix looks and data-driven early stopping need no alpha-spending.
    Returns the vacuous ``(0, 1)`` when the confidence set is empty at
    this level (the stream is > ``1/delta_s`` more likely under the
    mixture than under EVERY Bernoulli ``p`` -- the Bernoulli model is
    rejected for every ``p``)."""
    if n <= 0:
        return (0.0, 1.0)
    k = int(np.clip(int(k), 0, n))
    c = float(-np.log(float(delta_s)))
    p_hat = k / n
    g_mle = beta_mixture_log_evalue(k, n, p_hat, a, b) - c
    if g_mle > 0.0:
        return (0.0, 1.0)
    lo = 0.0
    if k > 0:
        if g_mle == 0.0:
            lo = p_hat
        else:
            lo = float(brentq(
                lambda p: beta_mixture_log_evalue(k, n, p, a, b) - c,
                _LOG_EPS, p_hat))
    hi = 1.0
    if n - k > 0:
        if g_mle == 0.0:
            hi = p_hat
        else:
            hi = float(brentq(
                lambda p: beta_mixture_log_evalue(k, n, p, a, b) - c,
                max(p_hat, _LOG_EPS), 1.0 - _LOG_EPS))
    return (lo, hi)


def anytime_stream_bounds(n_err, n_tri, delta_s: float,
                          a: float = 0.5, b: float = 0.5) \
                          -> tuple[list, list]:
    """Per-target anytime-valid lower/upper bounds of ``N_err`` errors
    over ``N_tri`` Bernoulli trials, each target at level ``delta_s``.
    Returns ``(los, his)``."""
    los, his = [], []
    for qq in range(len(n_tri)):
        lo, hi = beta_mixture_cs(int(n_err[qq]), int(n_tri[qq]), delta_s,
                                 a, b)
        los.append(float(lo))
        his.append(float(hi))
    return los, his


def anytime_qos_status(n_H0, n_H1, n_FA, n_MD, alpha: float, beta: float,
                       delta_fam: float = 0.05, n_streams: int = 32,
                       a: float = 0.5, b: float = 0.5,
                       ret_bounds: bool = False):
    """P4.2 ANYTIME-valid Dual-QoS status (advice/015 section 4): the
    simultaneous familywise certificate over the ``2 algorithms x 8
    targets x 2 error axes (FA/MD) = 32`` Bernoulli streams with
    ``delta_s = delta_fam / n_streams`` per stream (Bonferroni / union
    bound -- no independence needed, so CRN stays valid).  Every stream
    uses the Beta-mixture time-uniform e-process ``C_n``, so the set of
    (error probability, target) is certified SIMULTANEOUSLY across the
    pre-registered prefix looks:

        PASS(A)  <=>  max_q U^A_FA,q <= alpha  and  max_q U^A_MD,q <= beta
        FAIL(A)  <=>  exists q:  L^A_FA,q > alpha  or  L^A_MD,q > beta

    FAIL is checked across every target FIRST (an earlier UNCERTAIN
    target must not mask a later certified FAIL -- the P3.5-A rule);
    UNCERTAIN is returned only if some target is unresolved and none is
    a certified violation.
    """
    q = len(n_H0)
    delta_s = delta_fam / max(n_streams, 1)
    fa_lo, fa_hi = anytime_stream_bounds(n_FA, n_H0, delta_s, a, b)
    md_lo, md_hi = anytime_stream_bounds(n_MD, n_H1, delta_s, a, b)
    has_uncertain = False
    for qq in range(q):
        if fa_lo[qq] > alpha or md_lo[qq] > beta:
            if ret_bounds:
                return "FAIL", {"FA_lo": fa_lo, "FA_hi": fa_hi,
                                "MD_lo": md_lo, "MD_hi": md_hi}
            return "FAIL"
        if fa_hi[qq] > alpha or md_hi[qq] > beta:
            has_uncertain = True
    status = "UNCERTAIN" if has_uncertain else "PASS"
    if ret_bounds:
        return status, {"FA_lo": fa_lo, "FA_hi": fa_hi,
                        "MD_lo": md_lo, "MD_hi": md_hi}
    return status