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
    for qq in range(q):
        fa_lo, fa_hi = clopper_pearson(int(n_FA[qq]), int(n_H0[qq]),
                                       delta_q)
        md_lo, md_hi = clopper_pearson(int(n_MD[qq]), int(n_H1[qq]),
                                       delta_q)
        # certified violation check FIRST (FAIL is the strongest decision
        # the data can make at the simultaneous level)
        if fa_lo > alpha or md_lo > beta:
            return "FAIL"
        # certified PASS only when every UCB clears the spec
        if fa_hi > alpha or md_hi > beta:
            return "UNCERTAIN"
    return "PASS"


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