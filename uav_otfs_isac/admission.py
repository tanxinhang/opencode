"""Shared receiver-capacity admission primitive (advice/012 P4.1, P4.1a
fairness hotfix, P4.1b single-source-credibility refactor).

FRIDS-v2 and CA-FRIDS compare under the SAME physical receiver capacity
model.  This primitive IS the single source of that model:

    for every receiver ``j``:  sum_{i in A_j(t)} tau[i, j] <= T_air,

i.e. the admitted set ``A_j`` fits the receiver's airtime budget ``T_air``
(raw airtime units; the owner-local diagonal ``tau[j, j] = 0`` is free).
The physical order is always ``offer -> admission -> link outcome``: an
admitted transmission reserves (and is charged) the airtime even when the
later link Bernoulli fails.

`T_air` is MEANINGFUL here -- the primitive accumulates raw ``tau`` and
compares against ``T_air`` (advice/014 section 5).  Callers must NOT
pre-divide ``c = tau/T_air`` and compare against 1: that would split the
capacity model across call sites and silently accept a mismatched ``T_air``
if a caller ever passes a stale one.

Only ``who generates the offers`` and ``how the admission prioritises``
may differ between schedulers:

- ``policy="neutral"`` (FRIDS-v2, and CA arms without the airtime price):
  no priority score -- the admitted set is decided solely by the airtime
  budget, with exchangeable tie keys.  REQUIRES ``tie_keys`` (raises
  otherwise: the source-index fallback would recreate fixed low-index
  bias; a per-episode frozen key vector would recreate persistent source
  favoritism -- advice/013 section 1).  Keys may be a ``(k,)`` per-source
  vector or a ``(k, k)`` ``[receiver, source]`` policy tape
  ``U_policy[r, t, j, i]`` (temporal + receiver + label exchangeability).
- ``policy="density"`` (CA-FRIDS): the density priority ``score / tau``
  (the dual ``J^CA`` margin per airtime unit; ties use the same key
  semantics, or the plain deterministic source order when no keys are
  provided -- legacy path, kept for reproducibility).

Offer records are ``(src_uav, priority, tau_ij, payload)`` grouped per
receiver; the return ``admitted`` lists ``(receiver, src_uav, tau_ij,
payload)`` and ``n_capacity_dropped`` counts the offers the budget
rejected.
"""

from __future__ import annotations

import numpy as np


def _tie_value(tie_keys, receiver: int, src: int) -> float:
    """Tie key for ``(receiver, src)`` with either per-source ``(k,)`` or
    per-receiver-per-source ``(k, k)`` keys."""
    arr = np.asarray(tie_keys, dtype=float)
    if arr.ndim == 2:
        return float(arr[receiver, src])
    return float(arr[src])


def airtime_admit(
    offers_by_receiver: list[list[tuple]],
    t_air: float,
    *,
    policy: str = "density",
    tie_keys: np.ndarray | None = None,
) -> tuple[list[tuple], int]:
    """Admit offers under ``sum_i tau_ij <= T_air`` per receiver.

    ``offers_by_receiver[j]`` is a list of ``(src_uav, priority, tau_ij,
    payload)`` with ``tau_ij`` in RAW airtime units.  ``T_air`` is the
    physical per-receiver budget.  ``policy="density"`` sorts by
    ``(priority / tau desc, tie asc)``; ``policy="neutral"`` sorts by tie
    keys only (exchangeable) and REQUIRES ``tie_keys``.  A receiver keeps
    offers while the accumulated airtime stays within ``T_air``.  Returns
    ``(admitted, capacity_dropped)`` with ``admitted`` entries
    ``(receiver, src_uav, tau_ij, payload)``.
    """
    admitted: list[tuple] = []
    dropped = 0
    budget = float(t_air)
    if policy == "neutral" and tie_keys is None:
        raise ValueError(
            "airtime_admit(policy='neutral') requires tie_keys; the "
            "source-index fallback would recreate fixed low-index bias"
        )
    for receiver_j, off in enumerate(offers_by_receiver):
        if not off:
            continue
        if policy == "neutral":
            ordered = sorted(
                off,
                key=lambda o: (float(_tie_value(tie_keys, receiver_j, o[0])),
                               int(o[0])),
            )
        elif policy == "density":
            ordered = sorted(
                off,
                key=lambda o: (
                    -(float(o[1]) / max(float(o[2]), 1e-15)),
                    (float(_tie_value(tie_keys, receiver_j, o[0]))
                     if tie_keys is not None else 0.0),
                    int(o[0]),
                ),
            )
        else:
            raise ValueError(f"unknown admission policy {policy!r}")
        used = 0.0
        for src, priority, tau, payload in ordered:
            if used + float(tau) <= budget + 1e-12:
                admitted.append((receiver_j, src, float(tau), payload))
                used += float(tau)
            else:
                dropped += 1
    return admitted, dropped