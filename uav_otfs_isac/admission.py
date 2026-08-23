"""Shared receiver-capacity admission primitive (advice/012 P4.1,
hotfix advice/013 P4.1a).

FRIDS-v2 and CA-FRIDS compare under the SAME physical receiver capacity
model:

    for every receiver ``j``:  sum_{i in A_j(t)} tau[i, j] / T_air <= 1,

with ``c_ij = tau[i, j] / T_air`` the airtime fraction a report from ``i``
reserves at receiver ``j`` (the owner-local diagonal ``tau[j, j] = 0`` is
free).  The physical order is always ``offer -> admission -> link
outcome``: an admitted transmission reserves (and is charged) the airtime
even when the later link Bernoulli fails.

Only ``who generates the offers`` and ``how the admission prioritises``
may differ between schedulers:

- ``policy="neutral"`` (FRIDS-v2, and the CA arm WITHOUT airtime price):
  no priority score -- the admitted set is decided solely by the airtime
  budget, with exchangeable tie keys.  This policy REQUIRES ``tie_keys``
  (it raises otherwise): a fallback to the source index would silently
  reintroduce fixed low-index bias, and a frozen per-episode key vector
  would reintroduce persistent source favoritism (advice/013 section 1).
  Keys may be a ``(k,)`` per-source vector or a ``(k, k)``
  ``[receiver, source]`` matrix (the per-cycle, per-receiver policy tape
  ``U_policy[r, t, j, i]`` suggested by advice/013 section 2); both give
  temporal + receiver + label exchangeability.
- ``policy="density"`` (CA-FRIDS): the density priority ``score / c``
  (the dual ``J^CA`` margin per airtime unit); ties use the same key
  semantics, or the plain deterministic source order when no keys are
  provided (legacy path, kept for reproducibility).

Offer records are ``(src_uav, priority, c_air, payload)`` grouped per
receiver; the return ``admitted`` lists ``(receiver, src_uav, c_air,
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
    """Admit offers under ``sum_i c_ij <= 1`` per receiver.

    ``offers_by_receiver[j]`` is a list of ``(src_uav, priority, c_air,
    payload)``.  ``policy="density"`` sorts by
    ``(priority/c desc, tie asc)``; ``policy="neutral"`` sorts by tie
    keys only (exchangeable) and REQUIRES ``tie_keys``.  A receiver keeps
    offers while the accumulated airtime stays within ``1``.  Returns
    ``(admitted, capacity_dropped)`` with ``admitted`` entries
    ``(receiver, src_uav, c_air, payload)``.
    """
    admitted: list[tuple] = []
    dropped = 0
    if policy == "neutral" and tie_keys is None:
        # advice/013 section 2: neutral admission without keys silently
        # falls back to the source index -- a deterministic low-index
        # bias.  Refuse it.
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
        for src, priority, c_air, payload in ordered:
            if used + float(c_air) <= 1.0 + 1e-12:
                admitted.append((receiver_j, src, float(c_air), payload))
                used += float(c_air)
            else:
                dropped += 1
    return admitted, dropped