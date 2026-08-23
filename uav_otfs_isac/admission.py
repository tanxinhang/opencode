"""Shared receiver-capacity admission primitive (advice/012 P4.1).

FRIDS-v2 and CA-FRIDS must compare under the SAME physical receiver
capacity model.  The primitive is:

    for every receiver ``j``:  sum_{i in A_j(t)} tau[i, j] / T_air <= 1,

i.e. the admitted set ``A_j`` fits the receiver's airtime budget, where
``c_ij = tau[i, j] / T_air`` is the airtime fraction a report from ``i``
reserves at receiver ``j`` (the owner-local diagonal ``tau[j, j] = 0`` is
free).  The physical order is always ``offer -> admission -> link
outcome``: a transmission that is admitted reserves (and is charged) the
airtime even when the later link Bernoulli fails.

Only ``who generates the offers`` and ``how the admission prioritises``
may differ between schedulers:

- ``policy="neutral"`` (FRIDS-v2): no priority score -- the admitted set
  is decided solely by the airtime budget, with exchangeable tie keys on
  the UAV indices (label-equivariant, no low-index bias).
- ``policy="density"`` (CA-FRIDS): the current density priority
  ``score / c`` (the dual ``J^CA`` margin per airtime unit).

Offer records are ``(src_uav, priority, c_air, payload)`` grouped per
receiver; the return ``admitted`` lists ``(receiver, src_uav, c_air,
payload)`` and ``n_capacity_dropped`` counts the offers the budget
rejected.  ``tie_keys`` is a per-UAV uniform vector (length ``num_uavs``)
drawn by a DEDICATED policy RNG -- never part of the exogenous physics
tape.
"""

from __future__ import annotations

import numpy as np


def airtime_admit(
    offers_by_receiver: list[list[tuple]],
    t_air: float,
    *,
    policy: str = "density",
    tie_keys: np.ndarray | None = None,
) -> tuple[list[tuple], int]:
    """Admit offers under ``sum_i c_ij <= 1`` per receiver.

    ``offers_by_receiver[j]`` is a list of ``(src_uav, priority, c_air,
    payload)``.  Sorted by ``(priority desc, tie_keys[src] asc)`` when
    ``policy="density"``, or by ``tie_keys[src]`` alone (exchangeable
    neutral) when ``policy="neutral"``; a receiver keeps offers while the
    accumulated airtime stays within ``1``.  Returns ``(admitted,
    capacity_dropped)`` with ``admitted`` entries
    ``(receiver, src_uav, c_air, payload)``.
    """
    admit_capacity = float(t_air) / max(float(t_air), 1e-30)  # = 1 budget
    admitted: list[tuple] = []
    dropped = 0
    for receiver_j, off in enumerate(offers_by_receiver):
        if not off:
            continue
        if policy == "neutral":
            def _key(o):
                return (float(tie_keys[o[0]]) if tie_keys is not None else 0.0,
                        int(o[0]))
            ordered = sorted(off, key=_key)
        elif policy == "density":
            def _key(o):
                score, c_air = o[1], o[2]
                return (-(float(score) / max(float(c_air), 1e-15)),
                        float(tie_keys[o[0]]) if tie_keys is not None else 0.0,
                        int(o[0]))
            ordered = sorted(off, key=_key)
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