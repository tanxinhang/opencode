"""Exogenous CRN tape shared by FRIDS-v2 and CA-FRIDS.

advice/010 (P0-2): a real paired comparison needs both schedulers to
consume the SAME exogenous realizations.  Legacy paths draw from one
sequential ``default_rng(seed)`` stream (v2) or per-run ``default_rng(
[seed, r])`` streams (CA), whose draw positions diverge as soon as the
two schedulers pick different actions.  When a :class:`ExogenousTape`
is supplied instead, both simulators read:

- ``U_H[r, q]``            -- target presence for run ``r``,
- ``U_obs[r, t, uav, q]``  -- observation uniform for ``(uav, q)`` at
  cycle ``t`` of run ``r`` (mapped through the chosen action's atom
  CDF by :func:`draw_atom`, so the SAME base uniform drives both
  schedulers even when they pick different kernels),
- ``U_link[r, t, src, dst]`` -- link Bernoulli ``src -> dst``,
- ``U_mfac[r, t, uav, q]`` -- frozen-mobility evidence walk,
- ``U_adm[r, t, uav]`` / ``U_adm_extra[r, t, uav]`` -- admission
  fractional gate and exchangeable tie-break uniforms.

The tape blocks are independent SeedSequence substreams of the same
seed, so they are reproducible AND parallel-safe.  ``exog=None`` keeps
every legacy path byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExogenousTape:
    seed: int
    n_runs: int
    q: int
    k: int
    max_steps: int
    U_H: NDArray[np.float64]           # (n_runs, q)
    U_obs: NDArray[np.float64]         # (n_runs, max_steps, k, q)
    U_link: NDArray[np.float64]        # (n_runs, max_steps, k, k)
    U_mfac: NDArray[np.float64]        # (n_runs, max_steps, k, q)
    U_adm: NDArray[np.float64]         # (n_runs, max_steps, k)
    U_adm_extra: NDArray[np.float64]   # (n_runs, max_steps, k)
    # P4.1a (advice/013 section 2): ALGORITHMIC policy randomness in its
    # OWN independent block (never mixed with the physical U_H/U_obs/
    # U_link blocks).  ``U_policy[r, t, j, i]`` = exchangeable admission
    # tie key per (episode, cycle, receiver, source), so NEUTRAL
    # admission has temporal + receiver + label exchangeability -- no
    # per-episode frozen keys, no persistent source favoritism.  All arms
    # (v2-neutral, CA-neutral, CA-density ties) read the SAME tape cell
    # for the same (r, t, receiver, src).
    U_policy: NDArray[np.float64]      # (n_runs, max_steps, k, k)


def build_exogenous_tape(
    seed: int,
    n_runs: int,
    q: int,
    k: int,
    max_steps: int,
) -> ExogenousTape:
    """Deterministically pre-register the seven independent uniform
    blocks: six PHYSICAL blocks (presence, observation, link, mobility,
    admission) plus one ALGORITHMIC policy block."""
    base = np.random.SeedSequence(int(seed))
    children = base.spawn(7)
    blocks = [np.random.default_rng(child) for child in children]
    return ExogenousTape(
        seed=int(seed),
        n_runs=int(n_runs),
        q=int(q),
        k=int(k),
        max_steps=int(max_steps),
        U_H=blocks[0].random((int(n_runs), int(q))),
        U_obs=blocks[1].random((int(n_runs), int(max_steps), int(k), int(q))),
        U_link=blocks[2].random((int(n_runs), int(max_steps), int(k), int(k))),
        U_mfac=blocks[3].random(
            (int(n_runs), int(max_steps), int(k), int(q))
        ),
        U_adm=blocks[4].random((int(n_runs), int(max_steps), int(k))),
        U_adm_extra=blocks[5].random((int(n_runs), int(max_steps), int(k))),
        U_policy=blocks[6].random(
            (int(n_runs), int(max_steps), int(k), int(k))
        ),
    )


def draw_atom(p: NDArray[np.float64], u: float) -> int:
    """Inverse-CDF index draw from atom probabilities ``p`` using one
    base uniform ``u`` (identical semantics to ``rng.choice(len(p),
    p=p)`` in the legacy path)."""
    p = np.asarray(p, dtype=float)
    cdf = np.cumsum(p)
    total = cdf[-1]
    cdf = cdf / float(max(total, 1e-300))
    return int(np.searchsorted(cdf, float(u), side="right"))