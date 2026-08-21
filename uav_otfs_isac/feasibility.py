"""Feasibility theory for FRIDS (advice/012, Gate F0-G6).

The strongest information-load cut

    rho* = max_{empty != S subset Q}  rho(S),
    rho(S) = sum_{q in S} D_q^info / (H * sum_i max_{q in S} g_iq),

is the bottleneck-subset feasibility law: ``rho* > 1`` means SOME target
subset is information-theoretically infeasible within the horizon, no
matter the scheduler.  ``F(S) = sum_i max_{q in S} g_iq`` is monotone
submodular (each ``f_i(S) = max_{q in S} g_iq`` has diminishing
returns), ``D(S) = sum D_q^info`` is modular, so
``lambda H F(S) - D(S)`` is submodular and ``rho* > lambda`` is decided
by SUBMODULAR MINIMIZATION (polynomial), not by 2^Q enumeration: binary
search on lambda with a submodular-minimization oracle (Fujishige-Wolfe
minimum-norm-point on the base polytope, with the greedy linear oracle).

The communication load ``rho_C = max_i b_tok (K-1) / Bbar_i^rx`` is
computed separately (communication-budget limited) from the detection
information load ``rho_I*`` (sensing/reliable-information limited); the
two infeasibilities must not be mixed.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from uav_otfs_isac.difficulty_decomposition import d_kl_binary
from uav_otfs_isac.frids import g_reliable


def f_submodular_oracle(scenario, owner_of, lam: float, horizon: int,
                        info_floor: float):
    """Returns ``f(S) = lam * H * F(S) - D(S)`` (submodular) as a
    callable over frozensets."""
    k = scenario["k"]
    q = scenario["q"]
    # precompute per-UAV per-target g and per-target info deficits
    g = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            g[i, qq] = g_reliable(scenario, i, qq, owner_of)
    D = info_floor * np.ones(q)

    def f(S: frozenset) -> float:
        if not S:
            return 0.0
        idx = np.array(sorted(S))
        f_val = float(np.sum(np.max(g[:, idx], axis=1)))
        return lam * horizon * f_val - float(np.sum(D[idx]))

    return f


def _greedy_vertex(f, n: int, w: np.ndarray) -> np.ndarray:
    """Greedy (Edmonds) linear oracle over the base polytope B(f):
    maximize w.x subject to x(S) <= f(S), x(V) = f(V)."""
    order = np.argsort(-w)
    x = np.zeros(n)
    S = set()
    prev = 0.0
    for kk in order:
        S.add(int(kk))
        val = f(frozenset(S))
        x[kk] = val - prev
        prev = val
    return x


def _least_norm_weights(vertices, max_iter: int = 200):
    """min ||sum lam_v v||^2 over the probability simplex (convex QP)."""
    V = np.array(vertices)
    n = len(vertices)

    def obj(lam):
        y = lam @ V
        return float(y @ y)

    lam0 = np.full(n, 1.0 / n)
    res = minimize(obj, lam0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints={"type": "eq",
                                "fun": lambda l: np.sum(l) - 1.0},
                   options={"maxiter": max_iter, "ftol": 1e-14})
    lam = res.x if res.success else lam0
    lam = np.clip(lam, 0.0, 1.0)
    lam = lam / max(np.sum(lam), 1e-12)
    return lam


def fw_minimum_norm_point(f, n: int, iters: int = 800,
                          tol: float = 1e-10, seed: int = 0) -> np.ndarray:
    """Fujishige-Wolfe minimum-norm point of the base polytope B(f) of a
    submodular ``f``; the minimizer of ``f`` is ``{i : x_i < 0}``.

    Iteration: at the least-norm point y of conv(V), prune V to the
    support of the convex combination (Caratheodory), run the greedy
    linear oracle in direction ``-y``; if it returns y, stop, otherwise
    add the new vertex.  Each addition strictly decreases the norm of
    the least-norm point, so the loop terminates.
    """
    rng = np.random.default_rng(seed)
    v0 = _greedy_vertex(f, n, rng.normal(size=n))
    V = [np.array(v0)]
    last_norm = np.inf
    stall = 0
    for _ in range(iters):
        lam = _least_norm_weights(V)
        y = sum(l * v for l, v in zip(lam, V))
        norm = float(np.dot(y, y))
        if norm >= last_norm - 1e-12:
            stall += 1
            if stall >= 5:
                return np.asarray(y)
        else:
            stall = 0
            last_norm = norm
        support = [i for i, l in enumerate(lam) if l > 1e-10]
        V = [V[i] for i in support]
        v_new = _greedy_vertex(f, n, -y)
        if np.linalg.norm(v_new - y) < tol:
            return np.asarray(y)
        V.append(np.array(v_new))
    return np.asarray(y)


def submodular_minimize(f, n: int, iters: int = 400) -> tuple:
    """Minimize a submodular ``f`` (f(empty)=0) via the FW
    minimum-norm-point; returns (S*, value)."""
    x = fw_minimum_norm_point(f, n, iters=iters)
    S = frozenset(int(i) for i in range(n) if x[i] < 0.0)
    return S, float(f(S))


def strongest_load_cut(scenario, owner_of, horizon: int,
                       beta: float = 0.05, alpha: float = 0.05,
                       info_floor: float | None = None,
                       lo: float = 0.0, hi: float = 8.0,
                       bisect_iters: int = 45,
                       smf_iters: int = 400) -> dict:
    """The strongest information-load cut ``rho*`` by submodular
    minimization + binary search: ``rho* > lambda`` iff
    ``min_S [lambda H F(S) - D(S)] < 0``.  Returns the cut value, the
    bottleneck subset (at the largest infeasible lambda), and the
    feasibility flag."""
    if info_floor is None:
        info_floor = float(d_kl_binary(1.0 - beta, alpha))
    q = scenario["q"]

    def feasible(lam: float) -> bool:
        f = f_submodular_oracle(scenario, owner_of, lam, horizon,
                                info_floor)
        _, val = submodular_minimize(f, q, iters=smf_iters)
        return val >= -1e-9

    # binary search for the SMALLEST lambda with min_S f >= 0
    # (feasible(lam) is True iff lam >= rho*; the cut is the crossing)
    lo, hi = 0.0, hi
    while not feasible(hi):
        hi *= 2.0
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    rho = hi
    # bottleneck subset: the minimizer on the infeasible side (argmax
    # of rho(S))
    f = f_submodular_oracle(scenario, owner_of, rho - 1e-6,
                            horizon, info_floor)
    S, _ = submodular_minimize(f, q, iters=smf_iters)
    bottleneck = sorted(S)
    return {
        "rho_star": float(rho),
        "bottleneck_subset": bottleneck,
        "feasible_info": bool(rho <= 1.0),
    }


def rho_bruteforce(scenario, owner_of, horizon, beta=0.05, alpha=0.05,
                   info_floor=None) -> float:
    """Exhaustive rho* over all subsets (for verification at small Q)."""
    if info_floor is None:
        info_floor = float(d_kl_binary(1.0 - beta, alpha))
    q = scenario["q"]
    k = scenario["k"]
    g = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            g[i, qq] = g_reliable(scenario, i, qq, owner_of)
    best = 0.0
    for mask in range(1, 1 << q):
        idx = [qq for qq in range(q) if mask & (1 << qq)]
        d = info_floor * len(idx)
        f = float(np.sum(np.max(g[:, idx], axis=1)))
        best = max(best, d / (horizon * max(f, 1e-12)))
    return best


def communication_load(k: int, b_tok: float, rx_budget: float) -> float:
    """rho_C = max_i b_tok (K-1) / Bbar_i^rx (full-mesh receive load vs
    the per-UAV receive/decode budget)."""
    return float(b_tok * (k - 1) / max(rx_budget, 1e-12))
