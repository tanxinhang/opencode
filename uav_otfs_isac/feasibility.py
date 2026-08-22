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

from collections import deque

import numpy as np
from scipy.optimize import minimize

from uav_otfs_isac.difficulty_decomposition import d_kl_binary
from uav_otfs_isac.frids import g_reliable


class _DinicMaxFlow:
    """Dinic's algorithm for s-t max flow (float capacities).  Used by the
    structure-exact bottleneck cut (advice/008 section 9): the min-cut
    certificate replaces the SLSQP-in-FW numeric oracle with an exact
    combinatorial graph algorithm on the *same* polyhedral certificate."""

    __slots__ = ("n", "graph", "level", "it")

    def __init__(self, n: int):
        self.n = n
        self.graph: list[list] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: float):
        self.graph[u].append([v, float(cap), len(self.graph[v])])
        self.graph[v].append([u, 0.0, len(self.graph[u]) - 1])

    def _bfs(self, s: int, t: int) -> bool:
        self.level = [-1] * self.n
        dq = deque([s])
        self.level[s] = 0
        while dq:
            u = dq.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 1e-12 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    dq.append(v)
        return self.level[t] >= 0

    def _dfs(self, u: int, t: int, f: float) -> float:
        if u == t:
            return f
        for i in range(self.it[u], len(self.graph[u])):
            v, cap, rev = self.graph[u][i]
            if cap > 1e-12 and self.level[v] == self.level[u] + 1:
                d = self._dfs(v, t, min(f, cap))
                if d > 1e-12:
                    self.graph[u][i][1] -= d
                    self.graph[v][rev][1] += d
                    return d
            self.it[u] = i + 1
        return 0.0

    def maxflow(self, s: int, t: int) -> float:
        flow = 0.0
        inf = 1e18
        while self._bfs(s, t):
            self.it = [0] * self.n
            while True:
                f = self._dfs(s, t, inf)
                if f <= 1e-12:
                    break
                flow += f
        return flow

    def reachable_from(self, s: int) -> list[bool]:
        """Nodes reachable from the source in the RESIDUAL graph after the
        max flow -- the maximum-weight-closure side of the min cut."""
        seen = [False] * self.n
        dq = deque([s])
        seen[s] = True
        while dq:
            u = dq.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 1e-12 and not seen[v]:
                    seen[v] = True
                    dq.append(v)
        return seen


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


def strongest_load_cut(scenario, owner_of, horizon, beta=0.05, alpha=0.05,
                       info_floor=None, lo=0.0, hi=8.0, bisect_iters=45,
                       smf_iters=400):
    """Strongest information-load cut ``rho*`` (the Bottleneck-Subset
    Feasibility Law) with the STRUCTURE-EXACT certificate (advice/008
    section 9): the ``min_S [lambda H F(S) - D(S)]`` feasibility oracle
    is one s-t min-cut on the AND-or-OR closure graph -- NOT the
    SLSQP-in-Fujishige-Wolfe numeric oracle (same polyhedral
    certificate, exact combinatorial algorithm instead of a stall
    cutoff).  Identity documented by ``rho_bruteforce``."""
    out = strongest_load_cut_exact(scenario, owner_of, horizon,
                                   beta=beta, alpha=alpha,
                                   info_floor=info_floor, lo=lo, hi=hi,
                                   bisect_iters=bisect_iters)
    return out


def mincut_closure_oracle(g: np.ndarray, D: np.ndarray, lam: float,
                          H: int) -> tuple[float, list]:
    """Structure-exact bottleneck-subset certificate for the *special*
    information cut (advice/008 section 9).

    Minimizes, over all target subsets ``S`` (empty included; empty gives
    0), the submodular function

        lam * H * F(S) - D(S),   F(S) = sum_i max_{q in S} g_iq,

    exactly by an s-t min-cut on a minimum-weight-closure graph.  The
    ``max_{q in S} g_iq`` terms are expanded over the sorted per-UAV
    distinct ``g`` levels,

        0 = gamma_{i0} < gamma_{i1} < ... < gamma_{iL_i},
        max_{q in S} g_iq = sum_l (gamma_{i,l} - gamma_{i,l-1})
                            * 1{ exists q in S: g_iq >= gamma_{i,l} },

    turning ``F(S)`` into a weighted OR-over-cover of auxiliaries.  The
    AND-or-OR graph is a minimum weight closure, solved to machine
    precision by one s-t max-flow (NO SLSQP, NO iteration/stall cutoff --
    this is the exact combinatorial certificate, not a numeric oracle).

    Returns ``(min_val, S*)`` with ``S*`` the minimizing subset (the
    targets on the source side of the min cut)."""
    k, q = g.shape
    # per-UAV sorted distinct positive g levels (the ``gamma_{i,l}``)
    levels = {
        i: sorted({float(g[i, qq]) for qq in range(q) if g[i, qq] > 0.0})
        for i in range(k)
    }
    # graph nodes: targets 0..q-1, one auxiliary node per (i, level) gap
    aux = []
    for i in sorted(levels):
        prev = 0.0
        for ga in levels[i]:
            aux.append((i, prev, ga))
            prev = ga
    n = q + len(aux)
    s, t = n, n + 1
    mf = _DinicMaxFlow(n + 2)
    inf = 1e15
    # target nodes carry the deficit bounty (min closure sees them as
    # negative weight); source->target capacity = D_q
    for qq in range(q):
        mf.add_edge(s, qq, float(D[qq]))
    # auxiliary (i, level) nodes carry the lambda*H*d-gap cost;
    # a target that reaches the level forces the aux on (OR closure edge)
    for idx, (i, prev, ga) in enumerate(aux):
        v = q + idx
        mf.add_edge(v, t, lam * H * (ga - prev))
        for qq in range(q):
            if g[i, qq] >= ga - 1e-12:
                mf.add_edge(qq, v, inf)
    cut = mf.maxflow(s, t)
    # min over S of lam H F(S) - D(S) = maxflow - sum(D):
    # the closure side collects sum(D) - (min over S) = -min(...),
    # so ``cut - C`` with the constant ``C = sum D`` telescopes; the
    # residual sink-side reachability gives the minimizing subset.
    side = mf.reachable_from(s)
    S_star = [qq for qq in range(q) if side[qq]]
    return float(cut - float(np.sum(D))), S_star


def strongest_load_cut_exact(scenario, owner_of, horizon, beta=0.05,
                             alpha=0.05, info_floor=None,
                             lo=0.0, hi=8.0, bisect_iters=45) -> dict:
    """Strongest information-load cut ``rho*`` by the structure-exact
    min-cut certificate (advice/008 section 9) + binary search on the
    dual ``lambda``:

        rho* > lambda   iff   min_{S} [ lambda H F(S) - D(S) ] < 0.

    Each oracle call is ONE s-t max-flow on the AND-or-OR closure graph
    (exact combinatorial certificate).  Returns the cut, the bottleneck
    subset and the feasibility flag -- the direct replacement for the
    SLSQP-inside-Fujishige-Wolfe numeric oracle (same polyhedral
    certificate, exact graph algorithm instead of a stall-cutoff)."""
    if info_floor is None:
        info_floor = float(d_kl_binary(1.0 - beta, alpha))
    k = scenario["k"]
    q = scenario["q"]
    g = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            g[i, qq] = g_reliable(scenario, i, qq, owner_of)
    D = info_floor * np.ones(q)

    def bounded(side: float) -> bool:
        # min_S [ lam H F(S) - D(S) ] >= 0  <=>  lam >= rho*
        val, _ = mincut_closure_oracle(g, D, side, horizon)
        return val >= -1e-9

    while not bounded(hi):
        hi *= 2.0
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if bounded(mid):
            hi = mid
        else:
            lo = mid
    rho = hi
    val, S = mincut_closure_oracle(g, D, rho - 1e-6, horizon)
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
