"""Belief-state (posterior) controlled active detection.

Role (advice/005 section 10): ``deployment`` -- the **local action-value
layer** of the distributed scheduler.  The per-target delay values, the
numerically calibrated two-threshold stopping rule and the dual G-value /
Whittle / rollout indices built here are reused by ``distributed_audit``
as the per-UAV index ``J_{i,q,a}`` (advice/005 section 8); the Bellman /
joint-oracle functions remain as the offline audit reference (Gates
D1-D2).  Do not grow new global joint Bellmans here (advice/005 section
10).

advice/002.md restructures the three-layer static pipeline (time
prediction -> static allocation -> per-link quantizer design) into a single
optimal stopping + experiment design problem whose state is the posterior
log-odds

    L_{t+1} = L_t + log( p1^a(Y_t) / p0^a(Y_t) ),

exact for every post-communication kernel (quantization + BSC + detectable
erasure, ``detection_information.post_communication_likelihoods``).

Contents
--------
- ``action_kernels``: per-action post-communication ``(p0_y, p1_y)`` with
  the exact LLR atoms, ``I+`` and Chernoff values.
- ``blackwell_dominates``: LP feasibility test of ``p_h^b = p_h^a K`` for a
  stochastic garbling ``K`` (Blackwell order); if also ``c(a) <= c(b)``,
  ``b`` can never be Bellman-optimal (the controller can simulate ``b``
  through ``K`` after playing ``a``).
- ``exact_alpha_vectors``: finite-horizon POMDP value function by the
  piecewise-linear-concave (alpha-vector) recursion -- exact, no belief
  grid (exponential in the horizon, used for the oracle and small settings).
- ``grid_bellman_value``: dense-grid value iteration on the log-odds axis
  with exact kernel expectations (scalable variant, verified against the
  alpha-vector recursion on small instances).
- ``budget_bellman_value`` / ``budget_bellman_policy`` / ``rollout_budget``:
  the explicit-budget form ``V_t(pi, B)`` of the finite-horizon Bellman
  (advice/003 Gate D1): the state is (log-odds, remaining observation
  budget), observations drain ``c(a)`` from the budget, and an empty
  continuation set forces a terminal decision.  With a budget large enough
  to pay any policy the values coincide with ``grid_bellman_value``.
- ``residual_adaptive_policy`` / ``rollout_mismatch``: the mathematical
  Reflexion of advice/003 section 4.  The controller monitors the realized
  Bellman residual ``r_t = c(a_t) + V(l_{t+1}) - V(l_t)`` of the played
  observations (zero-mean under the correct model); when the EMA ``|r_t|``
  exceeds a threshold the model is judged unreliable and the controller
  switches from the nominal value policy to a robust one-step lookahead
  and, if the mismatch persists, to epistemic exploration (maximum
  information per cost), with hysteresis back to the nominal mode.
  ``rollout_mismatch`` evaluates policies whose observation kernels differ
  from the kernels used by the model (the environment draws from ``true``
  kernels while the belief update uses the model LLRs).
- policies: ``bellman_policy``, ``tau_pred_policy`` (myopic ``I+/c`` with
  Wald boundaries), ``chernoff_policy``, ``dpd_policy`` (one-step Bayesian
  lookahead), ``static_policy``.
- ``rollout``: Monte-Carlo evaluation (expected delay, P_FA, P_MD, expected
  cost, realized accumulated information).
- ``information_lower_bounds``: the sequential-testing information
  inequalities ``E_1[sum I+_t] >= d(1-beta || alpha)`` (advice eq. 5),
  ``E_0[sum I-_t] >= d(1-alpha || beta)`` (eq. 6) and the budget forms via
  ``rho^+ = max_a I_a^+/c(a)`` (eqs. 7-8), verified on rolled-out
  detectors.
- ``dual_decomposed_value``: per-target Bellman with fairness prices
  ``nu_q`` and resource price ``lambda`` (Lagrangian decomposition, advice
  eq. 3) plus the per-cycle knapsack scheduler over ``G = nu_q DV - lambda c``.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from .detection_information import (
    post_communication_likelihoods,
)
from .detection_quantization import quantizer_edges


# ---------------------------------------------------------------------------
# Kernels and Blackwell order
# ---------------------------------------------------------------------------

def action_kernels(
    mu0: float,
    var0: float,
    mu1: float,
    var1: float,
    bits: int,
    flip_probability: float,
    success_probability: float,
    span_std: float = 4.0,
) -> dict:
    """Post-communication kernel of one action (report/quantizer choice).

    Returns ``p0_y``/``p1_y`` over the quantized levels plus the erasure
    atom, the exact LLR atoms with their H1/H0 masses, ``i_plus``, and the
    Chernoff information.
    """
    edges, values = quantizer_edges(
        mu0, var0, mu1, var1, bits, span_std,
    )
    info = post_communication_likelihoods(
        mu0, var0, mu1, var1, edges, values,
        bits, flip_probability, success_probability,
    )
    p1 = np.asarray(info["p1_y"], dtype=float)
    p0 = np.asarray(info["p0_y"], dtype=float)
    keep = (p1 > 0.0) | (p0 > 0.0)
    p1 = p1[keep]
    p0 = p0[keep]
    llr = np.log(p1 / p0)
    if not np.all(np.isfinite(llr)):
        raise ValueError("kernel has infinite LLR atoms (zero-mass level)")
    return {
        "p0": p0,
        "p1": p1,
        "llr": llr,
        "i_plus": float(info["kl_plus"]),
        "i_minus": float(info["kl_minus"]),
        "chernoff": float(info["chernoff"]),
    }


def blackwell_dominates(
    kernel_a: dict,
    kernel_b: dict,
    tol: float = 1e-9,
) -> bool:
    """True if ``a`` dominates ``b`` in the Blackwell order.

    Feasibility of a stochastic matrix ``K`` with ``p_h^b = p_h^a K`` for
    ``h = 0, 1`` (``b`` is a garbling of ``a``), solved as a linear
    program.  If additionally ``c(a) <= c(b)``, action ``b`` is never
    Bellman-optimal: after playing ``a`` the controller can simulate
    ``b``'s observation through ``K`` and continue with any strategy that
    ``b`` would have used.
    """
    p0a = np.asarray(kernel_a["p0"], dtype=float)
    p1a = np.asarray(kernel_a["p1"], dtype=float)
    p0b = np.asarray(kernel_b["p0"], dtype=float)
    p1b = np.asarray(kernel_b["p1"], dtype=float)
    na, nb = len(p0a), len(p0b)
    n_var = na * nb
    # variables K[j, k] flattened row-major; constraints:
    #   row sums == 1 (na rows);  p_h^b == p_h^a @ K (2 * nb equations)
    a_eq = np.zeros((na + 2 * nb, n_var))
    b_eq = np.zeros(na + 2 * nb)
    for j in range(na):
        a_eq[j, j * nb:(j + 1) * nb] = 1.0
        b_eq[j] = 1.0
    row = na
    for k in range(nb):
        for j in range(na):
            a_eq[row, j * nb + k] = p0a[j]
        b_eq[row] = p0b[k]
        row += 1
    for k in range(nb):
        for j in range(na):
            a_eq[row, j * nb + k] = p1a[j]
        b_eq[row] = p1b[k]
        row += 1
    result = linprog(
        np.zeros(n_var),
        A_eq=a_eq, b_eq=b_eq,
        bounds=(0.0, None), method="highs",
    )
    if not result.success:
        return False
    return np.allclose(a_eq @ result.x, b_eq, atol=tol)


# ---------------------------------------------------------------------------
# Exact finite-horizon value function (alpha vectors)
# ---------------------------------------------------------------------------

def _prune_alpha_vectors(vectors: list[tuple]) -> list[tuple]:
    """Remove dominated alpha vectors of ``V(pi) = min_k (a_k pi + b_k)``.

    Vector ``j`` is dominated if some ``i`` has ``b_i <= b_j`` and
    ``a_i + b_i <= a_j + b_j`` (linear functions: check the two endpoints).
    Duplicate slopes keep the smallest intercept.
    """
    best = {}
    for v in vectors:
        a, b = float(v[0]), float(v[1])
        key = round(a, 15)
        if key not in best or b < best[key][1]:
            best[key] = (a, b) + tuple(v[2:])
    kept = list(best.values())
    out = []
    for j, vj in enumerate(kept):
        dominated = False
        for i, vi in enumerate(kept):
            if i == j:
                continue
            if vi[1] <= vj[1] + 1e-12 and vi[0] + vi[1] <= vj[0] + vj[1] + 1e-12:
                dominated = True
                break
        if not dominated:
            out.append(vj)
    return out


def exact_alpha_vectors(
    actions: list[dict],
    horizon: int,
    c10: float,
    c01: float,
    max_tuples: int = 4_000_000,
) -> dict:
    """Exact finite-horizon POMDP value function (alpha-vector recursion).

    ``V_h(pi)`` is piecewise-linear concave; the continuation of action
    ``a`` is ``c(a) + sum_y P(y|pi) V_{h-1}(pi')`` which is linear in
    ``pi`` with slope ``sum_y [alpha_k(y) p1_y + beta_k(y) (p1_y - p0_y)]``
    and intercept ``c(a) + sum_y beta_k(y) p0_y`` over tuples of vectors of
    ``V_{h-1}``.  The recursion is exact; the vector count grows
    exponentially in the horizon, so ``horizon`` stays small for the
    oracle.  Each vector carries ``(slope, intercept, branch, action)``
    where ``branch`` is ``"stop0"``, ``"stop1"`` or ``"continue"``.
    """
    terminal = [
        (-float(c10), float(c10), "stop1", None),
        (float(c01), 0.0, "stop0", None),
    ]
    vectors = _prune_alpha_vectors(terminal)
    history = [vectors]
    for _ in range(horizon):
        new = list(terminal)
        for ai, act in enumerate(actions):
            p0 = np.asarray(act["p0"], dtype=float)
            p1 = np.asarray(act["p1"], dtype=float)
            y = len(p0)
            cost = float(act.get("cost", 0.0))
            for combo in itertools.product(range(len(vectors)), repeat=y):
                a = 0.0
                b = cost
                for k in range(y):
                    vk = vectors[combo[k]]
                    a += vk[0] * p1[k] + vk[1] * (p1[k] - p0[k])
                    b += vk[1] * p0[k]
                new.append((a, b, "continue", ai))
            if len(new) > max_tuples:
                new = _prune_alpha_vectors(new)
        vectors = _prune_alpha_vectors(new)
        history.append(vectors)
    return {
        "vectors": vectors,
        "history": history,
        "horizon": horizon,
        "actions": actions,
    }


def belief_from_log_odds(l: float) -> float:
    """``pi = 1 / (1 + exp(-l))``."""
    if l >= 0.0:
        return 1.0 / (1.0 + np.exp(-l))
    return np.exp(l) / (1.0 + np.exp(l))


def _terminal_costs(l: float, c10: float, c01: float):
    pi = belief_from_log_odds(l)
    return (float(c01 * pi), float(c10 * (1.0 - pi)))


def grid_bellman_value(
    actions: list[dict],
    horizon: int,
    c10: float,
    c01: float,
    grid: int = 401,
    l_max: float = 10.0,
) -> dict:
    """Dense-grid value iteration on the log-odds axis.

    ``V_h(l) = min( stop0, stop1, min_a [ c(a) + E_Y V_{h-1}(l + llr_a(y)) ] )``
    with linear interpolation of ``V_{h-1}`` on the grid; the expectation
    over ``Y`` is exact given the interpolation.  Verified against
    ``exact_alpha_vectors`` on small instances.
    """
    ls = np.linspace(-l_max, l_max, grid)
    v = np.minimum(
        c01 * (1.0 / (1.0 + np.exp(-ls))),
        c10 * (1.0 / (1.0 + np.exp(ls))),
    )
    values = [v]
    prune_stats = []
    for _ in range(horizon):
        v_new = v.copy()
        kept = value_bound_prune(actions, ls, v, c10, c01)
        prune_stats.append(len(actions) - len(kept))
        for act in kept:
            p0 = np.asarray(act["p0"], dtype=float)
            p1 = np.asarray(act["p1"], dtype=float)
            llr = np.asarray(act["llr"], dtype=float)
            cont = _interp_expected(v, ls, p0, p1, llr)
            cont += float(act.get("cost", 0.0))
            v_new = np.minimum(v_new, cont)
        v = v_new
        values.append(v)
    return {"ls": ls, "v": v, "values": values, "horizon": horizon,
            "prune_stats": prune_stats}


def _interp_expected(v: NDArray[np.float64], ls: NDArray[np.float64],
                     p0: NDArray[np.float64], p1: NDArray[np.float64],
                     llr: NDArray[np.float64]) -> NDArray[np.float64]:
    """``E_Y V(l + llr_a(Y))`` over the grid, weighted by ``P(y | l)``."""
    cont = np.zeros(len(ls))
    pi = 1.0 / (1.0 + np.exp(-ls))
    for k in range(len(p0)):
        target = np.clip(ls + llr[k], ls[0], ls[-1])
        idx = np.clip(np.searchsorted(ls, target), 1, len(ls) - 1)
        x0, x1 = ls[idx - 1], ls[idx]
        w = (target - x0) / np.maximum(x1 - x0, 1e-300)
        base = v[idx - 1] * (1.0 - w) + v[idx] * w
        cont += base * (pi * p1[k] + (1.0 - pi) * p0[k])
    return cont


def value_bound_prune(actions: list[dict], ls: NDArray[np.float64],
                      v_next: NDArray[np.float64], c10: float, c01: float,
                      tol: float = 1e-9) -> list[dict]:
    """Level-2 pruning of the Blackwell hierarchy (advice/003 section 8):
    exact action elimination against the current value function.

    An action ``a`` whose continuation ``c(a) + E_Y V(l + llr_a(Y))`` never
    beats the terminal stop cost ``min(c01 pi, c10 (1 - pi))`` anywhere on
    the log-odds grid can never improve the Bellman update at this step, so
    it is dropped.  The elimination is exact (the min is unchanged) and
    re-evaluated every step, because an action dominated against ``V_{h-1}``
    may matter against ``V_{h'}``.
    """
    term = np.minimum(
        c01 * (1.0 / (1.0 + np.exp(-ls))),
        c10 * (1.0 / (1.0 + np.exp(ls))),
    )
    kept = []
    for act in actions:
        p0 = np.asarray(act["p0"], dtype=float)
        p1 = np.asarray(act["p1"], dtype=float)
        llr = np.asarray(act["llr"], dtype=float)
        cont = _interp_expected(v_next, ls, p0, p1, llr) \
            + float(act.get("cost", 0.0))
        if np.any(cont < term - tol):
            kept.append(act)
    return kept


def _cost_tokens(act: dict) -> int:
    """Integer budget tokens of an action; the budget axis requires integer
    costs (every library action in the gate is an integer unit count)."""
    cost = float(act.get("cost", 0.0))
    tokens = int(round(cost))
    if abs(cost - tokens) > 1e-9:
        raise ValueError(
            f"budget-state Bellman needs integer costs, got {cost}"
        )
    return tokens


def budget_bellman_value(
    actions: list[dict],
    horizon: int,
    budget: int,
    c10: float,
    c01: float,
    grid: int = 201,
    l_max: float = 8.0,
) -> dict:
    """Exact finite-horizon Bellman over (log-odds, remaining budget).

    State ``(l, b)``: ``l`` the posterior log-odds, ``b`` the remaining
    observation budget in integer cost units (advice/003 Gate D1:
    ``V_t(pi, B)``).  The recursion

        V_h(l, b) = min( stop0, stop1,
                         min_{a: c(a) <= b} [ c(a) + E_Y V_{h-1}(l + llr_a(Y), b - c(a)) ] )

    with ``V_0`` the terminal stop costs.  When ``b`` is smaller than every
    observation cost the continuation set is empty and the controller must
    stop; stopping itself consumes no budget.  The value grid over budget is
    exact given the log-odds interpolation; with ``budget`` large enough to
    pay every action the values coincide with ``grid_bellman_value``.
    """
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if not actions:
        raise ValueError("at least one action is required")
    ls = np.linspace(-l_max, l_max, grid)
    terminal = np.minimum(
        c01 * (1.0 / (1.0 + np.exp(-ls))),
        c10 * (1.0 / (1.0 + np.exp(ls))),
    )
    values = np.zeros((horizon + 1, budget + 1, grid))
    values[0, :, :] = terminal[None, :]
    prune_stats = []
    for h in range(1, horizon + 1):
        h_eliminations = 0
        for b in range(budget + 1):
            v_cur = terminal.copy()
            # level-2 pruning: an action whose continuation against the
            # next value never beats the terminal cost is dropped; the
            # same-budget next value makes the elimination conservative
            # (values[h-1, b-c] >= values[h-1, b]), so it stays exact
            kept = value_bound_prune(actions, ls, values[h - 1, b],
                                     c10, c01)
            h_eliminations += len(actions) - len(kept)
            for act in kept:
                c = _cost_tokens(act)
                if c > b:
                    continue
                p0 = np.asarray(act["p0"], dtype=float)
                p1 = np.asarray(act["p1"], dtype=float)
                llr = np.asarray(act["llr"], dtype=float)
                cont = _interp_expected(
                    values[h - 1, b - c], ls, p0, p1, llr,
                )
                v_cur = np.minimum(v_cur, cont + c)
            values[h, b] = v_cur
        prune_stats.append(h_eliminations)
    return {
        "ls": ls,
        "values": values,
        "horizon": horizon,
        "budget": budget,
        "actions": actions,
        "prune_stats": prune_stats,
    }


def budget_bellman_policy(v_budget: dict, actions: list[dict],
                          c10: float, c01: float):
    """Policy from the budget value: at ``(l, step, b_remaining)`` the
    argmin over stop0/stop1/continuations with ``c(a) <= b_remaining``."""
    ls = v_budget["ls"]
    values = v_budget["values"]
    horizon = v_budget["horizon"]
    max_budget = v_budget["budget"]

    def policy(l: float, step: int, b_remaining: float):
        rem = int(np.clip(horizon - step, 0, horizon))
        b = int(np.clip(int(round(b_remaining)), 0, max_budget))
        v = values[rem, b]
        l = float(np.clip(l, ls[0], ls[-1]))
        pi = belief_from_log_odds(l)
        v_stop0 = c01 * pi
        v_stop1 = c10 * (1.0 - pi)
        best_v = min(v_stop0, v_stop1)
        best = -1 if v_stop0 <= v_stop1 else -2
        for ai, act in enumerate(actions):
            c = _cost_tokens(act)
            if c > b:
                continue
            val = _action_value(act, l, ls, v, pi) + c
            if val < best_v - 1e-12:
                best_v = val
                best = ai
        return best

    return policy


def rollout_budget(
    policy,
    actions: list[dict],
    budget: int,
    true_h: int,
    n_runs: int,
    l0: float = 0.0,
    seed: int = 0,
    max_steps: int = 64,
) -> dict:
    """Monte-Carlo evaluation for a budget-aware policy with signature
    ``policy(l, step, b_remaining) -> action_idx | -1 | -2``.

    Every observation drains ``c(a)`` from the remaining budget; once the
    budget cannot pay any action the controller is forced to stop (the sign
    of the log-odds decides).  Returns the same summary as :func:`rollout`
    (mean delay, realized P_FA/P_MD, mean observation cost, mean
    accumulated information).
    """
    rng = np.random.default_rng(seed)
    delays = np.zeros(n_runs)
    costs = np.zeros(n_runs)
    infos = np.zeros(n_runs)
    declared_h1 = 0
    for r in range(n_runs):
        l = l0
        b = float(budget)
        cost = 0.0
        info = 0.0
        t = 0
        choice = None
        while t < max_steps:
            choice = policy(l, t, b)
            if choice in (-1, -2):
                break
            act = actions[choice]
            c = float(act.get("cost", 0.0))
            if c > b + 1e-12:
                choice = -2 if l > 0.0 else -1
                break
            cost += c
            b -= c
            info += float(act["i_plus"] if true_h == 1 else act["i_minus"])
            p = act["p1"] if true_h == 1 else act["p0"]
            y = int(rng.choice(len(p), p=p))
            l += float(act["llr"][y])
            t += 1
        else:
            # horizon exhausted without stopping: decide by the sign of l
            choice = -2 if l > 0.0 else -1
        delays[r] = float(t)
        costs[r] = cost
        infos[r] = info
        if choice == -2:
            declared_h1 += 1
    if true_h == 1:
        p_md = float(1.0 - declared_h1 / n_runs)
        p_fa = float("nan")
    else:
        p_fa = float(declared_h1 / n_runs)
        p_md = float("nan")
    return {
        "mean_delay": float(delays.mean()),
        "p_fa": p_fa,
        "p_md": p_md,
        "mean_cost": float(costs.mean()),
        "mean_info": float(infos.mean()),
    }


def _interp_value(v: NDArray[np.float64], ls: NDArray[np.float64],
                  l: float) -> float:
    """Scalar linear interpolation of the value grid at log-odds ``l``."""
    i = int(np.clip(int(np.searchsorted(ls, l)), 1, len(ls) - 1))
    w = (l - ls[i - 1]) / max(ls[i] - ls[i - 1], 1e-300)
    return float(v[i - 1] * (1.0 - w) + v[i] * w)


def residual_adaptive_policy(
    v_grid: dict,
    actions: list[dict],
    c10: float,
    c01: float,
    horizon: int,
    residual_margin: float = 0.25,
    explore_rounds: int = 2,
    warmup: int = 20,
):
    """(advice/003 section 4) Bellman-residual-triggered adaptation.

    The wrapper runs the nominal value policy and tracks the realized
    Bellman residual of every played observation

        r_t = c(a_t) + V(l_{t+1}) - V(l_t)

    with ``V`` the precomputed grid values at the remaining horizons (and,
    for a budget-state ``v_grid``, the remaining budget).  The residual is
    not hypothesis-free: the realized draw comes from one hypothesis while
    the value is a belief mixture, so the monitor standardizes ``r_t``
    against *both* model-conditional distributions of the played action,

        z_H = (r_t - mu_H) / sigma_H,
        mu_H = c + E_H[V(l + llr)] - V(l),   sigma_H^2 = Var_H[V(l + llr)],

    and accumulates the running means ``mean_H`` (converging estimators,
    unlike an EMA).  In a stream from a single hypothesis the model-mean of
    the active hypothesis's z is zero, so ``tau = min(|mean_0|, |mean_1|)``
    converges to zero under the correct model; when the model is unreliable
    both conditional standardizations shift and ``tau`` grows.  The trigger
    fires at ``tau > residual_margin`` and switches the controller to the
    robust one-step lookahead (``dpd``); if the mismatch persists for
    ``explore_rounds`` steps it switches to epistemic exploration (the
    maximum information-per-cost action).  Hysteresis returns the
    controller to the nominal mode once ``tau`` drops below half the
    threshold.

    Returns ``(policy, monitor)`` where ``policy(l, step, b_remaining)``
    has the budget-aware signature and ``monitor`` exposes the two running
    means, the trigger statistic ``tau``, the current mode and the
    per-step trace.
    """
    ls = v_grid["ls"]
    values = np.asarray(v_grid["values"], dtype=float)
    if values.ndim == 3:
        budget_max = values.shape[1] - 1

        def value_at(h, b, l):
            v = values[int(np.clip(h, 0, horizon)),
                       int(np.clip(int(round(b)), 0, budget_max))]
            return _interp_value(v, ls, l)
    else:
        def value_at(h, b, l):
            v = values[int(np.clip(h, 0, horizon))]
            return _interp_value(v, ls, l)

    monitor = {
        "mean_0": 0.0,
        "mean_1": 0.0,
        "tau": 0.0,
        "mode": "bellman",
        "triggered": False,
        "residuals": [],
        "z_values": [],
        "modes": [],
        "observations": 0,
    }
    state = {
        "l_before": None, "action": None, "cost": 0.0, "step": 0,
        "b_before": 0.0, "b_after": 0.0,
    }

    def _terminal_decision(l, pi):
        return -1 if c01 * pi <= c10 * (1.0 - pi) else -2

    def _conditional_stats(act, p_cond, h_after, b_after, l_before):
        """``(mu_H, sigma_H)`` of the one-step residual under a hypothesis:
        ``mu = c + E_H[V] - V_before``, ``sigma^2 = Var_H[V(l + llr)]``."""
        v_before = value_at(
            horizon - state["step"], state["b_before"], l_before)
        vals = np.asarray([
            value_at(h_after, b_after, float(l_before) + llr_k)
            for llr_k in act["llr"]
        ], dtype=float)
        p = np.asarray(p_cond, dtype=float)
        e_v = float(np.sum(p * vals))
        var_v = float(np.sum(p * (vals - e_v) ** 2))
        mu = float(act.get("cost", 0.0)) + e_v - v_before
        return mu, max(float(np.sqrt(var_v)), 1e-9)

    def _residual_update(l_after):
        """Close the previous observation: standardize the realized
        residual against both model-conditional distributions and update
        the two running means."""
        act = state["action"]
        h_after = horizon - (state["step"] + 1)
        v_before = value_at(
            horizon - state["step"], state["b_before"], state["l_before"])
        r = state["cost"] + value_at(h_after, state["b_after"], l_after) \
            - v_before
        mu0, sig0 = _conditional_stats(
            act, act["p0"], h_after, state["b_after"], state["l_before"])
        mu1, sig1 = _conditional_stats(
            act, act["p1"], h_after, state["b_after"], state["l_before"])
        z0 = (r - mu0) / sig0
        z1 = (r - mu1) / sig1
        n = monitor["observations"]
        monitor["observations"] = n + 1
        monitor["mean_0"] = (n * monitor["mean_0"] + z0) / (n + 1)
        monitor["mean_1"] = (n * monitor["mean_1"] + z1) / (n + 1)
        monitor["residuals"].append(float(r))
        monitor["z_values"].append((float(z0), float(z1)))
        if n + 1 <= warmup:
            mode = monitor["mode"]
            monitor["modes"].append(mode)
            state["l_before"] = None
            return
        mode = monitor["mode"]
        if mode == "bellman" and tau(monitor["mean_0"], monitor["mean_1"]) \
                > residual_margin:
            mode = "robust"
            state["mode_steps"] = 0
        elif mode in ("robust", "explore") \
                and tau(monitor["mean_0"], monitor["mean_1"]) \
                < residual_margin / 2.0:
            mode = "bellman"
        monitor["mode"] = mode
        monitor["modes"].append(mode)
        state["l_before"] = None

    def tau(ema0, ema1):
        return min(abs(ema0), abs(ema1))

    def _one_step(l, step, b):
        rem = int(np.clip(horizon - step, 0, horizon))
        pi = belief_from_log_odds(float(l))
        v_stop0 = c01 * pi
        v_stop1 = c10 * (1.0 - pi)
        best_v = min(v_stop0, v_stop1)
        best = _terminal_decision(l, pi)
        for ai, act in enumerate(actions):
            val = 0.0
            for k in range(len(act["p0"])):
                pi1 = belief_from_log_odds(float(l) + act["llr"][k])
                val += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) \
                    * min(c01 * pi1, c10 * (1.0 - pi1))
            val += float(act.get("cost", 0.0))
            if val < best_v - 1e-12:
                best_v = val
                best = ai
        return best

    def _explore(l, step, b):
        rem = int(np.clip(horizon - step, 0, horizon))
        v_rem = values[rem] if values.ndim == 2 else values[rem, :, :]
        pi = belief_from_log_odds(float(l))
        v_stop0 = c01 * pi
        v_stop1 = c10 * (1.0 - pi)
        best = _terminal_decision(l, pi)
        scores = [
            act["i_plus"] / max(float(act.get("cost", 1e-12)), 1e-12)
            for act in actions
        ]
        for ai in np.argsort(-np.asarray(scores)):
            act = actions[int(ai)]
            c = float(act.get("cost", 0.0))
            if c > b + 1e-12:
                continue
            val = _action_value(act, l, ls, v_rem, pi) + c
            if val < min(v_stop0, v_stop1) - 1e-12:
                return int(ai)
        return best

    def policy(l: float, step: int, b_remaining: float):
        nonlocal state
        if step == 0:
            # fresh episode: no pending observation from the previous run
            state["l_before"] = None
        if state["l_before"] is not None:
            _residual_update(l)
        monitor["tau"] = tau(monitor["mean_0"], monitor["mean_1"])
        monitor["triggered"] = monitor["tau"] > residual_margin
        if monitor["mode"] == "robust":
            if state.get("mode_steps", 0) >= explore_rounds:
                decision = _explore(l, step, b_remaining)
                if decision >= 0:
                    monitor["mode"] = "explore"
            else:
                decision = _one_step(l, step, b_remaining)
        elif monitor["mode"] == "explore":
            decision = _explore(l, step, b_remaining)
        else:
            rem = int(np.clip(horizon - step, 0, horizon))
            if values.ndim == 3:
                b = int(np.clip(int(round(b_remaining)), 0,
                                int(values.shape[1] - 1)))
                v = values[rem, b, :]
            else:
                v = values[rem]
            l_c = float(np.clip(l, ls[0], ls[-1]))
            pi = belief_from_log_odds(l_c)
            v_stop0 = c01 * pi
            v_stop1 = c10 * (1.0 - pi)
            best_v = min(v_stop0, v_stop1)
            decision = _terminal_decision(l_c, pi)
            for ai, act in enumerate(actions):
                c = float(act.get("cost", 0.0))
                if values.ndim == 3 and c > b + 1e-12:
                    continue
                val = _action_value(act, l_c, ls, v, pi) + c
                if val < best_v - 1e-12:
                    best_v = val
                    decision = ai
        if decision >= 0:
            act = actions[decision]
            cost = float(act.get("cost", 0.0))
            state.update({
                "l_before": float(l),
                "action": act,
                "cost": cost,
                "step": step,
                "b_before": float(b_remaining),
                "b_after": max(float(b_remaining) - cost, 0.0),
            })
            if monitor["mode"] == "robust":
                state["mode_steps"] = state.get("mode_steps", 0) + 1
        else:
            state["l_before"] = None
            monitor["modes"].append(monitor["mode"])
        return decision

    return policy, monitor


def rollout_mismatch(
    policy,
    model_actions: list[dict],
    true_actions: list[dict],
    true_h: int,
    n_runs: int,
    l0: float = 0.0,
    seed: int = 0,
    max_steps: int = 64,
    budget: int | None = None,
) -> dict:
    """Monte-Carlo evaluation under kernel mismatch.

    The environment draws observations from ``true_actions`` (indexed by
    the policy's chosen action) while the controller's belief update uses
    the model LLRs from ``model_actions``; ``budget`` optionally drains the
    observation budget as in :func:`rollout_budget`.  Returns the same
    summary as :func:`rollout` with the *realized* (true-kernel) errors and
    costs.
    """
    if len(model_actions) != len(true_actions):
        raise ValueError("model and true action libraries must match")
    rng = np.random.default_rng(seed)
    delays = np.zeros(n_runs)
    costs = np.zeros(n_runs)
    infos = np.zeros(n_runs)
    declared_h1 = 0
    for r in range(n_runs):
        l = l0
        b = float(budget) if budget is not None else float("inf")
        cost = 0.0
        info = 0.0
        t = 0
        choice = None
        while t < max_steps:
            if budget is not None:
                choice = policy(l, t, b)
            else:
                choice = policy(l, t)
            if choice in (-1, -2):
                break
            model_act = model_actions[choice]
            true_act = true_actions[choice]
            c = float(model_act.get("cost", 0.0))
            if c > b + 1e-12:
                choice = -2 if l > 0.0 else -1
                break
            cost += c
            b -= c
            info += float(true_act["i_plus"] if true_h == 1
                          else true_act["i_minus"])
            p = true_act["p1"] if true_h == 1 else true_act["p0"]
            y = int(rng.choice(len(p), p=p))
            l += float(model_act["llr"][y])
            t += 1
        else:
            choice = -2 if l > 0.0 else -1
        delays[r] = float(t)
        costs[r] = cost
        infos[r] = info
        if choice == -2:
            declared_h1 += 1
    if true_h == 1:
        p_md = float(1.0 - declared_h1 / n_runs)
        p_fa = float("nan")
    else:
        p_fa = float(declared_h1 / n_runs)
        p_md = float("nan")
    return {
        "mean_delay": float(delays.mean()),
        "p_fa": p_fa,
        "p_md": p_md,
        "mean_cost": float(costs.mean()),
        "mean_info": float(infos.mean()),
    }


# ---------------------------------------------------------------------------
# Policies and Monte-Carlo evaluation
# ---------------------------------------------------------------------------

def rollout(
    policy,
    actions: list[dict],
    horizon: int,
    true_h: int,
    n_runs: int,
    l0: float = 0.0,
    seed: int = 0,
    max_steps: int = 64,
) -> dict:
    """Monte-Carlo evaluation of ``policy(l, step) -> (action_idx | -1, -2)``.

    ``-1`` means stop-declare-H0, ``-2`` stop-declare-H1.  Returns the mean
    delay, the realized P_FA (declared H1 under H0) and P_MD (declared H0
    under H1), the mean accumulated information ``sum I+`` (and ``I-``)
    under the respective hypothesis, and the mean observation cost.
    """
    rng = np.random.default_rng(seed)
    delays = np.zeros(n_runs)
    costs = np.zeros(n_runs)
    infos = np.zeros(n_runs)
    declared_h1 = 0
    for r in range(n_runs):
        l = l0
        cost = 0.0
        info = 0.0
        t = 0
        while t < max_steps:
            choice = policy(l, t)
            if choice in (-1, -2):
                break
            act = actions[choice]
            cost += float(act.get("cost", 0.0))
            info += float(act["i_plus"] if true_h == 1 else act["i_minus"])
            p = act["p1"] if true_h == 1 else act["p0"]
            y = int(rng.choice(len(p), p=p))
            l += float(act["llr"][y])
            t += 1
        else:
            # horizon exhausted without stopping: decide by the sign of l
            choice = -2 if l > 0.0 else -1
        delays[r] = float(t)
        costs[r] = cost
        infos[r] = info
        if choice == -2:
            declared_h1 += 1
    if true_h == 1:
        p_md = float(1.0 - declared_h1 / n_runs)
        p_fa = float("nan")
    else:
        p_fa = float(declared_h1 / n_runs)
        p_md = float("nan")
    return {
        "mean_delay": float(delays.mean()),
        "p_fa": p_fa,
        "p_md": p_md,
        "mean_cost": float(costs.mean()),
        "mean_info": float(infos.mean()),
    }


def bellman_policy(ls: NDArray[np.float64], v: NDArray[np.float64],
                   actions: list[dict], horizon: int,
                   terminal_costs: tuple[float, float],
                   grid_l_max: float = 10.0):
    """Policy from the grid value function: argmin over
    stop0/stop1/continue at the current log-odds."""
    c10, c01 = terminal_costs
    l = ls
    pi = belief_from_log_odds(l)
    v_stop0 = c01 * pi
    v_stop1 = c10 * (1.0 - pi)
    v_cont = np.full(len(l), np.inf)
    for ai, act in enumerate(actions):
        p0 = act["p0"]
        p1 = act["p1"]
        llr = act["llr"]
        cost = act.get("cost", 0.0)
        val = np.full(len(l), 0.0)
        for k in range(len(p0)):
            target = np.clip(l + llr[k], -grid_l_max, grid_l_max)
            idx = np.searchsorted(ls, target)
            idx = np.clip(idx, 1, len(ls) - 1)
            x0, x1 = ls[idx - 1], ls[idx]
            w = (target - x0) / np.maximum(x1 - x0, 1e-300)
            base = v[idx - 1] * (1.0 - w) + v[idx] * w
            val += base * (pi * p1[k] + (1.0 - pi) * p0[k])
        v_cont = np.minimum(v_cont, val + cost)
    return v_stop0, v_stop1, v_cont


def tau_pred_policy(actions: list[dict], a_log_odds: float,
                    b_log_odds: float):
    """Wald-boundary policy with myopic ``I+/cost`` action ranking.

    The essence of the existing ``tau_pred`` scheduler: while the belief
    stays between the boundaries, spend the next observation on the action
    with the largest information-per-cost, ``I+`` being the asymptotic
    expected LLR drift under H1 (the first-order approximation of the
    Bellman policy in the high-confidence regime).
    """
    i_plus = np.array([a["i_plus"] for a in actions])
    cost = np.array([max(a.get("cost", 1.0), 1e-12) for a in actions])
    order = np.argsort(-(i_plus / cost))

    def policy(l: float, step: int):
        if l >= a_log_odds:
            return -2
        if l <= b_log_odds:
            return -1
        return int(order[0])

    return policy


def chernoff_policy(actions: list[dict], a_log_odds: float,
                    b_log_odds: float):
    """Wald-boundary policy with myopic Chernoff/cost action ranking."""
    c = np.array([a["chernoff"] for a in actions])
    cost = np.array([max(a.get("cost", 1.0), 1e-12) for a in actions])
    order = np.argsort(-(c / cost))

    def policy(l: float, step: int):
        if l >= a_log_odds:
            return -2
        if l <= b_log_odds:
            return -1
        return int(order[0])

    return policy


def dpd_policy(actions: list[dict], c10: float, c01: float):
    """One-step Bayesian lookahead (``Delta P_D``-style myopic policy).

    Chooses the action minimizing ``c(a) + E_Y[V_0(l + llr(Y))]`` with
    ``V_0`` the one-shot decision cost; stops when the one-shot decision is
    cheaper than every continuation.
    """

    def policy(l: float, step: int):
        pi = belief_from_log_odds(l)
        v_stop0 = c01 * pi
        v_stop1 = c10 * (1.0 - pi)
        best = None
        best_v = np.inf
        for ai, act in enumerate(actions):
            val = 0.0
            for k in range(len(act["p0"])):
                l1 = l + act["llr"][k]
                pi1 = belief_from_log_odds(l1)
                val += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) \
                    * min(c01 * pi1, c10 * (1.0 - pi1))
            val += act.get("cost", 0.0)
            if val < best_v:
                best_v = val
                best = ai
        if min(v_stop0, v_stop1) <= best_v:
            return -1 if v_stop0 <= v_stop1 else -2
        return int(best)

    return policy


def static_policy(actions: list[dict], a_log_odds: float,
                  b_log_odds: float):
    """Fixed best action (by Chernoff per cost) with Wald boundaries."""
    c = np.array([a["chernoff"] for a in actions])
    cost = np.array([max(a.get("cost", 1.0), 1e-12) for a in actions])
    best = int(np.argmax(c / cost))

    def policy(l: float, step: int):
        if l >= a_log_odds:
            return -2
        if l <= b_log_odds:
            return -1
        return best

    return policy


def _action_value(act: dict, l: float, ls: NDArray[np.float64],
                  v: NDArray[np.float64], pi: float) -> float:
    """``E_Y[V(l + llr_a(Y))]`` at log-odds ``l`` with belief ``pi``."""
    val = 0.0
    for k in range(len(act["p0"])):
        target = float(np.clip(l + act["llr"][k], ls[0], ls[-1]))
        j = int(np.clip(int(np.searchsorted(ls, target)), 1, len(ls) - 1))
        x0, x1 = ls[j - 1], ls[j]
        w = (target - x0) / max(x1 - x0, 1e-300)
        val += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) \
            * (v[j - 1] * (1.0 - w) + v[j] * w)
    return val


def bellman_action_policy(v_grid: dict, actions: list[dict],
                          c10: float, c01: float):
    """Policy from the grid Bellman value: stop when a terminal branch is
    optimal, otherwise play the argmin continuation action; at step ``t``
    the value used is ``V_{horizon - t}``."""
    ls = v_grid["ls"]
    values = v_grid["values"]
    horizon = v_grid["horizon"]

    def policy(l: float, step: int):
        rem = int(np.clip(horizon - step, 0, horizon))
        v = values[rem]
        l_clip = float(np.clip(l, ls[0], ls[-1]))
        pi = belief_from_log_odds(l_clip)
        v_stop0 = c01 * pi
        v_stop1 = c10 * (1.0 - pi)
        best_v = min(v_stop0, v_stop1)
        best = -1 if v_stop0 <= v_stop1 else -2
        for ai, act in enumerate(actions):
            val = _action_value(act, l_clip, ls, v, pi) \
                + float(act.get("cost", 0.0))
            if val < best_v - 1e-12:
                best_v = val
                best = ai
        return best

    return policy


# ---------------------------------------------------------------------------
# Information-theoretic lower bounds (advice section 9)
# ---------------------------------------------------------------------------

def bin_divergence(x: float, y: float) -> float:
    """``d(x || y) = x log(x/y) + (1-x) log((1-x)/(1-y))``."""
    x = np.clip(x, 1e-300, 1.0 - 1e-300)
    y = np.clip(y, 1e-300, 1.0 - 1e-300)
    return float(x * np.log(x / y) + (1.0 - x) * np.log((1.0 - x) / (1.0 - y)))


def information_lower_bounds(
    p_fa: float,
    p_md: float,
    actions: list[dict],
) -> dict:
    """Sequential-testing information lower bounds.

    For any detector with ``P_FA <= alpha`` and ``P_MD <= beta`` the
    accumulated information satisfies (advice eqs. 5-6)

        E_1[ sum_t I+_{a_t} ] >= d(1-beta || alpha),
        E_0[ sum_t I-_{a_t} ] >= d(1-alpha || beta),

    so with ``I+ <= I_max+`` and efficiency ``rho^+ = max_a I_a+/c(a)``
    (eqs. 7-8)

        E_1[T] >= d(1-beta || alpha) / I_max+,
        E_1[C] >= d(1-beta || alpha) / rho^+.

    The realized error probabilities are passed in; the bounds are returned
    for direct comparison with rolled-out mean delays and costs.
    """
    alpha = float(p_fa)
    beta = float(p_md)
    i_max_plus = max(float(a["i_plus"]) for a in actions)
    rho_plus = max(
        float(a["i_plus"]) / max(float(a.get("cost", 1.0)), 1e-12)
        for a in actions
    )
    return {
        "d_1": bin_divergence(1.0 - beta, alpha),
        "d_0": bin_divergence(1.0 - alpha, beta),
        "i_max_plus": float(i_max_plus),
        "rho_plus": float(rho_plus),
        "t1_lower": bin_divergence(1.0 - beta, alpha) / max(i_max_plus, 1e-12),
        "c1_lower": bin_divergence(1.0 - beta, alpha) / max(rho_plus, 1e-12),
    }


# ---------------------------------------------------------------------------
# Multi-target: Lagrangian decomposition (advice section 6-7)
# ---------------------------------------------------------------------------

def dual_decomposed_value(
    actions: list[dict],
    horizon: int,
    c10: float,
    c01: float,
    nu: float,
    lam: float,
    grid: int = 201,
    l_max: float = 10.0,
) -> dict:
    """Per-target Bellman with prices ``(nu, lam)``.

    ``V(l) = min( stop0, stop1, min_a [ nu + lam c(a) + E_Y V(l + llr) ] )``
    -- the ``nu`` per-step term is the fairness price of one more delay
    step, ``lam`` the resource price.  The per-cycle scheduler then plays
    actions with the largest net value ``nu * DV - lam c(a)`` under the
    budget (advice eq. 4).
    """
    ls = np.linspace(-l_max, l_max, grid)
    v = np.minimum(
        c01 * (1.0 / (1.0 + np.exp(-ls))),
        c10 * (1.0 / (1.0 + np.exp(ls))),
    )
    values = [v]
    for _ in range(horizon):
        v_new = v.copy()
        for act in actions:
            p0 = np.asarray(act["p0"], dtype=float)
            p1 = np.asarray(act["p1"], dtype=float)
            llr = np.asarray(act["llr"], dtype=float)
            cont = _interp_expected(v, ls, p0, p1, llr)
            cont += nu + lam * float(act.get("cost", 0.0))
            v_new = np.minimum(v_new, cont)
        v = v_new
        values.append(v)
    return {"ls": ls, "v": v, "values": values, "horizon": horizon,
            "nu": nu, "lam": lam}


def decomposed_scheduler(
    v_q: list[dict],
    actions_per_target: list[list[dict]],
    budget: int,
    c10: float,
    c01: float,
) -> callable:
    """Per-cycle policy: per-target G-value knapsack under the budget.

    ``G_{q,a} = nu_q * DV_{q,a}(l_q) - lam * c(a)`` with
    ``DV = V_q(l_q) - E_Y[V_q(l_q + llr)]``; a target stops when its best
    ``G`` (including the stopping action) is nonpositive or stopping beats
    every continuation.  The per-target best actions are then selected as a
    knapsack under the budget (small ``Q``: exhaustive).
    """
    n_q = len(v_q)

    def policy(l_vec, step):
        candidates = []
        for q in range(n_q):
            ls = v_q[q]["ls"]
            values = v_q[q]["values"]
            horizon = v_q[q]["horizon"]
            rem = int(np.clip(horizon - step, 0, horizon))
            v = values[rem]
            nu = v_q[q]["nu"]
            lam = v_q[q]["lam"]
            l = float(np.clip(l_vec[q], ls[0], ls[-1]))
            pi = belief_from_log_odds(l)
            v_stop0 = c01 * pi
            v_stop1 = c10 * (1.0 - pi)
            best_a = None
            best_g = -np.inf
            for ai, act in enumerate(actions_per_target[q]):
                val = _action_value(act, l, ls, v, pi)
                i = int(np.clip(int(np.searchsorted(ls, l)), 0, len(ls) - 1))
                dv = float(v[i]) - val
                g = nu * dv - lam * float(act.get("cost", 0.0))
                if g > best_g:
                    best_g = g
                    best_a = (ai, g, float(act.get("cost", 0.0)))
            candidates.append((q, best_a, v_stop0, v_stop1))
        # targets with a positive best G want to observe; the rest declare.
        continue_acts = [c for c in candidates if c[1] is not None and c[1][1] > 0]
        n_cont = len(continue_acts)
        best_mask = 0
        best_value = 0.0
        for mask in range(1 << n_cont):
            cost_sum = 0.0
            value = 0.0
            for j in range(n_cont):
                if mask & (1 << j):
                    cost_sum += continue_acts[j][1][2]
                    value += continue_acts[j][1][1]
            if cost_sum <= budget and value > best_value:
                best_value = value
                best_mask = mask
        decisions = []
        for q in range(n_q):
            ai = None
            for j in range(n_cont):
                if continue_acts[j][0] == q and (best_mask & (1 << j)):
                    ai = continue_acts[j][1][0]
                    break
            if ai is not None:
                decisions.append(ai)
            else:
                v_stop0, v_stop1 = candidates[q][2], candidates[q][3]
                want_stop = candidates[q][1] is None or candidates[q][1][1] <= 0
                if want_stop:
                    decisions.append(-1 if v_stop0 <= v_stop1 else -2)
                else:
                    decisions.append(-3)  # budget tight: wait, keep belief
        return decisions

    return policy


def rollout_multi(
    policy,
    actions_per_target: list[list[dict]],
    true_h: list[int],
    n_runs: int,
    l0: float = 0.0,
    seed: int = 0,
    max_steps: int = 40,
) -> dict:
    """Monte-Carlo multi-target evaluation: worst-target mean delay and the
    per-target delays."""
    rng = np.random.default_rng(seed)
    q = len(true_h)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    costs = np.zeros((n_runs, q))
    for r in range(n_runs):
        l = np.full(q, l0)
        stopped = np.zeros(q, dtype=bool)
        t = 0
        while not stopped.all() and t < max_steps:
            choices = policy(l, t)
            for qq in range(q):
                if stopped[qq]:
                    continue
                choice = choices[qq]
                if choice == -3:
                    continue
                if choice in (-1, -2):
                    stopped[qq] = True
                    delays[r, qq] = t
                    if choice == -2:
                        declared_h1[r, qq] = 1
                    continue
                act = actions_per_target[qq][choice]
                costs[r, qq] += float(act.get("cost", 0.0))
                p = act["p1"] if true_h[qq] == 1 else act["p0"]
                y = int(rng.choice(len(p), p=p))
                l[qq] += float(act["llr"][y])
            t += 1
    p_fa = [float(declared_h1[:, qq].mean()) if true_h[qq] == 0
            else float("nan") for qq in range(q)]
    p_md = [float(1.0 - declared_h1[:, qq].mean()) if true_h[qq] == 1
            else float("nan") for qq in range(q)]
    return {
        "mean_worst_delay": float(delays.max(axis=1).mean()),
        "mean_delays": [float(delays[:, qq].mean()) for qq in range(q)],
        "mean_costs": [float(costs[:, qq].mean()) for qq in range(q)],
        "p_fa": p_fa,
        "p_md": p_md,
    }


# ---------------------------------------------------------------------------
# Multi-target: exact joint Bellman (small Q, product grid)
# ---------------------------------------------------------------------------

def _interp2(v2: NDArray[np.float64], l1: float, l2: float,
             ls: NDArray[np.float64]) -> float:
    """Bilinear interpolation of the product-grid value at ``(l1, l2)``."""
    i1 = int(np.clip(int(np.searchsorted(ls, l1)), 1, len(ls) - 1))
    i2 = int(np.clip(int(np.searchsorted(ls, l2)), 1, len(ls) - 1))
    w1 = (l1 - ls[i1 - 1]) / max(ls[i1] - ls[i1 - 1], 1e-300)
    w2 = (l2 - ls[i2 - 1]) / max(ls[i2] - ls[i2 - 1], 1e-300)
    return (v2[i1 - 1, i2 - 1] * (1.0 - w1) * (1.0 - w2)
            + v2[i1, i2 - 1] * w1 * (1.0 - w2)
            + v2[i1 - 1, i2] * (1.0 - w1) * w2
            + v2[i1, i2] * w1 * w2)


def _grid_value(single: dict, h: int, l: float) -> float:
    """Single-target value with ``h`` steps remaining at log-odds ``l``."""
    v = single["values"][int(np.clip(h, 0, single["horizon"]))]
    ls = single["ls"]
    l = float(np.clip(l, ls[0], ls[-1]))
    i = int(np.clip(int(np.searchsorted(ls, l)), 1, len(ls) - 1))
    w = (l - ls[i - 1]) / max(ls[i] - ls[i - 1], 1e-300)
    return v[i - 1] * (1.0 - w) + v[i] * w


def _terminal(l: float, c10: float, c01: float) -> float:
    pi = belief_from_log_odds(l)
    return min(c01 * pi, c10 * (1.0 - pi))


def _terminal_decision(l: float, c10: float, c01: float) -> int:
    pi = belief_from_log_odds(l)
    return -1 if c01 * pi <= c10 * (1.0 - pi) else -2


def joint_bellman_value(
    actions_per_target: list[list[dict]],
    horizon: int,
    c10: float,
    c01: float,
    grid: int = 61,
    l_max: float = 8.0,
) -> dict:
    """Exact joint Bellman on the product log-odds grid (``Q = 2``).

    One move per cycle: stop both targets, stop one target (the other
    continues under its single-target optimal value with the remaining
    horizon), or take one action on one target.  The recursion

    ``V_h(l1, l2) = min( term1+term2,
                         term1 + V^{only2}_{h-1}(l2),
                         term2 + V^{only1}_{h-1}(l1),
                         min_{q,a} c(a) + E_Y V_{h-1}(l_q + llr, l_other) )``

    is exact for this move set; ``V^{only q}_h`` are the single-target grid
    values.  The per-target grid ``singles`` are kept for the policy.
    """
    singles = [
        grid_bellman_value(actions, horizon, c10, c01, grid=grid, l_max=l_max)
        for actions in actions_per_target
    ]
    ls = np.linspace(-l_max, l_max, grid)
    v0 = np.array([[_terminal(ls[i1], c10, c01) + _terminal(ls[i2], c10, c01)
                    for i2 in range(grid)] for i1 in range(grid)])
    values = [v0]
    for h in range(1, horizon + 1):
        vh = v0.copy()
        for i1 in range(grid):
            for i2 in range(grid):
                l1, l2 = ls[i1], ls[i2]
                vh[i1, i2] = min(
                    vh[i1, i2],
                    _terminal(l1, c10, c01) + _grid_value(singles[1], h - 1, l2),
                )
                vh[i1, i2] = min(
                    vh[i1, i2],
                    _terminal(l2, c10, c01) + _grid_value(singles[0], h - 1, l1),
                )
        for q in range(2):
            for act in actions_per_target[q]:
                p0 = np.asarray(act["p0"], dtype=float)
                p1 = np.asarray(act["p1"], dtype=float)
                llr = np.asarray(act["llr"], dtype=float)
                cost = float(act.get("cost", 0.0))
                prev = values[h - 1]
                for i1 in range(grid):
                    for i2 in range(grid):
                        l1, l2 = ls[i1], ls[i2]
                        l_q = l1 if q == 0 else l2
                        pi = belief_from_log_odds(l_q)
                        exp = 0.0
                        for k in range(len(p0)):
                            target = float(np.clip(l_q + llr[k], ls[0], ls[-1]))
                            if q == 0:
                                exp += (pi * p1[k] + (1.0 - pi) * p0[k]) \
                                    * _interp2(prev, target, l2, ls)
                            else:
                                exp += (pi * p1[k] + (1.0 - pi) * p0[k]) \
                                    * _interp2(prev, l1, target, ls)
                        vh[i1, i2] = min(vh[i1, i2], cost + exp)
        values.append(vh)
    return {"ls": ls, "v": values[-1], "values": values,
            "horizon": horizon, "singles": singles}


def _single_decision(single: dict, actions: list[dict], h: int, l: float,
                     c10: float, c01: float) -> int:
    """Argmin decision (stop/action) of the single-target value ``V_h``."""
    if h <= 0:
        return _terminal_decision(l, c10, c01)
    v = single["values"][int(np.clip(h, 0, single["horizon"]))]
    ls = single["ls"]
    l = float(np.clip(l, ls[0], ls[-1]))
    pi = belief_from_log_odds(l)
    v_stop0 = c01 * pi
    v_stop1 = c10 * (1.0 - pi)
    best_v = min(v_stop0, v_stop1)
    best = -1 if v_stop0 <= v_stop1 else -2
    for ai, act in enumerate(actions):
        val = _action_value(act, l, ls, v, pi) + float(act.get("cost", 0.0))
        if val < best_v - 1e-12:
            best_v = val
            best = ai
    return best


def joint_bellman_policy(v_joint: dict, actions_per_target: list[list[dict]],
                         c10: float, c01: float):
    """Policy from ``joint_bellman_value``: at each cycle the argmin branch
    (stop both / stop one + delegate / observe one action); once a target
    stops, its peer is delegated to the single-target optimal policy."""
    ls = v_joint["ls"]
    values = v_joint["values"]
    singles = v_joint["singles"]
    horizon = v_joint["horizon"]
    stopped = set()

    def policy(l_vec, step):
        if step == 0:
            stopped.clear()
        rem = int(np.clip(horizon - step, 0, horizon))
        v = values[rem]
        l1 = float(np.clip(l_vec[0], ls[0], ls[-1]))
        l2 = float(np.clip(l_vec[1], ls[0], ls[-1]))
        if 0 in stopped:
            return [-3, _single_decision(singles[1], actions_per_target[1],
                                         rem - 1, l2, c10, c01)]
        if 1 in stopped:
            return [_single_decision(singles[0], actions_per_target[0],
                                     rem - 1, l1, c10, c01), -3]
        best_v = _terminal(l1, c10, c01) + _terminal(l2, c10, c01)
        best_dec = [_terminal_decision(l1, c10, c01),
                    _terminal_decision(l2, c10, c01)]
        # stop one target now; the peer continues with rem-1 steps
        val = _terminal(l1, c10, c01) + _grid_value(singles[1], rem - 1, l2)
        if val < best_v - 1e-12:
            best_v = val
            best_dec = [_terminal_decision(l1, c10, c01),
                        _single_decision(singles[1], actions_per_target[1],
                                         rem - 1, l2, c10, c01)]
        val = _terminal(l2, c10, c01) + _grid_value(singles[0], rem - 1, l1)
        if val < best_v - 1e-12:
            best_v = val
            best_dec = [_single_decision(singles[0], actions_per_target[0],
                                         rem - 1, l1, c10, c01),
                        _terminal_decision(l2, c10, c01)]
        # observe one action on one target
        for q in (0, 1):
            l_q = l1 if q == 0 else l2
            pi = belief_from_log_odds(l_q)
            for ai, act in enumerate(actions_per_target[q]):
                exp = 0.0
                for k in range(len(act["p0"])):
                    target = float(np.clip(l_q + act["llr"][k], ls[0], ls[-1]))
                    if q == 0:
                        exp += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) \
                            * _interp2(v, target, l2, ls)
                    else:
                        exp += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) \
                            * _interp2(v, l1, target, ls)
                val = float(act.get("cost", 0.0)) + exp
                if val < best_v - 1e-12:
                    best_v = val
                    best_dec = [-3, -3]
                    best_dec[q] = ai
        for qq in range(2):
            if best_dec[qq] in (-1, -2):
                stopped.add(qq)
        return best_dec

    return policy

# ---------------------------------------------------------------------------
# Objective-aligned detection-delay control (advice/004, Gate D2)
# ---------------------------------------------------------------------------
# The cost-based Bellman of Gate D1 minimizes sampling cost + Bayesian
# decision error, but the system objective is a *constrained detection
# delay*:  min_Pi max_q E_1[T_q]  s.t.  P_FA <= alpha, P_MD <= beta.
# advice/004 therefore rebuilds the dynamic objective (P0) and replaces the
# P_D(n) checkpoint with a numerically calibrated two-threshold stopping
# rule (P1).  The per-cycle continuation cost is exactly 1 (one detection
# cycle), error declarations are priced by dual prices (xi for false
# alarms, zeta for misses) calibrated to meet the error constraints, and
# the observation cost is priced by ``lam``.


def delay_value_iteration(
    actions: list[dict],
    horizon: int,
    budget: int,
    xi: float,
    zeta: float,
    lam: float = 1.0,
    grid: int = 201,
    l_max: float = 8.0,
    cycle_cost: float = 1.0,
    bounds: tuple | None = None,
) -> dict:
    """Objective-aligned single-target Bellman (advice/004 eq. 1).

    ``V_h(l, b) = min( zeta * pi(l),            -- declare H0 (miss priced)
                       xi * (1 - pi(l)),        -- declare H1 (FA priced)
                       cycle_cost + min_{a: c(a) <= b} [ lam * c(a)
                               + E_Y V_{h-1}(l + llr_a(Y), b - c(a)) ] )``

    with ``V_0`` the forced terminal decision.  The continuation branch
    costs ``cycle_cost`` (one detection cycle) per observation, so the
    value is a detection-delay objective (E_1[T]) plus the priced error
    and resource terms; the dual prices ``(xi, zeta)`` are calibrated to
    meet ``P_FA <= alpha``, ``P_MD <= beta`` (see
    :func:`calibrate_delay_prices`).  In the joint oracle the per-cycle
    cost of target ``q`` is its min-max weight ``nu_q``.
    """
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if not actions:
        raise ValueError("at least one action is required")
    ls = np.linspace(-l_max, l_max, grid)
    pi = 1.0 / (1.0 + np.exp(-ls))
    if bounds is not None:
        # constraint-embedded stopping: declare (cost 0) exactly when the
        # calibrated two-threshold rule fires; declaring inside the band is
        # infeasible, so the value is a pure constrained detection delay
        a_bound, b_bound = float(bounds[0]), float(bounds[1])
        terminal = np.where((ls >= a_bound) | (ls <= b_bound), 0.0, 1e9)
    else:
        terminal = np.minimum(zeta * pi, xi * (1.0 - pi))
    values = np.zeros((horizon + 1, budget + 1, grid))
    values[0, :, :] = terminal[None, :]
    for h in range(1, horizon + 1):
        for b in range(budget + 1):
            v_cur = terminal.copy()
            kept = value_bound_prune(actions, ls, values[h - 1, b],
                                     xi, zeta)
            for act in kept:
                c = _cost_tokens(act)
                if c > b:
                    continue
                p0 = np.asarray(act["p0"], dtype=float)
                p1 = np.asarray(act["p1"], dtype=float)
                llr = np.asarray(act["llr"], dtype=float)
                cont = _interp_expected(
                    values[h - 1, b - c], ls, p0, p1, llr,
                )
                v_cur = np.minimum(v_cur, cycle_cost + lam * c + cont)
            values[h, b] = v_cur
    return {"ls": ls, "values": values, "horizon": horizon,
            "budget": budget, "actions": actions, "xi": xi, "zeta": zeta,
            "lam": lam, "cycle_cost": cycle_cost}


def delay_policy(v_delay: dict, actions: list[dict], xi: float, zeta: float,
                 lam: float = 1.0):
    """Policy from the delay value: at ``(l, step, b_remaining)`` the argmin
    over declare-H0 / declare-H1 / one more observation cycle."""
    ls = v_delay["ls"]
    values = v_delay["values"]
    horizon = v_delay["horizon"]
    budget_max = v_delay["budget"]

    def policy(l: float, step: int, b_remaining: float):
        rem = int(np.clip(horizon - step, 0, horizon))
        b = int(np.clip(int(round(b_remaining)), 0, budget_max))
        v = values[rem, b]
        l_c = float(np.clip(l, ls[0], ls[-1]))
        pi = belief_from_log_odds(l_c)
        v_stop0 = zeta * pi
        v_stop1 = xi * (1.0 - pi)
        best_v = min(v_stop0, v_stop1)
        best = -1 if v_stop0 <= v_stop1 else -2
        for ai, act in enumerate(actions):
            c = _cost_tokens(act)
            if c > b:
                continue
            val = _action_value(act, l_c, ls, v, pi) + lam * c + 1.0
            if val < best_v - 1e-12:
                best_v = val
                best = ai
        return best

    return policy


def sprt_boundary_policy(actions: list[dict], a_bound: float,
                         b_bound: float, selector=None):
    """True two-threshold sequential test (advice/004 P1):

    ``l >= A`` -> declare H1, ``l <= B`` -> declare H0, otherwise observe
    with ``selector(l, step, b_remaining)`` (default: the best
    information-per-cost action).  The boundaries are the *numerically
    calibrated* ``(A*, B*)`` (Wald's approximation is not exact for the
    quantized + BSC + erasure kernels).
    """

    def policy(l: float, step: int, b_remaining: float):
        if l >= a_bound:
            return -2
        if l <= b_bound:
            return -1
        if selector is not None:
            chosen = selector(l, step, b_remaining)
            if chosen is not None and chosen >= 0:
                return chosen
        scores = [
            act["i_plus"] / max(float(act.get("cost", 1e-12)), 1e-12)
            for act in actions
        ]
        return int(np.argmax(scores))

    return policy


def delay_action_selector(v_delay: dict, actions: list[dict], xi: float,
                          zeta: float, lam: float = 1.0):
    """Action part of the delay policy: at ``(l, step, b)`` the argmin
    continuation action of the detection-delay value (stopping is handled
    by the calibrated two-threshold rule)."""

    def selector(l: float, step: int, b_remaining: float):
        ls = v_delay["ls"]
        values = v_delay["values"]
        horizon = v_delay["horizon"]
        budget_max = v_delay["budget"]
        rem = int(np.clip(horizon - step, 0, horizon))
        b = int(np.clip(int(round(b_remaining)), 0, budget_max))
        v = values[rem, b]
        l_c = float(np.clip(l, ls[0], ls[-1]))
        pi = belief_from_log_odds(l_c)
        best = None
        best_v = np.inf
        for ai, act in enumerate(actions):
            c = _cost_tokens(act)
            if c > b:
                continue
            val = _action_value(act, l_c, ls, v, pi) + lam * c + 1.0
            if val < best_v:
                best_v = val
                best = ai
        return best

    return selector


def _evaluate_single(policy, actions, budget, alpha, beta, n_runs, seed):
    """MC errors and H1 detection delay of a budget-aware single-target
    policy (``policy(l, step, b)``); ``E_1[T]`` is the primary metric."""
    r0 = rollout_budget(policy, actions, budget, 0, n_runs=n_runs,
                        seed=seed)
    r1 = rollout_budget(policy, actions, budget, 1, n_runs=n_runs,
                        seed=seed + 1)
    return {
        "e1_delay": float(r1["mean_delay"]),
        "e0_delay": float(r0["mean_delay"]),
        "p_fa": float(r0["p_fa"]),
        "p_md": float(r1["p_md"]),
        "mean_cost": 0.5 * (r0["mean_cost"] + r1["mean_cost"]),
    }


def calibrate_delay_prices(
    actions: list[dict],
    alpha: float,
    beta: float,
    horizon: int,
    budget: int,
    lam: float = 1.0,
    grid: int = 201,
    l_max: float = 8.0,
    n_runs: int = 1500,
    seed: int = 0,
    price_grid: tuple = (16.0, 32.0, 64.0, 128.0, 256.0),
    mc_tol: float = 0.008,
) -> dict:
    """Dual calibration of ``(xi, zeta)`` (advice/004 section 2).

    Scans the price grid, evaluates each delay-Bellman policy by Monte
    Carlo, and returns the prices whose realized ``P_FA <= alpha +
    mc_tol`` and ``P_MD <= beta + mc_tol`` (MC noise tolerance) minimize
    the H1 detection delay ``E_1[T]``; the best candidate is re-evaluated
    with a larger sample for the reported errors.  The ratio ``xi/zeta``
    sets the decision boundary (``A = log(xi/zeta)``), the scale decides
    how many cycles are worth paying before declaring.
    """
    best = None
    for xi in price_grid:
        for zeta in price_grid:
            v = delay_value_iteration(
                actions, horizon, budget, xi, zeta, lam,
                grid=grid, l_max=l_max,
            )
            pol = delay_policy(v, actions, xi, zeta, lam)
            row = _evaluate_single(pol, actions, budget, alpha, beta,
                                   n_runs, seed)
            row.update({"xi": float(xi), "zeta": float(zeta)})
            if row["p_fa"] <= alpha + mc_tol \
                    and row["p_md"] <= beta + mc_tol:
                if best is None or row["e1_delay"] < best["e1_delay"]:
                    best = row
    if best is None:
        raise ValueError(
            f"no (xi, zeta) on the price grid meets P_FA <= {alpha}, "
            f"P_MD <= {beta}"
        )
    v = delay_value_iteration(
        actions, horizon, budget, best["xi"], best["zeta"], lam,
        grid=grid, l_max=l_max,
    )
    best["policy"] = delay_policy(v, actions, best["xi"], best["zeta"], lam)
    best["value"] = v
    fine = _evaluate_single(best["policy"], actions, budget, alpha, beta,
                            n_runs=max(n_runs, 4000), seed=seed + 7)
    for key in ("e1_delay", "e0_delay", "p_fa", "p_md", "mean_cost"):
        best[key] = fine[key]
    return best


def calibrate_sprt_boundaries(
    actions: list[dict],
    alpha: float,
    beta: float,
    budget: int,
    n_runs: int = 1500,
    seed: int = 10,
    margin: float = 1.0,
    points: int = 7,
    selector=None,
    selector_factory=None,
) -> dict:
    """Numerical calibration of the two thresholds ``(A*, B*)``
    (advice/004 P1).  Scans ``A`` and ``B`` around the Wald values
    ``log((1-beta)/alpha)`` and ``log(beta/(1-alpha))`` and returns the
    boundaries whose realized errors meet the constraints with the smallest
    H1 detection delay.  ``selector`` customizes the in-band action choice
    (default: information-per-cost); ``selector_factory(a_bound, b_bound)``
    rebuilds the selector per candidate boundary (for boundary-dependent
    selectors such as one-step crossing probabilities).
    """
    a_wald = float(np.log((1.0 - beta) / alpha))
    b_wald = float(np.log(beta / (1.0 - alpha)))
    a_grid = np.unique(np.concatenate([
        np.linspace(a_wald - margin, a_wald + margin, points),
        [a_wald],
    ]))
    b_grid = np.unique(np.concatenate([
        np.linspace(b_wald - margin, b_wald + margin, points),
        [b_wald],
    ]))
    best = None
    best_selector = None
    for a_bound in a_grid:
        for b_bound in b_grid:
            if a_bound <= b_bound:
                continue
            sel = (selector_factory(float(a_bound), float(b_bound))
                   if selector_factory is not None else selector)
            pol = sprt_boundary_policy(actions, float(a_bound),
                                       float(b_bound), selector=sel)
            row = _evaluate_single(pol, actions, budget, alpha, beta,
                                   n_runs, seed)
            row.update({"a_bound": float(a_bound), "b_bound": float(b_bound)})
            if row["p_fa"] <= alpha + 1e-9 and row["p_md"] <= beta + 1e-9:
                if best is None or row["e1_delay"] < best["e1_delay"]:
                    best = row
                    best_selector = sel
    if best is None:
        raise ValueError(
            f"no (A, B) on the calibration grid meets P_FA <= {alpha}, "
            f"P_MD <= {beta}"
        )
    best["policy"] = sprt_boundary_policy(actions, best["a_bound"],
                                          best["b_bound"],
                                          selector=best_selector)
    return best


def joint_delay_value(
    actions_per_target: list[list[dict]],
    horizon: int,
    budget: int,
    xi: float,
    zeta: float,
    lam: float = 1.0,
    grid: int = 33,
    l_max: float = 8.0,
    nu: tuple = (0.5, 0.5),
    bounds: tuple | None = None,
) -> dict:
    """Exact joint sequential oracle (advice/004 Gate D2-B, Q = 2).

    State ``(l1, l2, b)`` with one move per cycle: stop both targets,
    stop one target (the peer continues under its single-target delay
    value with the remaining budget), or take one action on one target
    (cost drains ``b``).  The per-cycle continuation cost of target ``q``
    is its min-max weight ``nu_q`` (``sum_q nu_q = 1``), so the value is
    the weighted total detection delay ``sum_q nu_q E[T_q]`` (the
    min-max proxy ``max_q E[T_q] = max_nu sum nu_q E[T_q]``).

    With ``bounds`` given (per-target calibrated two thresholds ``(A_q,
    B_q)``), stopping is constraint-embedded: a target may declare (cost
    0) exactly when ``l_q >= A_q`` or ``l_q <= B_q``, and declaring inside
    the band is infeasible -- the value is then a pure constrained
    detection delay and the decision rule of the rollout is exactly the
    boundary rule (no price-based stopping mismatch).
    """
    budget = int(budget)
    single_grid = max(grid, 101)
    singles = [
        delay_value_iteration(actions, horizon, budget, xi, zeta, lam,
                              grid=single_grid, l_max=l_max,
                              cycle_cost=float(nu[q]),
                              bounds=None if bounds is None else bounds[q])
        for q, actions in enumerate(actions_per_target)
    ]
    ls = np.linspace(-l_max, l_max, grid)
    cycle_cost = float(nu[0]) + float(nu[1])
    pi_grid = 1.0 / (1.0 + np.exp(-ls))

    if bounds is None:
        def terminal(l):
            pi = belief_from_log_odds(float(l))
            return min(zeta * pi, xi * (1.0 - pi))

        t0 = np.minimum(zeta * pi_grid, xi * (1.0 - pi_grid))
        t1 = t0
    else:
        def terminal(l, q):
            a_bound, b_bound = float(bounds[q][0]), float(bounds[q][1])
            return 0.0 if (l >= a_bound or l <= b_bound) else 1e9

        t0 = np.where((ls >= float(bounds[0][0]))
                      | (ls <= float(bounds[0][1])), 0.0, 1e9)
        t1 = np.where((ls >= float(bounds[1][0]))
                      | (ls <= float(bounds[1][1])), 0.0, 1e9)

    v0 = np.zeros((budget + 1, grid, grid))
    for b in range(budget + 1):
        v0[b] = t0[:, None] + t1[None, :]
    values = [v0]
    for h in range(1, horizon + 1):
        vh = v0.copy()
        for b in range(budget + 1):
            # stop one target now; the peer continues with h-1 cycles and
            # the same remaining budget (vectorized over the grid)
            for q in range(2):
                single = singles[q]
                s_vals = single["values"][
                    int(np.clip(h - 1, 0, single["horizon"])), b]
                sls = single["ls"]
                peer = np.interp(ls, sls, s_vals)  # (grid,)
                if q == 0:
                    vh[b] = np.minimum(
                        vh[b], t0[:, None] + peer[None, :])
                else:
                    vh[b] = np.minimum(
                        vh[b], peer[:, None] + t1[None, :])
            # observe one action on one target (vectorized over the grid)
            for q in range(2):
                pi_q = pi_grid[:, None] if q == 0 else pi_grid[None, :]
                for act in actions_per_target[q]:
                    p0 = np.asarray(act["p0"], dtype=float)
                    p1 = np.asarray(act["p1"], dtype=float)
                    llr = np.asarray(act["llr"], dtype=float)
                    c = _cost_tokens(act)
                    if c > b:
                        continue
                    prev = values[h - 1][b - c]
                    cont = np.zeros((grid, grid))
                    for k in range(len(p0)):
                        if q == 0:
                            target = np.broadcast_to(
                                np.clip(ls[:, None] + llr[k],
                                        ls[0], ls[-1]),
                                (grid, grid))
                            idx = np.clip(np.searchsorted(ls, target),
                                          1, grid - 1)
                            w = (target - ls[idx - 1]) / np.maximum(
                                ls[idx] - ls[idx - 1], 1e-300)
                            base = (np.take_along_axis(prev, idx - 1, 0)
                                    * (1.0 - w)
                                    + np.take_along_axis(prev, idx, 0) * w)
                        else:
                            target = np.broadcast_to(
                                np.clip(ls[None, :] + llr[k],
                                        ls[0], ls[-1]),
                                (grid, grid))
                            idx = np.clip(np.searchsorted(ls, target),
                                          1, grid - 1)
                            w = (target - ls[idx - 1]) / np.maximum(
                                ls[idx] - ls[idx - 1], 1e-300)
                            base = (np.take_along_axis(prev, idx - 1, 1)
                                    * (1.0 - w)
                                    + np.take_along_axis(prev, idx, 1) * w)
                        cont += base * (pi_q * p1[k]
                                        + (1.0 - pi_q) * p0[k])
                    vh[b] = np.minimum(vh[b], cycle_cost + lam * c + cont)
        values.append(vh)
    return {"ls": ls, "v": values[-1], "values": values,
            "horizon": horizon, "budget": budget, "singles": singles,
            "xi": xi, "zeta": zeta, "lam": lam, "nu": nu,
            "bounds": bounds}


def joint_delay_policy(v_joint: dict, actions_per_target: list[list[dict]],
                       xi: float, zeta: float, lam: float = 1.0):
    """Policy from the joint delay value: each cycle the argmin branch
    (stop both / stop one + delegate / observe one action); once a target
    stops, its peer is delegated to the single-target delay policy.  With
    constraint-embedded bounds the stopping rule is exactly the calibrated
    two-threshold rule."""
    ls = v_joint["ls"]
    values = v_joint["values"]
    singles = v_joint["singles"]
    bounds = v_joint.get("bounds")
    horizon = v_joint["horizon"]
    budget_max = v_joint["budget"]
    stopped = set()

    if bounds is None:
        def terminal(l):
            pi = belief_from_log_odds(float(l))
            return min(zeta * pi, xi * (1.0 - pi))

        def terminal_decision(l):
            pi = belief_from_log_odds(float(l))
            return -1 if zeta * pi <= xi * (1.0 - pi) else -2
    else:
        def terminal(l, q):
            a_bound, b_bound = float(bounds[q][0]), float(bounds[q][1])
            return 0.0 if (l >= a_bound or l <= b_bound) else 1e9

        def terminal_decision(l, q):
            a_bound, b_bound = float(bounds[q][0]), float(bounds[q][1])
            if l >= a_bound:
                return -2
            if l <= b_bound:
                return -1
            return None  # infeasible inside the band

    def single_val(single, h, b, l):
        values = single["values"]
        v = values[int(np.clip(h, 0, single["horizon"])),
                   int(np.clip(int(b), 0, single["budget"]))]
        sls = single["ls"]
        i = int(np.clip(int(np.searchsorted(sls, l)), 1, len(sls) - 1))
        w = (l - sls[i - 1]) / max(sls[i] - sls[i - 1], 1e-300)
        return v[i - 1] * (1.0 - w) + v[i] * w

    def policy(l_vec, step, b_remaining):
        if step == 0:
            stopped.clear()
        rem = int(np.clip(horizon - step, 0, horizon))
        b = int(np.clip(int(round(b_remaining)), 0, budget_max))
        v = values[rem][b]
        l1 = float(np.clip(l_vec[0], ls[0], ls[-1]))
        l2 = float(np.clip(l_vec[1], ls[0], ls[-1]))
        if 0 in stopped:
            dpol1 = delay_policy(singles[1], actions_per_target[1],
                                 xi, zeta, lam)
            return [-3, dpol1(l2, step, b_remaining)]
        if 1 in stopped:
            dpol0 = delay_policy(singles[0], actions_per_target[0],
                                 xi, zeta, lam)
            return [dpol0(l1, step, b_remaining), -3]
        if bounds is not None:
            d0 = terminal_decision(l1, 0)
            d1 = terminal_decision(l2, 1)
        else:
            d0 = terminal_decision(l1)
            d1 = terminal_decision(l2)
        if d0 is None or d1 is None:
            # at least one target is inside its band: stopping it now is
            # infeasible, so the stop branches are disabled
            best_v = float("inf")
            best_dec = None
        else:
            if bounds is not None:
                best_v = terminal(l1, 0) + terminal(l2, 1)
            else:
                best_v = terminal(l1) + terminal(l2)
            best_dec = [d0, d1]
        if bounds is not None:
            t1_v = terminal(l1, 0)
            t2_v = terminal(l2, 1)
        else:
            t1_v = terminal(l1)
            t2_v = terminal(l2)
        if d0 is not None:
            val = t1_v + single_val(singles[1], rem - 1, b, l2)
            if val < best_v - 1e-12:
                best_v = val
                best_dec = [d0,
                            _single_delay_decision(singles[1],
                                                   actions_per_target[1],
                                                   xi, zeta, lam,
                                                   rem - 1, l2,
                                                   b_remaining)]
        if d1 is not None:
            val = t2_v + single_val(singles[0], rem - 1, b, l1)
            if val < best_v - 1e-12:
                best_v = val
                best_dec = [_single_delay_decision(singles[0],
                                                   actions_per_target[0],
                                                   xi, zeta, lam,
                                                   rem - 1, l1,
                                                   b_remaining),
                            d1]
        for q in (0, 1):
            l_q = l1 if q == 0 else l2
            pi = belief_from_log_odds(l_q)
            for ai, act in enumerate(actions_per_target[q]):
                c = _cost_tokens(act)
                if c > b:
                    continue
                exp = 0.0
                for k in range(len(act["p0"])):
                    target = float(
                        np.clip(l_q + act["llr"][k], ls[0], ls[-1]))
                    if q == 0:
                        exp += (pi * act["p1"][k]
                                + (1.0 - pi) * act["p0"][k]) \
                            * _interp2(v, target, l2, ls)
                    else:
                        exp += (pi * act["p1"][k]
                                + (1.0 - pi) * act["p0"][k]) \
                            * _interp2(v, l1, target, ls)
                val = sum(float(x) for x in v_joint["nu"]) + lam * c + exp
                if val < best_v - 1e-12:
                    best_v = val
                    best_dec = [-3, -3]
                    best_dec[q] = ai
        for qq in range(2):
            if best_dec is not None and best_dec[qq] in (-1, -2):
                stopped.add(qq)
        if best_dec is None:
            return [-3, -3]
        return best_dec

    return policy


def _single_delay_decision(single: dict, actions: list[dict], xi: float,
                           zeta: float, lam: float, h: int, l: float,
                           b_remaining: float) -> int:
    """Argmin decision of the single-target delay value at ``(h, l, b)``."""
    if h <= 0:
        pi = belief_from_log_odds(float(l))
        return -1 if zeta * pi <= xi * (1.0 - pi) else -2
    ls = single["ls"]
    values = single["values"]
    budget_max = single["budget"]
    b = int(np.clip(int(round(b_remaining)), 0, budget_max))
    v = values[int(np.clip(h, 0, single["horizon"])), b]
    l_c = float(np.clip(l, ls[0], ls[-1]))
    pi = belief_from_log_odds(l_c)
    v_stop0 = zeta * pi
    v_stop1 = xi * (1.0 - pi)
    best_v = min(v_stop0, v_stop1)
    best = -1 if v_stop0 <= v_stop1 else -2
    for ai, act in enumerate(actions):
        c = _cost_tokens(act)
        if c > b:
            continue
        val = _action_value(act, l_c, ls, v, pi) + lam * c + 1.0
        if val < best_v - 1e-12:
            best_v = val
            best = ai
    return best


def rollout_delay_multi(
    policy,
    actions_per_target: list[list[dict]],
    true_h: list[int],
    budget: int,
    n_runs: int,
    l0: float = 0.0,
    seed: int = 0,
    max_steps: int = 40,
) -> dict:
    """Monte-Carlo multi-target evaluation of a budget-aware joint policy
    with signature ``policy(l_vec, step, b_remaining)``.  Returns the
    per-target mean detection delays (under H1: ``e1_delay``), the realized
    per-target P_FA/P_MD and the mean observation cost."""
    rng = np.random.default_rng(seed)
    q = len(true_h)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    costs = np.zeros((n_runs, q))
    for r in range(n_runs):
        l = np.full(q, l0)
        b = float(budget)
        stopped = np.zeros(q, dtype=bool)
        t = 0
        while not stopped.all() and t < max_steps:
            choices = policy(l, t, b)
            for qq in range(q):
                if stopped[qq]:
                    continue
                choice = choices[qq]
                if choice == -3:
                    continue
                if choice in (-1, -2):
                    stopped[qq] = True
                    delays[r, qq] = t
                    if choice == -2:
                        declared_h1[r, qq] = 1
                    continue
                act = actions_per_target[qq][choice]
                c = float(act.get("cost", 0.0))
                if c > b + 1e-12:
                    choice = -2 if l[qq] > 0.0 else -1
                    stopped[qq] = True
                    delays[r, qq] = t
                    if choice == -2:
                        declared_h1[r, qq] = 1
                    continue
                b -= c
                costs[r, qq] += c
                p = act["p1"] if true_h[qq] == 1 else act["p0"]
                y = int(rng.choice(len(p), p=p))
                l[qq] += float(act["llr"][y])
            t += 1
    p_fa = [float(declared_h1[:, qq].mean()) if true_h[qq] == 0
            else float("nan") for qq in range(q)]
    p_md = [float(1.0 - declared_h1[:, qq].mean()) if true_h[qq] == 1
            else float("nan") for qq in range(q)]
    return {
        "mean_worst_delay": float(delays.max(axis=1).mean()),
        "mean_delays": [float(delays[:, qq].mean()) for qq in range(q)],
        "e1_delays": [float(delays[:, qq].mean()) if true_h[qq] == 1
                      else float("nan") for qq in range(q)],
        "mean_costs": [float(costs[:, qq].mean()) for qq in range(q)],
        "p_fa": p_fa,
        "p_md": p_md,
    }



# ---------------------------------------------------------------------------
# Deployable controllers (advice/004 Gate D2-D)
# ---------------------------------------------------------------------------
# The exact joint oracle (joint_delay_value) has an exponential state space
# and is not deployable beyond tiny Q.  This block implements the scalable
# controller family of advice/004 section 6 -- the per-target delay values
# V_q(l_q, b) are computed once (linear in Q), and every cycle a scalar
# index decides which (q, a) to play:
#
#   dual G-value   :  argmax_{q,a} [ nu_q (V_q - (c + E V_q)) - lam * c(a) ]
#   Whittle index  :  argmax_{q,a} [ (V_q - (c + E V_q)) / c(a) ]
#   one-step rollout: argmax_{q,a} of the singles-based joint lookahead
#
# All three are O(Q * |A| * atoms) per cycle with O(Q) memory -- deployable.


def _delay_gain(v_delay, act, l, step, b_remaining):
    """``V(l, b) - [cycle_cost + lam*c + E V(l + llr, b - c)]``: the value
    gain of playing ``act`` now, from the target's delay value."""
    ls = v_delay["ls"]
    values = v_delay["values"]
    horizon = v_delay["horizon"]
    budget_max = v_delay["budget"]
    cycle = float(v_delay.get("cycle_cost", 1.0))
    lam = float(v_delay["lam"])
    rem = int(np.clip(horizon - step, 0, horizon))
    b = int(np.clip(int(round(b_remaining)), 0, budget_max))
    c = _cost_tokens(act)
    b_next = int(np.clip(b - c, 0, budget_max))
    l_c = float(np.clip(l, ls[0], ls[-1]))
    pi = belief_from_log_odds(l_c)
    v_now = float(np.interp(l_c, ls, values[rem, b]))
    exp = 0.0
    for k in range(len(act["p0"])):
        target = float(np.clip(l_c + act["llr"][k], ls[0], ls[-1]))
        v_next = float(np.interp(target, ls, values[rem - 1, b_next]))
        exp += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) * v_next
    return v_now - (cycle + lam * c + exp)


def make_deployable_controllers(actions_per_target, singles, bounds,
                                nu=(0.5, 0.5), lam=1.0, depth=1):
    """Build the deployable per-cycle policies sharing the precomputed
    per-target delay values and the calibrated two-threshold bounds.
    Returns a dict of policies with signature ``(l_vec, step, b)``."""

    def stop_decision(l, q):
        a_bound, b_bound = float(bounds[q][0]), float(bounds[q][1])
        if l >= a_bound:
            return -2
        if l <= b_bound:
            return -1
        return None

    def dual_policy(l_vec, step, b_remaining):
        decisions = [-3] * len(l_vec)
        best = None
        best_g = -np.inf
        for q in range(len(l_vec)):
            d = stop_decision(l_vec[q], q)
            if d is not None:
                decisions[q] = d
                continue
            for ai, act in enumerate(actions_per_target[q]):
                c = _cost_tokens(act)
                if c > b_remaining + 1e-12:
                    continue
                gain = _delay_gain(singles[q], act, l_vec[q], step,
                                   b_remaining)
                g = float(nu[q]) * gain - lam * c
                if g > best_g:
                    best_g = g
                    best = (q, ai)
        if best is None:
            return decisions
        decisions[best[0]] = best[1]
        return decisions

    def whittle_policy(l_vec, step, b_remaining):
        decisions = [-3] * len(l_vec)
        best = None
        best_idx = -np.inf
        for q in range(len(l_vec)):
            d = stop_decision(l_vec[q], q)
            if d is not None:
                decisions[q] = d
                continue
            for ai, act in enumerate(actions_per_target[q]):
                c = _cost_tokens(act)
                if c > b_remaining + 1e-12:
                    continue
                gain = _delay_gain(singles[q], act, l_vec[q], step,
                                   b_remaining)
                idx = gain / max(c, 1e-12)
                if idx > best_idx:
                    best_idx = idx
                    best = (q, ai)
        if best is None:
            return decisions
        decisions[best[0]] = best[1]
        return decisions

    def rollout_policy(l_vec, step, b_remaining):
        """One-step rollout: for every (q, a) estimate the resulting
        joint value as the sum over targets of their single-target
        continuation values -- the peers either wait one cycle (depth 1)
        or play their own best action next cycle (depth 2) -- and play
        the best.  Stopping by the calibrated bounds."""
        q_count = len(l_vec)
        decisions = [-3] * q_count
        any_active = False
        for q in range(q_count):
            d = stop_decision(l_vec[q], q)
            if d is not None:
                decisions[q] = d
            else:
                any_active = True
        if not any_active:
            return decisions
        best = None
        best_v = np.inf
        for q in range(q_count):
            if decisions[q] != -3:
                continue
            for ai, act in enumerate(actions_per_target[q]):
                c = _cost_tokens(act)
                if c > b_remaining + 1e-12:
                    continue
                total = _continuation_value(
                    singles[q], act, l_vec[q], step, b_remaining)
                for qq in range(q_count):
                    if qq == q:
                        continue
                    if stop_decision(l_vec[qq], qq) is not None:
                        continue  # the peer stops at its boundary
                    if depth >= 2:
                        # the peer plays its own best action next cycle
                        total += _best_peer_value(
                            singles[qq], actions_per_target[qq],
                            l_vec[qq], step + 1,
                            b_remaining - c)
                    else:
                        total += _wait_value(
                            singles[qq], l_vec[qq], step, b_remaining)
                if total < best_v:
                    best_v = total
                    best = (q, ai)
        if best is None:
            return decisions
        decisions[best[0]] = best[1]
        return decisions

    def rollout_policy2(l_vec, step, b_remaining):
        """Depth-2 rollout (see rollout_policy with depth >= 2)."""
        return _rollout_depth2(l_vec, step, b_remaining)

    def _rollout_depth2(l_vec, step, b_remaining):
        q_count = len(l_vec)
        decisions = [-3] * q_count
        any_active = False
        for q in range(q_count):
            d = stop_decision(l_vec[q], q)
            if d is not None:
                decisions[q] = d
            else:
                any_active = True
        if not any_active:
            return decisions
        best = None
        best_v = np.inf
        for q in range(q_count):
            if decisions[q] != -3:
                continue
            for ai, act in enumerate(actions_per_target[q]):
                c = _cost_tokens(act)
                if c > b_remaining + 1e-12:
                    continue
                total = _continuation_value(
                    singles[q], act, l_vec[q], step, b_remaining)
                for qq in range(q_count):
                    if qq == q:
                        continue
                    if stop_decision(l_vec[qq], qq) is not None:
                        continue
                    total += _best_peer_value(
                        singles[qq], actions_per_target[qq],
                        l_vec[qq], step + 1, b_remaining - c)
                if total < best_v:
                    best_v = total
                    best = (q, ai)
        if best is None:
            return decisions
        decisions[best[0]] = best[1]
        return decisions

    return {
        "dual_gvalue": dual_policy,
        "whittle_index": whittle_policy,
        "rollout_1step": rollout_policy,
        "rollout_2step": rollout_policy2,
    }


def _best_peer_value(v_delay, actions, l, step, b_remaining):
    """The peer's best one-step continuation value at its state
    (its own action, its own cycle cost), used by the depth-2 rollout."""
    ls = v_delay["ls"]
    values = v_delay["values"]
    horizon = v_delay["horizon"]
    budget_max = v_delay["budget"]
    cycle = float(v_delay.get("cycle_cost", 1.0))
    lam = float(v_delay["lam"])
    rem = int(np.clip(horizon - step, 0, horizon))
    b = int(np.clip(int(round(b_remaining)), 0, budget_max))
    l_c = float(np.clip(l, ls[0], ls[-1]))
    pi = belief_from_log_odds(l_c)
    best = np.inf
    for act in actions:
        c = _cost_tokens(act)
        if c > b:
            continue
        b_next = int(np.clip(b - c, 0, budget_max))
        exp = 0.0
        for k in range(len(act["p0"])):
            target = float(np.clip(l_c + act["llr"][k], ls[0], ls[-1]))
            v_next = float(np.interp(target, ls, values[rem - 1, b_next]))
            exp += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) * v_next
        best = min(best, cycle + lam * c + exp)
    return best


def _continuation_value(v_delay, act, l, step, b_remaining):
    """``cycle + lam*c + E V(l + llr, b - c)`` (observe now)."""
    ls = v_delay["ls"]
    values = v_delay["values"]
    horizon = v_delay["horizon"]
    budget_max = v_delay["budget"]
    cycle = float(v_delay.get("cycle_cost", 1.0))
    lam = float(v_delay["lam"])
    rem = int(np.clip(horizon - step, 0, horizon))
    b = int(np.clip(int(round(b_remaining)), 0, budget_max))
    c = _cost_tokens(act)
    b_next = int(np.clip(b - c, 0, budget_max))
    l_c = float(np.clip(l, ls[0], ls[-1]))
    pi = belief_from_log_odds(l_c)
    exp = 0.0
    for k in range(len(act["p0"])):
        target = float(np.clip(l_c + act["llr"][k], ls[0], ls[-1]))
        v_next = float(np.interp(target, ls, values[rem - 1, b_next]))
        exp += (pi * act["p1"][k] + (1.0 - pi) * act["p0"][k]) * v_next
    return cycle + lam * c + exp


def _wait_value(v_delay, l, step, b_remaining):
    """The value of waiting one cycle (the peer is not observed): its
    delay value at ``rem - 1`` with the same budget (delay accrues)."""
    ls = v_delay["ls"]
    values = v_delay["values"]
    horizon = v_delay["horizon"]
    budget_max = v_delay["budget"]
    rem = int(np.clip(horizon - step, 0, horizon))
    b = int(np.clip(int(round(b_remaining)), 0, budget_max))
    l_c = float(np.clip(l, ls[0], ls[-1]))
    cycle = float(v_delay.get("cycle_cost", 1.0))
    return cycle + float(np.interp(l_c, ls, values[rem - 1, b]))
