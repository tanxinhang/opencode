"""P1 service-delay bridge theory (advice/003, Theorem 4.110/4.111).

The LLR decomposition

    L_q(t)  =  L_q(0) + A_q(t) + M_q(t),

with ``A_q(t) = sum_tau sum_i x_{iq,tau} g_{iq,tau}`` the cumulative
predictable reliable service and ``M_q`` a martingale, holds exactly when
the rule chooses actions as functions of the past (``a_i = pi_i(I_{i,t})``,
P-DIST): given the history, the H1 drift of the delivered LLR increment is
the reliable information ``g_iq`` (Theorem 4.94).  The increments are
bounded by the finite (quantized + erasure) alphabet, ``|Z - E[Z|F]| <=
b_q``, so martingale concentration does not need a Gaussian shortcut --
the Freedman-type two-sided deviation bound

    P( M_q(t) <= -eta ) <= exp[ -eta^2 / (2(V_q(t) + b_q eta / 3)) ]

with ``V_q(t) = sum_tau Var(Z_{q,tau} | F_{tau-1})``.  Stopping: target q
stops on the owner LLR crossing ``A_q*`` (H1) or ``B_q`` (H0); with
``D_q = A_q* - L_q(0)`` and ``beta_q`` the H1 miss-probability budget, if
the predictable service reached ``A_q(t) >= D_q + eta`` while the target
has still not crossed H1, a deviation of size >= eta happened, so

    P_1(T_q > t)  <=  beta_q
        + exp[ - (A_q(t) - D_q)^2 / (2(V_q(t) + b_q (A_q(t) - D_q)/3)) ].

Theorem B (bridge 2, advice/003 section 5): the mirror-descent rule
approaches the demand-normalized relaxation optimum ``z* = max_x min_q
sum_i x_iq g_iq / D_q`` in time average: ``min_q (1/T) sum_t r_q(t) >=
z* - eps_T`` with ``eps_T = O(sqrt(log Q / T))`` in the static
relaxation, plus the distributed-information loss ``eps_loc`` (local
dual disagreement; Gate F0-G9A measured <= ~1.8% delay, and the
normalized-service audit below measures it directly).

All functions are pure and numerically verified by
``scripts/run_delay_bridge_gate.py`` (results/delay_bridge_gate.json)
and ``tests/test_reliable_service_bridge.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def static_relaxation_optimum(g: np.ndarray, d: np.ndarray, eps: float = 0.1,
                              tol: float = 1e-9) -> float:
    """The demand-normalized relaxation optimum of Theorem 4.95
    (``max_{x,z} z`` with ``sum_i x_iq g_iq / (D_q + eps) >= z`` and
    ``sum_q x_iq <= 1``), i.e. the static ``z*`` that Theorem B says the
    mirror-descent time average approaches: ``min_q (1/T) sum_t r_q(t)
    >= z* - eps_T``.  Linear program on ``(x, z)`` (polynomial, no
    enumeration)."""
    k, qq = np.asarray(g, dtype=float).shape
    g = np.asarray(g, dtype=float).clip(min=0.0)
    d = np.asarray(d, dtype=float).clip(min=0.0) + eps
    c = np.zeros(k * qq + 1)
    c[-1] = -1.0                             # maximize z
    A_ub = []
    b_ub = []
    for qq_ in range(qq):
        row = np.zeros(k * qq + 1)
        for i in range(k):
            row[i * qq + qq_] = -g[i, qq_] / d[qq_]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    for i in range(k):
        row = np.zeros(k * qq + 1)
        for qq_ in range(qq):
            row[i * qq + qq_] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
    bounds = [(0.0, 1.0)] * (k * qq) + [(None, None)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=bounds, method="highs")
    if not res.success:
        return 0.0
    return float(max(res.x[-1], 0.0))


def freedman_tail(eta: float, v: float, b: float) -> float:
    """Freedman-type upper bound ``exp(-eta^2 / (2(v + b eta / 3)))``
    for a zero-mean martingale with predictable quadratic variation
    ``v`` and increments bounded in absolute value by ``b`` (Freedman
    1975; the deployed increments are bounded by the finite alphabet, so
    no Gaussian assumption is used)."""
    if eta <= 0.0:
        return 1.0
    denom = 2.0 * (max(v, 0.0) + max(b, 0.0) * eta / 3.0)
    if denom <= 0.0:
        return 1.0
    return float(np.exp(-eta * eta / denom))


def stopping_tail_bound(a_q: float, d_q: float, v_q: float, b_q: float,
                        beta_q: float) -> float:
    """The target stopping tail (Theorem 4.110): ``P_1(T_q > t) <=
    beta_q + exp(...)`` with ``eta = max(A_q(t) - D_q, 0)``.  ``beta_q``
    covers the H1-crossing of the lower threshold (the miss budget); the
    exponential covers the large negative martingale deviation needed
    for the H1 threshold to still be crossed-out at time ``t``."""
    eta = max(float(a_q) - float(d_q), 0.0)
    return float(min(beta_q + freedman_tail(eta, v_q, b_q), 1.0))


def relative_error_bound(freedman_violations: np.ndarray,
                         probabilities: np.ndarray) -> dict:
    """Max-ratio audit of a concentration-like bound family: how often
    and by how much the empirical LHS exceeds the theoretical RHS."""
    probs = np.asarray(probabilities, dtype=float)
    viols = np.asarray(freedman_violations, dtype=float)
    ratio = np.zeros_like(probs)
    safe = probs > 0.0
    ratio[safe] = viols[safe] / np.maximum(probs[safe], 1e-12)
    return {
        "max_ratio": float(np.max(ratio)) if len(ratio) else 0.0,
        "violation_fraction": float(
            np.mean(viols > probs)) if len(viols) else 0.0,
        "n_cases": int(len(probs)),
    }


def martingale_decomposition(
    bridge_out: dict,
    q: int,
    target_alpha: float = 0.05,
    target_beta: float = 0.05,
    max_eta_ratio: float = 0.25,
) -> dict:
    """Verify ``L = A + M`` (advice/003 section 1): (i) the realized
    increment ``Z_t`` equals the recorded drift ``A`` plus a martingale
    residual ``M``; (ii) the empirical deviation tail fits the
    Freedman-type bound at a grid of ``eta`` thresholds; (iii) the
    per-cycle residual is the realized minus predictable increment.

    Distances/``eta`` are in nats (LLR units), ``V`` in nats^2, ``b`` in
    nats -- dimensionless accounting of the quantized+erasure alphabet.
    """
    L = np.asarray(bridge_out["L"])            # (R, T, Q)
    A = np.asarray(bridge_out["A"])
    A_raw = np.asarray(bridge_out.get("A_raw", A))
    V = np.asarray(bridge_out["V"])
    M = np.asarray(bridge_out["M"])
    H = np.asarray(bridge_out["H"])            # (R, Q)
    T_stop = np.asarray(bridge_out["T"])       # (R, Q)
    n_runs, max_steps, _ = L.shape
    # a.s. bound of one owner-LLR increment: every serving UAV
    # contributes at most ``b_llr`` (finite quantized+erasure alphabet),
    # so ``|Z_t - E[Z_t|F]| <= 2 * (max concurrent servers) * max b_llr``
    # over every run/cycle (a.s.); ``max concurrent servers`` is read
    # from the recorded ``n_served`` and bounded by ``K`` by the capacity
    # constraint ``sum_q x_iq <= 1`` of the relaxation.
    max_b = float(np.max(np.asarray(bridge_out["b_llr"]))) \
        if len(bridge_out["b_llr"]) > 0 else 1e-9
    max_conc = float(np.max(np.asarray(bridge_out["n_served"]))) \
        if bridge_out["n_served"].size > 0 else 1.0
    b_theo = 2.0 * max_conc * max_b
    bq = max(b_theo, 1e-9)

    def tail_rows(which: str):
        rows = []        # (v_avg, empirical_LHS, bound_RHS, H1_only)
        for qq in range(q):
            mask = H[:, qq] == True  # noqa: E712
            if not np.any(mask):
                continue
            l_sub = L[mask, :, qq]
            a_sub = A[mask, :, qq]
            v_sub = V[mask, :, qq]
            m_sub = M[mask, :, qq]
            # residual-decomposition error for the martingale claim
            resid = l_sub - (a_sub + m_sub)
            err_abs = float(np.max(np.abs(resid)))
            # (unconditional) realized deviation tail per time
            for t in range(max_steps):
                mcol = m_sub[:, t]
                vcol = v_sub[:, t]
                for f in (0.05, 0.1, 0.25, 0.5):
                    eta = (float(np.mean(vcol)) ** 0.5) * f
                    emp_lhs = 0.0
                    bound = freedman_tail(eta, float(np.mean(vcol)), bq)
                    if which == "one-sided":
                        emp_lhs = float(np.mean(mcol <= -eta))
                    else:
                        emp_lhs = float(np.mean(np.abs(mcol) >= eta))
                    rows.append((vcol.mean(), emp_lhs, bound))
        return [r for r in rows if r[2] > 0.0]

    # quantization/token correction (advice/003 section 4): the scheduler
    # books the unquantized ``g`` (A_raw) while the deployed belief uses
    # the token atoms (A).  If the relative gap is tiny, the
    # finite-threshold correction of FRIDS-v2 is negligible and the
    # g-only scheduler stays frozen.
    corr_ratio = []
    for qq in range(q):
        a_raw_q = A_raw[:, :, qq]
        a_q = A[:, :, qq]
        keep = a_raw_q > 1e-12
        if np.any(keep):
            corr_ratio.append(float(np.mean(
                np.abs(a_raw_q[keep] - a_q[keep]) / a_raw_q[keep])))
    quant_gap_mean = float(np.mean(corr_ratio)) if corr_ratio else 0.0
    quant_gap_max = float(np.max(corr_ratio)) if corr_ratio else 0.0
    rows = tail_rows("one-sided")
    viols = np.array([r[1] for r in rows], dtype=float)
    probs = np.array([r[2] for r in rows], dtype=float)
    br = relative_error_bound(viols, probs)
    # residual statistics over H1 (run, target) pairs, over all cycles
    m_vals = []
    for qq in range(q):
        run_mask = H[:, qq]
        if not np.any(run_mask):
            continue
        m_vals.append(M[run_mask, :, qq])
    m_vals = np.concatenate(m_vals) if m_vals else np.array([0.0])
    return {
        "decomposition_max_abs_error": float(
            np.max(np.abs(L - (A + M)))),
        "martingale_residual_mean": float(float(np.mean(m_vals))),
        "martingale_residual_sd": float(float(np.std(m_vals))),
        "quantization_gap_mean": quant_gap_mean,
        "quantization_gap_max": quant_gap_max,
        "b_q": float(bq),
        "freedman": br,
        "freedman_n_cases": br["n_cases"],
        "empirical_pM_lower_avg": float(
            viols.mean() if len(viols) else 0.0),
        "empirical_pM_lower_max": float(
            viols.max() if len(viols) else 0.0),
        "target_alpha": target_alpha,
        "target_beta": target_beta,
    }


def stopping_tail_verify(
    bridge_out: dict,
    q: int,
    target_beta: float = 0.05,
) -> dict:
    """Theorem 4.110 direct check: at each (target, cycle) with
    ``A_q(t) >= A_q*`` the empirical H1-survive fraction
    ``Pr_1(T_q > t)`` is compared to the theoretical tail bound.  The
    H0-side is omitted (``beta_q`` is the H0/miss budget on H1)."""
    L = np.asarray(bridge_out["L"])
    A = np.asarray(bridge_out["A"])
    V = np.asarray(bridge_out["V"])
    H = np.asarray(bridge_out["H"])
    T_stop = np.asarray(bridge_out["T"])
    a_thr = np.asarray(bridge_out["a_thr"], dtype=float)
    max_b = float(np.max(np.asarray(bridge_out["b_llr"]))) \
        if len(bridge_out["b_llr"]) > 0 else 1e-9
    max_conc = float(np.max(np.asarray(bridge_out["n_served"]))) \
        if bridge_out["n_served"].size > 0 else 1.0
    bq = max(2.0 * max_conc * max_b, 1e-9)
    n_runs, max_steps, _ = L.shape
    surv_emp = []        # empirical Pr_1(T_q > t)
    surv_bound = []      # theoretical bound
    for qq in range(q):
        mask = H[:, qq] == True  # noqa: E712
        hh = np.arange(n_runs)[mask]
        if len(hh) == 0:
            continue
        la = A[hh, :, qq]
        lv = V[hh, :, qq]
        lt = T_stop[hh, qq]
        for t in range(max_steps):
            a_col = la[:, t]
            v_col = lv[:, t]
            pred = float(np.mean(a_col))
            if pred < a_thr[qq] - 1e-9:
                continue          # premise not met: service not at D_q
            emp = float(np.mean(lt > t))       # still undecided at t
            bnd = stopping_tail_bound(pred, a_thr[qq], float(np.mean(v_col)),
                                      bq, target_beta)
            surv_emp.append(emp)
            surv_bound.append(bnd)
    emp = np.array(surv_emp, dtype=float)
    bnd = np.array(surv_bound, dtype=float)
    ok = bool(np.all(emp <= bnd + 1e-12)) if len(emp) else True
    return {
        "cases": len(emp),
        "empirical_survive_max": float(np.max(emp)) if len(emp) else 0.0,
        "bound_survive_max": float(np.max(bnd)) if len(bnd) else 0.0,
        "violation_fraction": float(
            np.mean(emp > bnd + 1e-12)) if len(emp) else 0.0,
        "empirical_survive_avg": float(
            float(np.mean(emp)) if len(emp) else 0.0),
        "bound_survive_avg": float(float(np.mean(bnd)) if len(bnd) else 0.0),
        "satisfied": bool(ok),
    }


def normalized_service_time_average(
    bridge_out: dict,
    q: int,
    g_mat: np.ndarray | None = None,
    target_beta: float = 0.05,
    eps: float = 0.1,
) -> dict:
    """Theorem B bridge (advice/003 section 5): time-averaged
    demand-normalized service ``min_q (1/T) sum_t r_q(t)`` versus the
    static relaxation optimum ``z* = max_x min_q sum_i x_iq g_iq /
    D_q`` and the mirror-descent regret+distributed-loss bound
    ``eps_T ~ O(sqrt(log Q / T))``.

    ``r_q(t)`` is measured two ways: ``r_real`` (realized delivered
    normalized service, owner-accounting -- what the owner actually
    received) and ``r_pred`` (predictable scheduled service).  The gap
    between the two is the ``eps_loc`` distributed-information loss
    (delivery erasure + local-dual disagreement), the same quantity
    bounded by Theorem 4.109 and measured in Gate F0-G9A.  ``g_mat`` is
    the ``(K, Q)`` reliable-information matrix used to evaluate the
    static ``z*`` with the t=0 deficits ``a_thr`` (the LLR threshold)."""
    r_real = np.asarray(bridge_out["r_real"], dtype=float)   # (R,T,Q)
    r_pred = np.asarray(bridge_out["r_pred"], dtype=float)
    n_served = np.asarray(bridge_out["n_served"])
    L_own = np.asarray(bridge_out["L"], dtype=float)
    H = np.asarray(bridge_out["H"])
    T_stop = np.asarray(bridge_out["T"], dtype=float)
    a_thr = np.asarray(bridge_out["a_thr"], dtype=float)
    n_runs, max_steps, _ = r_real.shape

    # only undecided cycles count: after T_q the recorded service stays 0,
    # so the time average must be taken over t < T_q (the realized service
    # trajectory stops at the stopping rule)
    def avg_undecided(x: np.ndarray, qq: int) -> np.ndarray:
        ts = np.arange(max_steps)[None, :]
        mask = ts < np.asarray(T_stop)[:, qq][:, None]      # (R, T)
        masked = x[:, :, qq] * mask
        return np.sum(masked, axis=1) / np.maximum(np.sum(mask, axis=1),
                                                   1e-12)

    r_real_q = np.array([avg_undecided(r_real, qq) for qq in range(q)])
    r_pred_q = np.array([avg_undecided(r_pred, qq) for qq in range(q)])
    serv_real = [float(np.mean(r_real_q[qq][H[:, qq] == True]))   # noqa: E712
                 if np.any(H[:, qq] == True) else 0.0             # noqa: E712
                 for qq in range(q)]
    serv_pred = [float(np.mean(r_pred_q[qq])) for qq in range(q)]
    min_real = float(np.min(serv_real))
    min_pred = float(np.min(serv_pred))
    eps_loc = float(np.mean([abs(a - b) for a, b in zip(serv_pred,
                                                         serv_real)]))
    # Theorem B: the advice/003 demand-normalized ``r_q(t) = sum_i x_iq
    # g_iq / (D_q(t)+eps)`` has a shrinking denominator near the decision
    # boundary (D -> 0), so the per-cycle value explodes there; the
    # STATIC relaxation statement (Theorem 4.96: the rule tracks the
    # frozen instantaneous relaxation) instead normalizes by the fixed
    # initial deficit, giving the bounded static service
    # ``r_q^st(t) = A-increment / (a_thr_q + eps)`` (schedule-level, no
    # boundary explosion).  The static relaxation optimum
    # ``z* = max_x min_q sum x g / (a_thr_q + eps)`` is the LP value;
    # the mirror-descent rule should reach ``z*`` in per-cycle time
    # average up to ``eps_T = O(sqrt(log Q / T))`` plus the measured
    # distributed-information loss ``eps_loc`` (realized delivered vs
    # scheduled: delivery erasure + local dual disagreement, the same
    # quantity of Theorem 4.109 / Gate F0-G9A).
    A = np.asarray(bridge_out["A"], dtype=float)      # (R, T, Q)
    A_prev = np.concatenate([np.zeros((n_runs, 1, q)), A[:, :-1, :]],
                            axis=1)
    d0 = a_thr + eps
    def avg_static(x: np.ndarray, qq: int) -> np.ndarray:
        ts = np.arange(max_steps)[None, :]
        mask = ts < np.asarray(T_stop)[:, qq][:, None]      # (R, T)
        xq = x[:, :, qq]
        return np.sum(xq * mask, axis=1) / np.maximum(np.sum(mask, axis=1),
                                                      1e-12)
    r_static_pred = (A - A_prev) / d0[None, None, :]
    r_static_q = np.array([avg_static(r_static_pred, qq) for qq in range(q)])
    min_static_pred = float(np.min([float(np.mean(r_static_q[qq]))
                                    for qq in range(q)]))
    z_avg = None
    if g_mat is not None:
        z_avg = float(static_relaxation_optimum(np.asarray(g_mat,
                                                           dtype=float),
                                                d0.astype(float) - eps,
                                                eps=eps))
    eps_T_est = float(z_avg - min_static_pred) if z_avg is not None else None
    # distributed-information loss at the FIXED initial deficit (no
    # boundary explosion): the realized delivered service (i_plus,
    # owner-accounting) vs the scheduled predictable service, both
    # normalized by a_thr + eps; this is delivery erasure + the
    # quantization gap -- the deployable analogue of Theorem 4.109.
    S = np.asarray(bridge_out["S"], dtype=float)            # (R, T, Q)
    S_prev = np.concatenate([np.zeros((n_runs, 1, q)), S[:, :-1, :]],
                            axis=1)
    r_static_real = (S - S_prev) / d0[None, None, :]
    eps_loc_static = float(np.mean(np.abs(r_static_pred - r_static_real)))
    return {
        "z_star_static": z_avg,
        "eps_T_est": eps_T_est,
        "min_q_time_avg_r_static": min_static_pred,
        "per_target_r_static": [float(np.mean(r_static_q[qq]))
                                for qq in range(q)],
        "sqrt_logQ_over_T": float(
            np.sqrt(np.log(max(q, 2)) / max(max_steps, 1))),
        "min_q_time_avg_r_real": min_real,
        "min_q_time_avg_r_pred": min_pred,
        "per_target_r_real": serv_real,
        "per_target_r_pred": serv_pred,
        "eps_loc_mean": eps_loc,
        "eps_loc_static_mean": float(eps_loc_static),
        "target_beta": target_beta,
    }