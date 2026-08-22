"""P1 service-delay bridge theory (advice/003, Theorem 4.110/4.111).

The deployed owner LLR is the QUANTIZED token score (advice/004 section
3): ``hat L = Q(LLR)`` is not the exact LLR of the quantized symbol, so
the theory is stated for the deployed score process

    hat L_q(t)  =  tilde A_q(t) + M_q(t),     (M a martingale)

where ``tilde A_q(t) = sum_tau sum_i x_{iq,tau} E_1[hat Z_{iq,tau} |
F_{tau-1}]`` is the cumulative predictable drift of the deployed atom
(``hat Z`` the delivered quantized atom, its H1 conditional mean is
``tilde g_{iq}``), and the quantization correction between the exact
reliable information and the deployed score drift is recorded
separately: ``delta_Q = |g_{iq} - tilde g_{iq}|`` (measured 4-10% at 5
bits -- too small to justify a finite-threshold FRIDS; advice/003
section 4, advice/004 section 3).  The increments are bounded by the
finite (quantized + erasure) alphabet, ``|Z - E[Z|F]| <= b_q``, so
martingale concentration does not need a Gaussian shortcut.

THEOREM-CONSISTENT GATE (advice/004 section 2): a Freedman-type bound
is a statement about the JOINT event ``{M_t <= -eta, V_t <= v}`` for a
DETERMINISTIC ``(eta, v)`` pair,

    P_1( M_t <= -eta,  V_t <= v )  <=  exp[ -eta^2 / (2(v + b_q eta / 3)) ],

not about replacing the path ``V_t`` by its Monte-Carlo mean.  The gate
therefore verifies the joint event on a deterministic ``(eta, v)`` grid,
and, because the deployed object is a STOPPING TIME, additionally the
time-uniform/line-crossing (Freedman maximal) form

    P_1( exists t<=T: M_t <= -eta,  V_t <= v )  <=  exp[ -eta^2 / (2(v + b_q eta / 3)) ].

The recorded processes are fill-forwarded after stopping
(``M_{t wedge T}``, advice/004 P0.5-1) so the audit checks the stopped
martingale, not a process that returns to zero.

Stopping: target q stops on the owner LLR crossing ``A_q*`` (H1) or
``B_q`` (H0); with ``D_q = A_q* - L_q(0)`` and ``beta_q`` the H1
miss-probability budget, if the predictable service reached
``A_q(t) >= D_q + eta`` while the target has still not crossed H1, a
deviation of size >= eta happened, so (integrated over paths with the
H1-miss budget ``beta_q``)

    P_1(T_q > t)  <=  beta_q + E[ exp[ - (A_q(t) - D_q)_+^2 /
        (2(V_q(t) + b_q (A_q(t) - D_q)_+/3)) ] ].

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
    eta_fractions: tuple = (0.5, 1.0, 1.5, 2.0),
    v_fractions: tuple = (0.5, 1.0, 2.0),
) -> dict:
    """Theorem-consistent martingale verification (advice/004 section 2):
    (i) the joint event ``{M_q(t) <= -eta, V_q(t) <= v}`` is verified on
    a deterministic ``(eta, v)`` grid against the Freedman bound (NOT
    by replacing ``V`` with its MC mean -- that is not the object of the
    inequality); (ii) the time-uniform / line-crossing form ``{exists t:
    M_q(t) <= -eta, V_q(t) <= v}`` (the natural object for a stopping
    time) is verified likewise; (iii) ``delta_Q`` the quantization
    correction of the deployed score drift vs exact ``g``.

    ``eta`` grid: ``sqrt(V_ref) * eta_fraction`` with ``V_ref`` the
    deterministic per-target variance UPPER bound (the 100pct quantile
    of the recorded ``V``); ``v`` grid: ``V_ref * v_fraction``.  All
    quantities are in nats (LLR units).  The recorded processes are the
    fill-forwarded stopped processes ``M_{t wedge T}`` (advice/004
    P0.5-1).
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

    # PRE-REGISTERED deterministic per-target variance upper bound
    # (advice/005 section 3): ``V_q(t) <= t * sum_i max_a [s sigma^2 +
    # s(1-s) g~^2]`` -- each UAV serves at most one target per cycle
    # (``sum_q x_iq <= 1``), so the per-cycle conditionally-Bernoulli
    # deliverable variance is bounded by the max over the deployed
    # quantized kernels.  NOT the sample-max of the audit MC (the v grid
    # is fixed before the draws).
    # PRE-REGISTERED per-target variance bound: the bridge recording
    # already carries ``v_up_analytic[q]`` (nats^2 per cycle, computed
    # from the deployed kernels BEFORE the audit draws), so the v grid is
    # ``v_ref = max_steps * v_up_analytic[q]`` -- deterministic and
    # experiment-independent.  Only if the recording lacks it (older
    # snapshot) do we fall back to the path max, flagged as sample-derived.
    v_up_an = np.asarray(bridge_out.get(
        "v_up_analytic", np.zeros(q)), dtype=float)
    if np.max(v_up_an) > 0.0:
        v_up = [float(max_steps) * float(v_up_an[qq]) for qq in range(q)]
        vup_source = "pre-registered analytic (t * sum_i max_a [...] )"
    else:
        v_up = [float(np.max(V[H[:, qq] == True, :, qq]))  # noqa: E712
                if np.any(H[:, qq] == True) else 0.0  # noqa: E712
                for qq in range(q)]
        vup_source = "sample-max (fallback: v_up_analytic absent)"

    def joint_rows(which: str):
        """Deterministic (eta, v) grid joint events over the H1 stopped
        paths: pointwise ``{M(t)<=-eta, V(t)<=v}`` or time-uniform
        ``{exists t: M(t)<=-eta, V(t)<=v}`` vs the Freedman bound."""
        rows = []        # (eta, v, empirical_LHS, bound_RHS)
        for qq in range(q):
            mask = H[:, qq] == True  # noqa: E712
            if not np.any(mask):
                continue
            m_sub = M[mask, :, qq]
            v_sub = V[mask, :, qq]
            v_ref = max(v_up[qq], 1e-9)
            for vf in v_fractions:
                v = v_ref * vf
                for ef in eta_fractions:
                    eta = float(np.sqrt(v)) * ef
                    joint = (m_sub <= -eta) & (v_sub <= v)   # (R, T)
                    if which == "uniform":
                        # line-crossing: the first time the stopped
                        # martingale crosses -eta while V stays <= v
                        emp = float(np.mean(np.any(joint, axis=1)))
                    else:
                        # fixed-time worst case: sup over t of the joint
                        # event probability (each single t is the object
                        # of the pointwise Freedman inequality)
                        emp = float(np.mean(np.any(joint, axis=1)))
                    bound = freedman_tail(eta, v, bq)
                    if which != "uniform":
                        # pointwise: worst over t of the single-time joint
                        emp = float(np.max(np.mean(joint, axis=0)))
                    rows.append((eta, v, emp, bound))
        return [r for r in rows if r[3] > 0.0]

    # quantization/token correction (advice/003 section 4, advice/004
    # section 3): ``delta_Q`` the deployed-score drift vs the exact g.
    # If the relative gap is tiny, the finite-threshold correction of
    # FRIDS-v2 is negligible and the g-only scheduler stays frozen.
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
    rows = joint_rows("pointwise")
    viols = np.array([r[2] for r in rows], dtype=float)
    probs = np.array([r[3] for r in rows], dtype=float)
    br = relative_error_bound(viols, probs)
    rows_u = joint_rows("uniform")
    viols_u = np.array([r[2] for r in rows_u], dtype=float)
    probs_u = np.array([r[3] for r in rows_u], dtype=float)
    br_u = relative_error_bound(viols_u, probs_u)
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
        "freedman_uniform": br_u,
        "freedman_n_cases": br["n_cases"],
        "v_upper_per_target": v_up,
        "v_upper_source": vup_source,
        "eta_grid": [float(np.sqrt(np.max(v_up)) * ef)
                     for ef in eta_fractions],
        "v_grid": [float(np.max(v_up) * vf) for vf in v_fractions],
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
    """Theorem 4.110 SAFE form (advice/005 section 4): the claimed
    object is the DETERMINISTIC joint event

        E_t(eta, v) = { T_q > t,  A_q(t) - D_q >= eta,  V_q(t) <= v }

    with ``D_q = a_thr_q`` (owner 0-deficit), for pre-registered
    deterministic ``(eta, v)`` pairs:

        P_1( E_t(eta,v) ) <= beta_q + exp[ -eta^2 / (2(v + b_q eta / 3)) ],

    and the fully safe decomposition of the survive fraction

        P_1(T_q > t) <= beta_q + exp[ -eta^2 / (2(v + b_q eta/3)) ]
                      + P_1( A_q(t) - D_q < eta ) + P_1( V_q(t) > v ).

    The PATH-INTEGRATED ``E[exp f(A,V)]`` form is NOT a theorem (A,V
    are random historical processes), so it is removed entirely; the
    joint-event form plus the union decomposition are the honest
    structural bounds (advice/005: delete the path-integrated claim).
    The H0-side is ``beta_q`` (the H1 miss budget)."""
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
    n_runs, max_steps, _ = np.asarray(bridge_out["A"]).shape
    # pre-registered deterministic v bound (analytic per-target)
    v_up_an = np.asarray(bridge_out.get(
        "v_up_analytic", np.zeros(q)), dtype=float)
    if np.max(v_up_an) > 0.0:
        v_ref_q = [float(max_steps) * float(v_up_an[qq]) for qq in range(q)]
        vup_source = "analytic"
    else:
        v_ref_q = [float(np.max(V[H[:, qq] == True, :, qq]))  # noqa: E712
                   if np.any(H[:, qq] == True) else 0.0  # noqa: E712
                   for qq in range(q)]
        vup_source = "sample-max"
    joint_emp = []
    joint_bnd = []
    decomp_emp = []
    decomp_bnd = []
    for qq in range(q):
        mask = H[:, qq] == True  # noqa: E712
        hh = np.arange(n_runs)[mask]
        if len(hh) == 0:
            continue
        la = A[hh, :, qq]
        lv = V[hh, :, qq]
        lt = T_stop[hh, qq]
        d_q = float(a_thr[qq])
        v_ref = max(v_ref_q[qq], 1e-9)
        for vf in (0.5, 1.0):
            v = v_ref * vf
            for ef in (0.5, 1.0):
                eta = float(np.sqrt(v)) * ef
                # joint event E_t(eta, v)
                joint_ev = (lt[:, None] > np.arange(max_steps)[None, :]) \
                    & (la - d_q >= eta) & (lv <= v)
                emp_j = 0.0
                # only count (t) pairs where at least one path meets the
                # premise A-D >= eta (otherwise the event is empty)
                act = np.any((la - d_q >= eta) & (lv <= v), axis=0)
                if np.any(act):
                    emp_j = float(np.mean(
                        np.any(joint_ev[:, act], axis=0)))
                bnd_j = float(min(target_beta + freedman_tail(
                    eta, v, bq), 1.0))
                joint_emp.append(emp_j)
                joint_bnd.append(bnd_j)
                # survive decomposition at the same (eta, v)
                for t in range(max_steps):
                    emp_s = float(np.mean(lt > t))
                    pA = float(np.mean(la[:, t] - d_q < eta))
                    pV = float(np.mean(lv[:, t] > v))
                    bnd_s = float(min(
                        target_beta + freedman_tail(eta, v, bq) + pA + pV,
                        1.0))
                    decomp_emp.append(emp_s)
                    decomp_bnd.append(bnd_s)
    j_emp = np.array(joint_emp, dtype=float)
    j_bnd = np.array(joint_bnd, dtype=float)
    d_emp = np.array(decomp_emp, dtype=float)
    d_bnd = np.array(decomp_bnd, dtype=float)
    j_ok = bool(np.all(j_emp <= j_bnd + 1e-12)) if len(j_emp) else True
    d_ok = bool(np.all(d_emp <= d_bnd + 1e-12)) if len(d_emp) else True
    return {
        "cases": len(j_emp),
        "joint_event_violation_fraction": float(
            np.mean(j_emp > j_bnd + 1e-12)) if len(j_emp) else 0.0,
        "decomposition_violation_fraction": float(
            np.mean(d_emp > d_bnd + 1e-12)) if len(d_emp) else 0.0,
        "joint_event_satisfied": bool(j_ok),
        "decomposition_satisfied": bool(d_ok),
        "joint_emp_max": float(np.max(j_emp)) if len(j_emp) else 0.0,
        "joint_bnd_max": float(np.max(j_bnd)) if len(j_bnd) else 0.0,
        "decomp_emp_max": float(np.max(d_emp)) if len(d_emp) else 0.0,
        "decomp_bnd_max": float(np.max(d_bnd)) if len(d_bnd) else 0.0,
        "v_upper_source": vup_source,
        "satisfied": bool(j_ok and d_ok),
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


def run_static_mirror_descent(
    g: np.ndarray,
    d0: np.ndarray,
    horizon: int,
    mu: float = 0.5,
    eps: float = 0.1,
    seed: int = 0,
) -> dict:
    """Static shadow experiment (advice/004 P0.5-4): the frozen-relaxation
    FRIDS mirror descent with NO stopping and NO erasure randomness --
    targets never decide, ``D_q(t) = D_q(0)`` fixed, ``g`` fixed, every
    cycle every UAV serves ``argmax_q y^{(i)}_q g_{iq}/(D_q+eps)`` and
    the per-UAV price follows the exponentiated gradient on the
    normalized service gap.  This isolates the pure convergence of the
    dual rule to the static optimum ``z* = max_x min_q sum_i x_iq g_iq /
    (D_q + eps)`` (the object of Theorem 4.111's ``eps_T ~ O(sqrt(logQ
    / T))`` claim), WITHOUT the stopping / active-set-change confound of
    the deployed system (the advice/004 section 5 criticism of comparing
    against the LP of a different system).

    Deterministic (no RNG), so a single run per horizon is exact.
    """
    k, q = np.asarray(g, dtype=float).shape
    g = np.asarray(g, dtype=float).clip(min=0.0)
    d = np.asarray(d0, dtype=float).clip(min=0.0) + eps
    # THEOREM-consistent object (advice/004 P0.5-4): the static relaxation
    # of Theorem 4.95/4.111 has ONE common shadow price ``y`` on the
    # simplex, not K per-UAV local duals.  The per-UAV local-dual herd
    # (every UAV chases the same best target against its own price) is
    # EXACTLY the distributed-information loss ``eps_loc`` measured by
    # ``local_vs_common_gap`` and does NOT close here; the common-price
    # form is the object whose ``gap(T) = z* - min_q bar r_q(T)`` decays
    # as O(sqrt(log Q / T)).
    y = np.full(q, 1.0 / q)                 # single common price
    z_acc = np.zeros(q)                     # per-target service accum.
    for t in range(1, horizon + 1):
        # exponentiated-gradient (exp-gradient) step on the normalized
        # service gap, tuned as mu_t = mu / sqrt(t) -- the O(sqrt(logQ/T))
        # regret rate of Theorem 4.96 requires the DECAYING step; a fixed
        # step does not close the gap in time average.
        mu_t = float(mu) / float(np.sqrt(t))
        # deterministic best-response of every UAV against the common
        # price (ties: first target)
        serve = np.zeros((k, q))
        for i in range(k):
            scores = g[i] * y / d
            best = int(np.argmax(scores))
            if scores[best] > 0.0:
                serve[i, best] = 1.0
        # per-target normalized service from the served UAVs
        r_q = np.zeros(q)
        for qq in range(q):
            r_q[qq] = float(np.sum(g[:, qq] * serve[:, qq]) / d[qq])
        # mirror descent on the common normalized service gap
        rbar = float(np.mean(r_q))
        e = rbar - r_q
        num = y * np.exp(mu_t * e)
        y = num / max(np.sum(num), 1e-12)
        z_acc += r_q
    z_star = static_relaxation_optimum(g, np.asarray(d0, dtype=float),
                                       eps=eps)
    # time-averaged per-target service, min over targets
    serv_pred = z_acc / max(horizon, 1)
    return {
        "z_star": float(z_star),
        "horizon": int(horizon),
        "min_q_time_avg_r": float(np.min(serv_pred)),
        "per_target_r": [float(x) for x in serv_pred],
        "gap": float(z_star - float(np.min(serv_pred))),
        "sqrt_logQ_over_T": float(np.sqrt(np.log(max(q, 2)) / horizon)),
    }


def static_md_convergence(
    g: np.ndarray,
    d0: np.ndarray,
    horizons: tuple = (20, 40, 80, 160, 320),
    mu: float = 0.5,
    eps: float = 0.1,
) -> dict:
    """Run the static mirror descent on a sweep of horizons and report
    the rate-consistency constant (advice/005 section 5)

        C_emp(T) = gap(T) / sqrt(log Q / T),
        gap(T)   = z* - min_q r_q(T).

    The formal theorem remains the mirror-descent regret
    ``gap(T) <= C sqrt(logQ/T)`` (Theorem 4.111); the empirical ``slope``
    is NOT a proof, so the gate checks ``sup_T C_emp(T) < C_max`` (a
    bounded empirical constant), and the log-log slope is reported only
    as a diagnostic (a well-conditioned instance can reach ``z*``\n    immediately with gap ~ 0 -- strictly stronger than the rate)."""
    rows = {}
    for T in horizons:
        rows[str(T)] = run_static_mirror_descent(g, d0, int(T), mu=mu, eps=eps)
    gaps = np.array([rows[str(T)]["gap"] for T in horizons], dtype=float)
    Ts = np.array([int(T) for T in horizons], dtype=float)
    logq_over_T = np.array([rows[str(T)]["sqrt_logQ_over_T"] for T in horizons],
                           dtype=float)
    c_emp = np.divide(gaps, logq_over_T, out=np.zeros_like(gaps),
                      where=logq_over_T > 0.0)
    # a well-conditioned instance can reach z* immediately (gap ~ 0); the
    # log-log slope is then undefined (and is a STRONGER result), so the
    # slope is measured only over strictly-positive gaps (diagnostic)
    pos = gaps > 1e-6
    slope = None
    if pos.sum() >= 2:
        slope = float(np.polyfit(np.log(Ts[pos]), np.log(gaps[pos]), 1)[0])
    return {
        "rows": rows,
        "gaps": [float(x) for x in gaps],
        "horizons": [int(T) for T in horizons],
        "loglog_slope_diagnostic": slope,
        "sqrt_logQ_over_T": [float(x) for x in logq_over_T],
        "C_emp": [float(x) for x in c_emp],
        "C_emp_max": float(np.max(c_emp)) if len(c_emp) else 0.0,
        "gap_max": float(np.max(gaps)) if len(gaps) else 0.0,
        "converged_immediately": bool(np.max(gaps) <= 1e-6),
        "rate_consistent": bool(np.max(c_emp) <= 1.0),
        "rate_budget_sqrt_logQ": 1.0,
    }


def local_vs_common_gap(
    bridge_local: dict,
    bridge_common: dict,
    q: int,
    eps: float = 0.1,
) -> dict:
    """Advice/004 P0.5-5: ``eps_loc`` redefined as the local-vs-common
    price effect on the time-averaged static normalized service, under
    COMMON RANDOM NUMBERS (both bridge recordings come from the SAME
    scenario seed; common-price uses ``price_mode=\"common\"`` with
    ``y = mean_i y^{(i)}``, the F0-G9A offline oracle).  The two runs
    share the observations, so the gap isolates the price disagreement
    cost, not the delivery randomness."""
    def static_r(b):
        a_thr = np.asarray(b["a_thr"], dtype=float)
        A = np.asarray(b["A"], dtype=float)
        S = np.asarray(b["S"], dtype=float)
        H = np.asarray(b["H"])
        T_stop = np.asarray(b["T"], dtype=float)
        n_runs, max_steps, _ = A.shape
        d0 = a_thr + eps
        A_prev = np.concatenate([np.zeros((n_runs, 1, q)), A[:, :-1, :]],
                                axis=1)
        r_st = (A - A_prev) / d0[None, None, :]
        out = []
        for qq in range(q):
            ts = np.arange(max_steps)[None, :]
            mask = ts < T_stop[:, qq][:, None]
            xq = r_st[:, :, qq] * mask
            avg = np.sum(xq, axis=1) / np.maximum(np.sum(mask, axis=1), 1e-12)
            out.append(float(np.mean(avg)))
        return out

    r_l = np.array(static_r(bridge_local))
    r_c = np.array(static_r(bridge_common))
    # the DELAY-relevant loss is the one on the BOTTLENECK (binding)
    # target, not the largest mid-target swing: F0-G9A showed the local
    # dual disagreement costs delay only through the weakest target's
    # service, so the honest eps_loc_dual is the gap at the min-service
    # target (the one that sets the worst-target delay).
    min_l = float(np.min(r_l))
    min_c = float(np.min(r_c))
    return {
        "per_target_local": [float(x) for x in r_l],
        "per_target_common": [float(x) for x in r_c],
        "eps_loc_dual": float(np.max(np.abs(r_l - r_c))),
        "eps_loc_dual_mean": float(np.mean(np.abs(r_l - r_c))),
        "eps_loc_bottleneck": float(abs(min_l - min_c)),
        "min_service_local": min_l,
        "min_service_common": min_c,
    }