"""Gate F0-G8A: evidence-dependence audit (advice/013 section 5-8).

The FRIDS scheduler aggregates each UAV's singleton reliable information
`g_iq = max_a I+^post * s_{i, owner_q}` into the target service/deficit
accounting, implicitly assuming the observations of different UAVs are
independent.  With a common scatterer / clutter component the JOINT
detection information of a UAV set `S`,

    G_q(S) = D_KL( P_{1,q}^{Y_S} || P_{0,q}^{Y_S} ),

is strictly less than the singleton sum.  The redundancy ratio

    R_q(S) = 1 - G_q(S) / sum_{i in S} g_iq

is the fraction of the singleton aggregation that is double-counted.

The audit uses the Gaussian common-factor evidence model (the same
covariance under H0/H1),

    Y_i = delta_i * H1 + sqrt(rho_s) * C + sqrt(1 - rho_s) * N_i,

with `C, N_i` i.i.d. standard normal, `delta_i = sqrt(2 g_iq)` backed
out of the reliable information.  The marginal KL is `g_i = delta_i^2/2`
and the joint covariance is `Sigma_S = rho_s 11' + (1 - rho_s) I`, so by
Sherman-Morrison the joint KL has the closed form

    G_q(S) = (1 / (2(1 - rho_s))) * ( sum delta^2
              - rho_s * (sum delta)^2 / (1 + (|S|-1) rho_s) ),
    R_q(S) = (rho_s / (1 - rho_s)) * ( a / (1 + (|S|-1) rho_s) - 1 ),
    a = (sum delta)^2 / sum delta^2   (alignment, in [1, |S|]).

The KL-chain-rule conditional marginal (the G8-B quantity)

    Delta G_{i|S,q} = G_q(S union {i}) - G_q(S) >= 0

is how much UAV `i` actually adds given the existing coalition.  The
delay consequence: with the full coalition serving target `q` each
cycle, the joint-correct sequential drift is `G_q(S)`, so the singleton-
optimistic delay is understated by a factor `sum g / G = 1/(1 - R)`;
a Gaussian sequential test confirms it numerically.

The audit does NOT change FRIDS (advice/013 section 8): it reports
`R_q(S)`, the conditional marginals and the delay consequence over a
controlled `rho_s` grid.  Life gate: if `R_q < 5%` everywhere and the
delay consequence is negligible, close the direction; if `R_q >
10-20%` with a clear delay penalty, start the conditional-information
FRIDS (G8-B).
"""

from __future__ import annotations

import numpy as np

from uav_otfs_isac.frids import g_reliable


def singleton_kl(delta: float) -> float:
    """Marginal KL of one Gaussian observation: ``g = delta^2 / 2``."""
    return float(delta) ** 2 / 2.0


def alignment(delta: np.ndarray) -> float:
    """``a = (sum delta)^2 / sum delta^2 in [1, |S|]`` -- how aligned the
    UAVs' information is (1 = concentrated on one UAV, |S| = all equal)."""
    d = np.asarray(delta, dtype=float)
    s2 = float(np.sum(d * d))
    if s2 <= 0.0:
        return 1.0
    return float(np.sum(d) ** 2 / s2)


def joint_kl(delta: np.ndarray, rho_s: float) -> float:
    """Joint KL of the full set with correlation ``rho_s`` (closed form)."""
    d = np.asarray(delta, dtype=float)
    k = len(d)
    s2 = float(np.sum(d * d))
    s1 = float(np.sum(d))
    rho = float(rho_s)
    if k == 0 or s2 <= 0.0:
        return 0.0
    if abs(rho) < 1e-12:
        return s2 / 2.0
    denom = 1.0 + (k - 1.0) * rho
    return (s2 - rho * s1 * s1 / denom) / (2.0 * (1.0 - rho))


def joint_kl_subset(delta: np.ndarray, subset, rho_s: float) -> float:
    """Joint KL of a target subset (index list) with correlation."""
    d = np.asarray(delta, dtype=float)
    idx = [int(i) for i in subset]
    return joint_kl(d[idx], rho_s)


def redundancy(delta: np.ndarray, subset, rho_s: float) -> float:
    """``R_q(S) = 1 - G_q(S) / sum_{i in S} g_i``."""
    idx = [int(i) for i in subset]
    d = np.asarray(delta, dtype=float)[idx]
    g_sum = float(np.sum(d * d) / 2.0)
    if g_sum <= 0.0:
        return 0.0
    return 1.0 - joint_kl(d, rho_s) / g_sum


def conditional_marginal(delta: np.ndarray, subset, i: int, rho_s: float) -> float:
    """``Delta G_{i|S,q} = G(S union {i}) - G(S)`` (KL chain rule)."""
    idx = [int(x) for x in subset]
    if int(i) in idx:
        return 0.0
    g_without = joint_kl_subset(delta, idx, rho_s)
    g_with = joint_kl_subset(delta, idx + [int(i)], rho_s)
    return max(0.0, g_with - g_without)


def sample_joint_kl(delta: np.ndarray, rho_s: float, n: int = 200000,
                    seed: int = 0) -> float:
    """Monte-Carlo estimate of ``G_q(S)`` for the full set (verification):
    ``E_{P1}[ log P1(Y)/P0(Y) ]`` over draws of the common-factor model."""
    d = np.asarray(delta, dtype=float)
    k = len(d)
    rho = float(rho_s)
    rng = np.random.default_rng(seed)
    cov = np.full((k, k), rho)
    np.fill_diagonal(cov, 1.0)
    half = 0.0
    for _ in range(n):
        c = rng.standard_normal()
        n_i = rng.standard_normal(k)
        y = d + np.sqrt(rho) * c + np.sqrt(1.0 - rho) * n_i
        half += float(np.dot(d, np.linalg.solve(cov, y)))
    return half / n - float(np.dot(d, np.linalg.solve(cov, d))) / 2.0


def build_delta_from_scenario(scenario: dict, owner_of: list) -> np.ndarray:
    """``delta_{iq} = sqrt(2 * g_reliable(scenario, i, q))``: the exact
    reliable-information profile that FRIDS aggregates, converted to the
    Gaussian-common-factor evidence scale."""
    k, q = scenario["k"], scenario["q"]
    delta = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            g = g_reliable(scenario, i, qq, owner_of)
            delta[i, qq] = float(np.sqrt(2.0 * max(g, 0.0)))
    return delta


def sequential_correlation_check(delta: np.ndarray, rho_s: float,
                                 alpha: float = 0.05, beta: float = 0.05,
                                 n_runs: int = 2000, max_steps: int = 200,
                                 seed: int = 0) -> dict:
    """Gaussian sequential test of the full coalition on the correlated
    evidence stream, using the JOINT (correct) likelihood ratio and the
    Wald two-threshold rule.  Reports the realized H1 delay, the errors,
    and the delay ratio against the independent baseline (rho = 0) --
    the sequential confirmation that correlated evidence slows the
    detection by ~ ``1/(1-R)``."""
    d = np.asarray(delta, dtype=float)
    k = len(d)
    rng = np.random.default_rng(seed)
    a_thr = float(np.log((1.0 - beta) / alpha))
    b_thr = float(np.log(beta / (1.0 - alpha)))

    def run(rho):
        cov = np.full((k, k), rho)
        np.fill_diagonal(cov, 1.0)
        cinv = np.linalg.inv(cov)
        g = float(np.dot(d, np.dot(cinv, d))) / 2.0
        delays = np.zeros(n_runs)
        h1 = np.zeros(n_runs, dtype=bool)
        declared_h1 = np.zeros(n_runs, dtype=bool)
        for r in range(n_runs):
            H = rng.random() < 0.5
            h1[r] = H
            L = 0.0
            T = float(max_steps)
            stop_sign = 0.0
            for t in range(max_steps):
                c = rng.standard_normal()
                n_i = rng.standard_normal(k)
                if H:
                    y = d + np.sqrt(rho) * c + np.sqrt(1.0 - rho) * n_i
                else:
                    y = np.sqrt(rho) * c + np.sqrt(1.0 - rho) * n_i
                llr = float(np.dot(d, np.dot(cinv, y))) - g
                L += llr
                if L >= a_thr:
                    T = float(t + 1)
                    stop_sign = 1.0
                    break
                if L <= b_thr:
                    T = float(t + 1)
                    stop_sign = -1.0
                    break
            delays[r] = T
            declared_h1[r] = bool(stop_sign >= 0.0)
        e1 = float(delays[h1].mean())
        p_fa = float(np.mean((~h1) & declared_h1))
        p_md = float(np.mean(h1 & (~declared_h1)))
        return e1, g, delays, h1, p_fa, p_md

    e1_0, g0, del0, h0, fa0, md0 = run(0.0)
    e1_r, gr, del_r, hr, far, mdr = run(rho_s)
    return {
        "rho_s": float(rho_s),
        "independent_drift": float(g0),
        "correlated_drift": float(gr),
        "E1_T_independent": float(e1_0),
        "E1_T_correlated": float(e1_r),
        "delay_ratio_measured": float(e1_r / max(e1_0, 1e-9)),
        "p_fa_independent": float(fa0),
        "p_md_independent": float(md0),
        "p_fa_correlated": float(far),
        "p_md_correlated": float(mdr),
    }