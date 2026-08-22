"""CA-FRIDS: Joint-Capacity Task-Oriented Dual-Bus FRIDS (advice/008,
P3.2).

The advice/008 upgrade of FRIDS-v2: the action index is no longer the
information-only ``J_v2 = y_i,o * g / (D_i,o + eps)``.  It is the dual
best response of the JOINT sensing--communication relaxation

    max x, z   z
    s.t.       sum_{i,a} x_{iqa} g_{iqa}/(D_q+eps) >= z        (all q)
               sum_{q,a} x_{iqa} <= 1                          (all i)
               sum_{i,q:o(q)=j,a} c_{iqa} x_{iqa} <= 1         (all j)

whose Lagrangian gives the local index

    J^CA_{iqa} = pi_q g_{iqa} - lambda_{o(q)} c_{iqa},

with the task price ``pi_q = y_q/(D_q+eps)`` (owner-anchored) and the
receiver airtime price ``lambda_j`` (dual ascent on the observed
receive-load ratio).  ``g`` is the reliable post-communication
detection information of the deployed action, ``c`` the token airtime
fraction its report consumes at the owner -- sensing and communication
enter the SAME score (not stitched afterwards).

Architecture change (the Dual-Bus): ONE task-price plane, ONE airtime-
price plane, and an evidence plane that sends each evidence token ONLY
to the target owner.  Non-owner UAVs no longer replicate the full
evidence history nor any local belief: every UAV acts on its own
locally-computable ``g``/``c`` and the two broadcast price vectors.  The
stopping decision stays on the owner belief (the system detection rule
is unchanged).

The joint LP needs the idle option: without ``x=0`` (do not report), a
purely additive price is a no-op (Lemma 4.99); with the idle score 0
the price decides report vs silence AND reorders the sensing target.
"""

from __future__ import annotations

import numpy as np

from uav_otfs_isac.difficulty_decomposition import d_kl_binary
from uav_otfs_isac.distributed_audit import (
    quantize_llr,
    quantize_with,
)


def _g_and_c_actions(scenario: dict, owner_of: list,
                     airtime: dict, s_for_g: np.ndarray | None,
                     power_cap: np.ndarray | None):
    """Per-(UAV, target) list of ``(g_eff, c_air, power, i_plus, rel)``
    for every kernel: ``g = I+ * s_{i,o(q)}`` (scheduler-believed
    delivery), ``c = b_tok(action) / (R_{i,o(q)} * T_air)`` (fraction of
    the owner's per-cycle airtime budget one report consumes -- the
    physical token airtime ``b/R``, scaled by the airtime budget)."""
    k = scenario["k"]
    q = scenario["q"]
    rate = np.asarray(airtime["rate"], dtype=float)
    t_air = float(airtime["t_air"])
    s = s_for_g if s_for_g is not None else scenario["u2u_success"]
    acts: dict = {}
    g_max = 0.0
    c_max = 0.0
    for i in range(k):
        for qq in range(q):
            owner = owner_of[qq]
            rel = 1.0 if i == owner else float(s[i, owner])
            rr = float(rate[i, owner])
            row = []
            for act in scenario["by_host"][(i, qq)]:
                if power_cap is not None and float(act["power"]) \
                        > float(power_cap[qq]):
                    continue
                b_tok = float(act.get("bits", airtime.get("b_tok", 0.0)))
                g_eff = float(act["i_plus"]) * rel
                c_air = b_tok / max(rr * max(t_air, 1e-12), 1e-12)
                g_max = max(g_max, g_eff)
                c_max = max(c_max, c_air)
                row_act = (g_eff, c_air, float(act["power"]),
                           float(act["i_plus"]), rel, act)
                row.append(row_act)
            acts[(i, qq)] = row
    return acts, g_max, c_max


def _best_g_matrix(scenario, owner_of, s_for_g, power_cap):
    """``g_mat[i,q]`` = max over actions of the reliable information
    (same quantity the FRIDS-v2 ``g_mat`` uses) -- the sensing-only
    information ceiling used by the feasibility audit and the task
    plane."""
    k, q = scenario["k"], scenario["q"]
    s = s_for_g if s_for_g is not None else scenario["u2u_success"]
    g = np.zeros((k, q))
    for i in range(k):
        for qq in range(q):
            owner = owner_of[qq]
            rel = 1.0 if i == owner else float(s[i, owner])
            best = 0.0
            for act in scenario["by_host"][(i, qq)]:
                if power_cap is not None and float(act["power"]) \
                        > float(power_cap[qq]):
                    continue
                best = max(best, float(act["i_plus"]) * rel)
            g[i, qq] = best
    return g


def _quantize_price(v: np.ndarray, lo: float, hi: float,
                    bits: int) -> np.ndarray:
    """Uniform mid-rise quantization of a broadcast price vector over a
    PRE-REGISTERED range (the price error bound for the action-
    invariance certificate is ``eps = (hi-lo)/2^bits``)."""
    levels = int(2 ** max(0, int(bits)))
    step = (hi - lo) / max(levels, 1)
    x = np.clip(v, lo, hi - 1e-9)
    idx = np.floor((x - lo) / step).astype(int)
    idx = np.clip(idx, 0, levels - 1)
    return lo + (idx + 0.5) * step


def _audit_cycle(act_lists, choices, ideal, perturbed, g_max, c_max,
                 eps_pi, eps_lambda, aud, owner_anchor):
    """Advice/008 section 8: the ideal joint score vs the broadcast
    (quantized) score, the ideal top-1 margin ``m_i`` and the
    action-invariance certificate
    ``P(m_i > 2 (g_max eps_pi + c_max eps_lambda))``."""
    k = len(ideal)
    q = choices["q"]
    for i in range(k):
        ide = ideal[i]
        per = perturbed[i]
        if not ide:
            continue
        qs = sorted(ide, key=ide.get, reverse=True)
        a1 = qs[0]
        if len(qs) < 2:
            continue
        m_i = float(ide[a1] - ide[qs[1]])
        e_i = g_max * eps_pi + c_max * eps_lambda
        aud["margin_total"] += 1.0
        if m_i > 2.0 * e_i:
            aud["margin_ok"] += 1.0
        if max(per, key=per.get) != a1:
            aud["action_change"] += 1.0
        aud["action_total"] += 1.0
    aud["n_cycles"] += 1.0


def simulate_ca_frids(
    scenario: dict,
    bounds: list,
    airtime: dict,
    n_runs: int = 200,
    seed: int = 0,
    max_steps: int = 40,
    alpha: float = 0.05,
    beta: float = 0.05,
    mu: float = 0.5,
    mu_c: float = 0.2,
    eps: float = 0.1,
    quantizer: dict | None = None,
    delivery_matrix: np.ndarray | None = None,
    s_for_g: np.ndarray | None = None,
    pi_bits: int = 10,
    lam_bits: int = 10,
    lam_cap: float = 2.0,
    power_cap: np.ndarray | None = None,
    audit: bool = False,
    raw_counts: bool = False,
) -> dict:
    """MC run of the CA-FRIDS Dual-Bus scheduler (advice/008 P3.2).

    Every cycle:

    - each owner computes its OWN residual deficit ``D_q = [A*_q -
      L_{o(q),q}]_+`` from its own belief (owner-anchored, no local
      replicates) and updates its OWN task price ``y_q`` by
      exponentiated-gradient on the normalized service gap of the
      targets it owns; broadcasts ``pi_q = y_q/(D_q+eps)``;
    - each receiver ``j`` updates ``lambda_j`` by dual ascent on the EMA
      of its committed airtime load; broadcasts it;
    - each UAV computes the LOCAL best response over its own kernels and
      the undecided targets

        (i, q, a)* = argmax [ pi_q g_{iqa} - lambda_{o(q)} c_{iqa} ],

      with the idle option ``score 0`` (the price decides report vs
      silence -- without it the price is a no-op; Lemma 4.99);
    - the evidence plane delivers the token to the TARGET OWNER ONLY
      (physical Bernoulli + overflow thinning under the airtime budget);
      the owner belief is the only process the stopping rule reads.

    ``pi_bits``/``lam_bits`` quantize the two broadcast price planes
    over their pre-registered ranges; ``audit`` records the ideal-vs-
    broadcast action-invariance certificate (``margin_ok`` fraction) --
    the price-quantization/staleness theory of advice/008 section 8.
    """
    k = scenario["k"]
    q = scenario["q"]
    owner_of = scenario["owner_of"]
    delivery = delivery_matrix if delivery_matrix is not None \
        else scenario["u2u_success"]
    acts, g_max, c_max = _g_and_c_actions(scenario, owner_of, airtime,
                                          s_for_g, power_cap)
    g_mat = _best_g_matrix(scenario, owner_of, s_for_g, power_cap)
    tau = np.asarray(airtime["tau"], dtype=float)
    t_air = float(airtime["t_air"])
    pi_hi = 1.0 / max(eps, 1e-9)          # certified pi range (y<=1, D>=0)

    rng = np.random.default_rng(seed)
    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    infeasible_cycles = np.zeros((n_runs, q))
    decided_upper = np.zeros((n_runs, q)) if raw_counts else None
    comm_airtime = np.zeros(n_runs)
    comm_tx = np.zeros(n_runs)
    comm_rx = np.zeros(n_runs)
    comm_max_ratio = np.zeros(n_runs)
    comm_thinned = np.zeros(n_runs)
    comm_feasible = np.zeros(n_runs)
    comm_cycles = np.zeros(n_runs)
    aud = {
        "margin_ok": 0.0, "margin_total": 0.0,
        "action_change": 0.0, "action_total": 0.0, "n_cycles": 0.0,
    }
    a_thr = np.array([float(bounds[qq][0]) for qq in range(q)])
    # t=0 dual cold-start: every receiver seeds lambda from its full-mesh
    # receive-load scarcity forecast (binds immediately in congestion)
    load0 = np.array([float(np.sum(tau[:, j]))
                      for j in range(k)]) / max(t_air, 1e-12)
    load_smooth = load0 * max(t_air, 1e-30)

    for r in range(n_runs):
        H = rng.random(q) < 0.5
        H_all[r] = H
        L = np.zeros((k, q))
        decided = np.zeros(q, dtype=bool)
        y = np.full(q, 1.0 / q)
        lam = np.clip(mu_c * np.maximum(load0 - 1.0, 0.0), 0.0, lam_cap)
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            comm_cycles[r] += 1.0
            # task-price plane: owner-anchored prices, then broadcast
            D = np.maximum(a_thr - np.array(
                [L[owner_of[qq], qq] for qq in range(q)]), 0.0)
            pi = y / (D + eps)
            pi_b = _quantize_price(pi, 0.0, pi_hi, pi_bits)
            lam_b = _quantize_price(lam, 0.0, lam_cap, lam_bits)
            eps_pi = pi_hi / max(int(2 ** pi_bits), 1)
            eps_lam = lam_cap / max(int(2 ** lam_bits), 1)
            # per-UAV local best response with the idle option
            choices = [None] * k
            ideal = [None] * k
            perturbed = [None] * k
            for uav in range(k):
                best = None
                best_score = 0.0
                best_ideal = 0.0
                sc_ide = {}
                sc_per = {}
                for qq in undecided:
                    owner = owner_of[qq]
                    lamq = float(lam[owner])
                    lamq_b = float(lam_b[owner])
                    pib = float(pi_b[qq])
                    for (g_eff, c_air, power, i_plus, rel, act) \
                            in acts[(uav, qq)]:
                        score = pi[qq] * g_eff - lamq * c_air
                        score_b = pib * g_eff - lamq_b * c_air
                        if qq not in sc_ide or score > sc_ide[qq]:
                            sc_ide[qq] = score
                        if qq not in sc_per or score_b > sc_per[qq]:
                            sc_per[qq] = score_b
                        if score_b > best_score:
                            best_score = score_b
                            best_ideal = score
                            best = (qq, g_eff, c_air, i_plus, rel, act)
                choices[uav] = best
                ideal[uav] = sc_ide
                perturbed[uav] = sc_per
            if audit:
                _audit_cycle(acts, {"q": q}, ideal, perturbed, g_max,
                             c_max, eps_pi, eps_lam, aud, owner_of)
            # sensing (the chosen kernel is still sensed exactly once;
            # the joint price decided WHETHER and on WHAT)
            obs_target = np.full(k, -1, dtype=int)
            obs_llr = np.zeros(k)
            obs_iplus = np.zeros(k)
            for uav in range(k):
                c = choices[uav]
                if c is None:
                    continue
                qq, g_eff, c_air, i_plus, rel, act = c
                y_obs = int(rng.choice(len(act["p1"]),
                                       p=act["p1"] if H[qq] else act["p0"]))
                llr_obs = (quantize_with(quantizer, float(act["llr"][y_obs]))
                           if quantizer is not None
                           else quantize_llr(float(act["llr"][y_obs])))
                obs_target[uav] = qq
                obs_llr[uav] = llr_obs
                obs_iplus[uav] = float(act["i_plus"])
                if owner_of[qq] == uav:
                    L[uav, qq] += llr_obs
            # evidence plane: token to the TARGET OWNER ONLY; physical
            # Bernoulli delivery, airtime overload thinning at the owner.
            # The commit load of the owner is the SUM of the airtimes of
            # the cross-UAV tokens it receives (its own observation is
            # local and consumes no receive airtime).  The delivery draw
            # recorded here is THE delivered token (the price-update
            # service S uses this same outcome -- no second draw).
            load_now = np.zeros(k)
            for uav in range(k):
                qq = int(obs_target[uav])
                if qq < 0:
                    continue
                owner = owner_of[qq]
                if uav != owner:
                    load_now[owner] += float(tau[uav, owner])
            thin = np.minimum(1.0, t_air / np.maximum(load_now, 1e-15))
            S = np.zeros(q)
            for uav in range(k):
                qq = int(obs_target[uav])
                if qq < 0:
                    continue
                owner = owner_of[qq]
                if uav == owner:
                    S[qq] += obs_iplus[uav]
                    continue
                # the overflow-thinning loss mass at the owner (advice/013
                # ledger): ``1 - thin`` is exactly the airtime-induced drop
                # probability under the committed load -- the Bernoulli
                # outage is counted separately (it is physical link loss,
                # NOT receiver capacity loss)
                comm_thinned[r] += float(1.0 - thin[owner])
                if rng.random() <= delivery[uav, owner] * thin[owner]:
                    L[owner, qq] += obs_llr[uav]
                    S[qq] += obs_iplus[uav]
                    comm_tx[r] += 1.0
            comm_airtime[r] += float(np.sum(load_now))
            comm_rx[r] += float(np.mean(load_now)) if k else 0.0
            comm_max_ratio[r] = max(comm_max_ratio[r],
                                    float(np.max(load_now) / max(t_air, 1e-15))
                                    if k else 0.0)
            comm_feasible[r] += float(np.max(load_now)
                                      <= t_air + 1e-12) if k else 0.0
            # owner task-price update (advice/008 section 7): each OWNER
            # computes ``pi_q = y_q/(D_q+eps)`` from its OWN belief and
            # its own received service -- no replicated global belief, no
            # global simplex (every undecided target a UAV does NOT own is
            # priced by the broadcast ``pi``).  ``y_q`` is the owner's
            # exponentiated-gradient dual of its own targets' normalized
            # service gap ``rbar - ratio``, normalized on the owner's own
            # undecided simplex (a single-target owner keeps ``y=1`` and
            # its price is set by its residual deficit alone -- the
            # owner-anchored minimal-messaging variant; the F0-G7
            # deficit-normalization weakness is the honest cost the P3.4
            # gate must measure).
            y_new = y.copy()
            ratios = np.array([S[qq] / max(D[qq] + eps, 1e-12)
                               for qq in range(q)])
            for uav in range(k):
                owned = [qq for qq in range(q)
                         if owner_of[qq] == uav and not decided[qq]]
                if not owned:
                    continue
                rbar = float(np.mean(ratios[owned]))
                for qq in owned:
                    y_new[qq] = y[qq] * np.exp(mu * (rbar - ratios[qq]))
                s_own = float(np.sum(y_new[owned]))
                y_new[owned] = y_new[owned] / max(s_own, 1e-12)
            y = y_new
            # receiver airtime price: dual ascent on the EMA load
            load_smooth = 0.8 * load_now + 0.2 * load_smooth
            rho = load_smooth / max(t_air, 1e-30)
            lam = np.clip(lam + mu_c * (rho - 1.0), 0.0, lam_cap)
            # stopping on the owner belief (unchanged system rule)
            for qq in undecided:
                l_own = L[owner_of[qq], qq]
                if l_own >= bounds[qq][0]:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)
                    if H[qq]:
                        declared_h1[r, qq] = 1.0
                    if raw_counts:
                        decided_upper[r, qq] = 1.0
                elif l_own <= bounds[qq][1]:
                    decided[qq] = True
                    delays[r, qq] = float(t + 1)

    e1 = [float(delays[H_all[:, qq], qq].mean()) for qq in range(q)]
    p_fa = [float(declared_h1[~H_all[:, qq], qq].mean()) for qq in range(q)]
    p_md = [float(1.0 - declared_h1[H_all[:, qq], qq].mean())
            for qq in range(q)]
    active = np.maximum(comm_cycles, 1.0)
    out = {
        "worst_target_delay": float(np.max(e1)),
        "e1_delays": e1,
        "p_fa": p_fa,
        "p_md": p_md,
        "infeasible_cycle_fraction": [0.0] * q,
        "comm": {
            "airtime_per_cycle": float(np.mean(comm_airtime / active)),
            "tx_reports_per_uav": float(np.mean(comm_tx / active) / k),
            "rx_load_per_uav": float(np.mean(comm_rx)),
            "max_load_ratio": float(np.mean(comm_max_ratio)),
            "budget_feasible_fraction": float(
                np.mean(comm_feasible / active)),
            "thinned_tokens_per_cycle": float(
                np.mean(comm_thinned / active)),
        },
    }
    if audit:
        out["audit"] = {
            "margin_ok_fraction": float(
                aud["margin_ok"] / max(aud["margin_total"], 1)),
            "margin_samples": float(aud["margin_total"]),
            "action_change_rate": float(
                aud["action_change"] / max(aud["action_total"], 1)),
            "g_max": float(g_max),
            "c_max": float(c_max),
            "eps_pi": float(eps_pi),
            "eps_lambda": float(eps_lam),
            "n_cycles": float(aud["n_cycles"]),
        }
    if raw_counts:
        n_H0, n_H1, n_FA, n_MD = [], [], [], []
        for qq in range(q):
            h0 = int(np.sum(~H_all[:, qq]))
            h1 = int(np.sum(H_all[:, qq]))
            n_H0.append(h0)
            n_H1.append(h1)
            n_FA.append(int(np.sum(decided_upper[~H_all[:, qq], qq])))
            n_MD.append(int(h1 - np.sum(decided_upper[H_all[:, qq], qq])))
        out["raw_counts"] = {"n_H0": n_H0, "n_H1": n_H1,
                             "n_FA": n_FA, "n_MD": n_MD}
    return out