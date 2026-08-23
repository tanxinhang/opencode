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

from uav_otfs_isac.admission import airtime_admit
from uav_otfs_isac.crn_tape import ExogenousTape, draw_atom
from uav_otfs_isac.difficulty_decomposition import d_kl_binary
from uav_otfs_isac.distributed_audit import (
    quantize_llr,
    quantize_with,
)


def _global_simplex(mu: float, ratios: np.ndarray, y: np.ndarray,
                    undecided: list) -> np.ndarray:
    """P3.6 global-simplex mirror update (advice/009 section 7):

        y_q^+ = y_q * exp(-mu * r_q)  /  sum_p y_p * exp(-mu * r_p)

    over the UNDECIDED targets only.  This is the standard entropic
    mirror descent of the max-min relaxation with the exact global dual
    normalizer ``Z = sum_p w_p``: every owner keeps a log-weight
    ``theta_q`` whose softmax over the whole undecided simplex is the
    target price ``y_q``.  The public factor ``exp(mu * rbar)`` cancels in
    the normalization, so NO global ``rbar`` is needed -- the only global
    quantity is the scalar normalizer ``Z`` (a spanning-tree / gossip
    reduction), and ``r_q = S_q/(D_q+eps)`` is owner-local (its own
    received reliable service).  This EXACTLY preserves ``sum_q y_q = 1``,
    which the owner-local per-owner simplex of CA-v0 breaks (advice/009
    section 6: with M active owners the v0 prices sum to ``M``, and a
    single-target owner is frozen at ``y_q = 1``)."""
    # P3.6-R (advice/010 section 11): STANDARD log-sum-exp in log space.
    # ``ell_q = log y_q - mu * r_q`` then ``ell_q <- ell_q - max_p ell_p``
    # before exponentiating -- this is the numerically-safe softmax tail
    # (the previous max-shift of ``r`` could still exponentiate a large
    # positive ``-mu*(r-rmax)`` when ``mu`` is large; log-space centering
    # centres the FULL argument, not just ``r``).  The update is exactly
    # invariant to adding any constant to every ``log w_q``, so the
    # global ``log Z`` (and ``rbar``) never needs to be known -- only the
    # scalar normaliser ``Z`` of the price bus is a networked quantity.
    und = list(undecided)
    if not und:
        return y.copy()
    ell = np.log(np.clip(y[und], 1e-300, None)) - mu * ratios[und]
    ell = ell - float(np.max(ell))
    w = np.exp(ell)
    Z = float(np.sum(w))
    out = y.copy()
    out[und] = w / max(Z, 1e-300)
    return out


def _g_and_c_actions(scenario: dict, owner_of: list,
                     airtime: dict, s_for_g: np.ndarray | None,
                     power_cap: np.ndarray | None):
    """Per-(UAV, target) list of ``(g_eff, c_air, power, i_plus, rel)``
    for every kernel.

    ``g = I+ * s_{i,o(q)}`` (scheduler-believed delivery).  ``c`` is the
    fraction of the OWNER's per-cycle airtime budget one report consumes,
    taken DIRECTLY from the outage-inverted airtime ledger ``tau``
    (advice/009 P0-4): ``c = tau[i, owner] / T_air``.  Because
    ``build_airtime_model`` fills the diagonal with ``0.0``, an owner's
    OWN local evidence carries ``c = 0`` (it is local, it consumes NO
    U2U receive airtime) -- the CA score must not tax owner-local
    sensing with a communication price it never pays.  The physical
    token airtime is therefore exactly the receiver-side ``tau``, and no
    link-rates are recomputed here (``tau`` already is ``b/R``)."""
    k = scenario["k"]
    q = scenario["q"]
    tau = np.asarray(airtime["tau"], dtype=float)
    t_air = float(airtime["t_air"])
    s = s_for_g if s_for_g is not None else scenario["u2u_success"]
    acts: dict = {}
    g_max = 0.0
    c_max = 0.0
    for i in range(k):
        for qq in range(q):
            owner = owner_of[qq]
            rel = 1.0 if i == owner else float(s[i, owner])
            row = []
            for act in scenario["by_host"][(i, qq)]:
                if power_cap is not None and float(act["power"]) \
                        > float(power_cap[qq]):
                    continue
                g_eff = float(act["i_plus"]) * rel
                c_air = float(tau[i, owner]) / max(t_air, 1e-15)
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
    """P3.6-R (advice/010 section 11): uniform MID-TREAD quantization of a
    broadcast price vector over a PRE-REGISTERED range ``[lo, hi]``.

    ``lo`` is EXACTLY one of the ``2^bits`` levels (codeword ``0`` prints
    the true ``lo``), so a zero ``lambda`` (the structural no-congestion
    state) broadcasts as EXACTLY ``0`` instead of a half-bin ``lo+step/2``
    -- the mid-rise quantizer would keep a phantom congestion price at
    ``lambda == 0``, which is a price-plane systematic error.  The
    action-invariance error bound is still ``step = (hi-lo)/2^bits``."""
    levels = int(2 ** max(0, int(bits)))
    if levels <= 1:
        return np.full_like(v, lo, dtype=float)
    step = (hi - lo) / max(levels, 1)
    x = np.clip(v, lo, hi - 1e-9)
    idx = np.floor((x - lo) / step).astype(int)
    idx = np.clip(idx, 0, levels - 1)
    return lo + idx * step


def _density_sorted_offers(
    off: list,
    tie_keys: np.ndarray,
) -> list:
    """P0-B (advice/011 section 6): density-ordered admission list with
    EXCHANGEABLE ties.

    ``off`` entries are ``(uav, qq, c_air, score, ...)``; the primary key
    is ``score / c_air`` descending and the tie key is ``tie_keys[uav]``
    ascending.  The keys live on the UAV INDICES, so a permutation of UAV
    labels permutes the keys exactly the same way -- the admission is
    label-equivariant (no fixed low-index bias), which is the property the
    plain ``sorted`` density break does NOT give.
    """
    return sorted(
        off,
        key=lambda o: (-o[3] / max(o[2], 1e-15), float(tie_keys[o[0]])),
    )


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
    price_mode: str = "owner_local",
    admission_policy: str = "density",
    airtime_price: bool = True,
    audit: bool = False,
    raw_counts: bool = False,
    exog: ExogenousTape | None = None,
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

    # P3.5-A (advice/009 P0-3): per-MC-run RNGs.  Every realization draws
    # ``rng_r = default_rng([seed, r])`` (SeedSequence mixing) so episodes
    # are independent AND individually reproducible, and the Clopper-
    # Pearson QoS certificate's independent-Bernoulli model is not
    # violated by algorithmic cross-run state coupling.
    rng = np.random.default_rng(seed)
    H_all = np.zeros((n_runs, q), dtype=bool)
    delays = np.full((n_runs, q), float(max_steps))
    declared_h1 = np.zeros((n_runs, q))
    infeasible_cycles = np.zeros((n_runs, q))
    decided_upper = np.zeros((n_runs, q)) if raw_counts else None
    comm_airtime = np.zeros(n_runs)
    # P0-C (advice/011 section 6): split the ledger -- every cross-UAV
    # OFFER is an ``attempt`` (its airtime is only committed once it is
    # ADMITTED).  ``comm_admitted_tx`` is the admitted token count, so
    #   attempts = admitted + capacity_dropped    and
    #   admitted = delivered + link_dropped
    # hold separately and the communication-cost interpretation cannot
    # shift between "offered" and "admitted".
    comm_tx_attempts = np.zeros(n_runs)   # offers committed to airtime intent
    comm_admitted_tx = np.zeros(n_runs)   # admitted sends (airtime charged)
    comm_rx_delivered = np.zeros(n_runs)
    comm_rx_link_dropped = np.zeros(n_runs)
    comm_rx_capacity_dropped = np.zeros(n_runs)
    comm_rx_load = np.zeros(n_runs)
    comm_max_ratio = np.zeros(n_runs)
    comm_feasible = np.zeros(n_runs)
    comm_cycles = np.zeros(n_runs)
    # P3.6-R (advice/010 section 4): the CONTROL plane is a real comm
    # cost that the evidence ledger never counted.  Every cycle the
    # Dual-Bus broadcasts ``Q`` task prices ``pi_q`` and ``K`` receiver
    # airtime prices ``lambda_j`` (plus the global-simplex scalar ``Z``
    # normaliser in ``global_simplex`` mode -- here counted as one extra
    # scalar worth of ``pi_bits``).  The evidence ledger above only
    # counts token transmissions, so the control overhead must be added
    # separately to report the SYSTEM total (advice/010: 8*10+16*10 =
    # 240 bits/cycle at Q=8, K=16, b=10).
    comm_control_bits = np.zeros(n_runs)
    aud = {
        "margin_ok": 0.0, "margin_total": 0.0,
        "action_change": 0.0, "action_total": 0.0, "n_cycles": 0.0,
    }
    a_thr = np.array([float(bounds[qq][0]) for qq in range(q)])
    # t=0 dual cold-start: every receiver seeds lambda from its full-mesh
    # receive-load scarcity forecast (binds immediately in congestion)
    load0 = np.array([float(np.sum(tau[:, j]))
                      for j in range(k)]) / max(t_air, 1e-12)

    for r in range(n_runs):
        rng_r = np.random.default_rng([seed, r])
        # CRN (advice/010 P0-2): the shared tape drives target presence so
        # FRIDS-v2 and CA-FRIDS evaluate the SAME H[r, q]; legacy keeps the
        # per-run stream.
        H = (exog.U_H[r] < 0.5) if exog is not None \
            else (rng_r.random(q) < 0.5)
        H_all[r] = H
        L = np.zeros((k, q))
        decided = np.zeros(q, dtype=bool)
        y = np.full(q, 1.0 / q)
        # P4.1a (advice/013 section 3): the ``airtime_price=False`` arm (B0)
        # keeps the airtime price at zero -- it measures the CA
        # architecture + task-price gain WITHOUT the congestion price, so
        # ``Delta_lambda = J_B1 - J_B0`` isolates the price steering.
        lam = (np.clip(mu_c * np.maximum(load0 - 1.0, 0.0), 0.0, lam_cap)
               if airtime_price else np.zeros(k))
        # P3.5-A (advice/009 P0-3): the airtime-load EMA is an EPISODE
        # state; it is re-seeded at the t=0 forecast every run so the
        # (r+1)-th realization does NOT inherit the r-th final load.
        load_smooth = load0 * max(t_air, 1e-30)
        for t in range(max_steps):
            undecided = [qq for qq in range(q) if not decided[qq]]
            if not undecided:
                break
            comm_cycles[r] += 1.0
            # control-plane overhead (advice/010 section 4): Q pi prices,
            # K lambda prices, plus the global-simplex scalar ``Z`` in the
            # global-simplex mode (one scalar at pi resolution)
            comm_control_bits[r] += (q * max(pi_bits, 1)
                                     + k * max(lam_bits, 1)
                                     + (max(pi_bits, 1) if price_mode
                                        == "global_simplex" else 0))
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
                            best = (qq, g_eff, c_air, i_plus, rel, act,
                                    score_b)
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
                qq, g_eff, c_air, i_plus, rel, act, score_b = c
                # CRN: same base observation uniform mapped through the
                # kernel this scheduler chose (legacy uses rng_r.choice).
                y_obs = (
                    draw_atom(
                        act["p1"] if H[qq] else act["p0"],
                        float(exog.U_obs[r, t, uav, qq]),
                    )
                    if exog is not None
                    else int(rng_r.choice(len(act["p1"]),
                                          p=act["p1"] if H[qq] else act["p0"]))
                )
                llr_obs = (quantize_with(quantizer, float(act["llr"][y_obs]))
                           if quantizer is not None
                           else quantize_llr(float(act["llr"][y_obs])))
                obs_target[uav] = qq
                obs_llr[uav] = llr_obs
                obs_iplus[uav] = float(act["i_plus"])
                if owner_of[qq] == uav:
                    L[uav, qq] += llr_obs
            # evidence plane (P3.5-B, advice/009 section 5): the Dual
            # price STEERS, the receiver HARD-ADMITS.  Stage 1: every
            # sensing UAV that cleared the idle gate makes ONE offer to
            # the target owner.  Stage 2: each owner admits a
            # density-ranked subset ``J^+/c`` under the PATHWISE budget
            # ``sum_{(uav) in A_j} c <= 1`` (exchangeable ties via
            # ``rng_r.choice``); ``tau`` diagonal is zero so owner-local
            # sensing is free (P0-4).  Airtime is charged for ADMITTED
            # sends and is fully consumed before the link Bernoulli
            # (transmission airtime/energy are spent on outage too); the
            # un-admitted offers are receiver-capacity drops (P0-5 --
            # lambda is the steering mechanism, hard admission is the
            # pathwise MAC feasibility fuse, and v2/CA now share the same
            # capacity model).
            offers = {j: [] for j in range(k)}
            offered_load = np.zeros(k)
            for uav in range(k):
                c = choices[uav]
                if c is None:
                    continue
                qq, g_eff, c_air, i_plus, rel, act, score_b = c
                if score_b <= 0.0:
                    continue                    # idle gate (Lemma 4.99)
                owner = owner_of[qq]
                if uav != owner:
                    # P3.5-B ledger (advice/009 section 15): a TRANSMISSION
                    # attempt is a CROSS-UAV send that commits airtime;
                    # owner-local sensing is a token-less local event and
                    # is NOT part of the token ledger (it appears in S)
                    comm_tx_attempts[r] += 1.0
                    offers[owner].append((uav, qq, c_air, score_b, c))
                    offered_load[owner] += float(tau[uav, owner])
            admitted = set()
            load_now = np.zeros(k)
            # P4.1 shared-capacity closure (advice/012): admission goes
            # through the SAME ``airtime_admit`` primitive as FRIDS-v2
            # (the constraint ``sum tau/T_air <= 1`` per receiver is
            # identical); only the priority differs -- CA uses the density
            # ``J^+/c`` order, v2 uses the exchangeable neutral order.  The
            # physical order ``offer -> admission -> link outcome`` (an
            # admitted send is charged even on link outage) is shared too.
            offers_by_receiver = [[] for _ in range(k)]
            for owner, off in offers.items():
                for (uav, qq, c_air, score_b, c) in off:
                    offers_by_receiver[owner].append(
                        (uav, float(score_b),
                         float(tau[uav, owner]), qq))
            # P4.1a (advice/013 sections 1-2): admission tie keys are the
            # SHARED (episode, cycle, receiver, source)-indexed policy tape
            # ``exog.U_policy[r, t]`` when the tape is present (every arm
            # reads the same cell, so the neutral and density arms differ
            # ONLY in priority, never in tie randomness), and a dedicated
            # per-cycle policy stream ``[seed, r, t, ...]`` otherwise
            # (``t`` is INSIDE the seed: the keys refresh every cycle, so
            # no persistent source favoritism within an episode).
            tie_keys = (exog.U_policy[r, t]
                        if exog is not None
                        else np.random.default_rng(
                            [seed, r, t, 90213]).random((k, k)))
            admitted_list, dropped = airtime_admit(
                offers_by_receiver, t_air, policy=admission_policy,
                tie_keys=tie_keys,
            )
            comm_rx_capacity_dropped[r] += float(dropped)
            for nb, uav, tau_ij, qq in admitted_list:
                admitted.add((nb, uav))
                load_now[nb] += float(tau[uav, nb])
            S = np.zeros(q)
            for uav in range(k):
                c = choices[uav]
                if c is None:
                    continue
                qq, g_eff, c_air, i_plus, rel, act, score_b = c
                owner = owner_of[qq]
                if uav == owner:
                    S[qq] += obs_iplus[uav]
                    continue
                if score_b <= 0.0 or (owner, uav) not in admitted:
                    continue
                # P0-C: an admitted send is the exact amount whose airtime
                # is charged (``load_now`` above); count it separately from
                # the offers that the admission dropped.
                comm_admitted_tx[r] += 1.0
                # the physical Bernoulli is drawn for every ADMITTED
                # token (its airtime was already spent -- the link outage
                # is NOT a capacity loss, P3.5-B ledger); with the CRN tape
                # the SAME (src, owner) link uniform drives v2 and CA.
                delivered = (
                    exog is not None
                    and float(exog.U_link[r, t, uav, owner])
                    <= delivery[uav, owner]
                ) or (exog is None
                      and rng_r.random() <= delivery[uav, owner])
                if delivered:
                    L[owner, qq] += obs_llr[uav]
                    S[qq] += obs_iplus[uav]
                    comm_rx_delivered[r] += 1.0
                else:
                    comm_rx_link_dropped[r] += 1.0
            comm_airtime[r] += float(np.sum(load_now))
            comm_rx_load[r] += float(np.mean(load_now)) if k else 0.0
            comm_max_ratio[r] = max(comm_max_ratio[r],
                                    float(np.max(load_now) / max(t_air, 1e-15))
                                    if k else 0.0)
            comm_feasible[r] += float(np.max(load_now)
                                      <= t_air + 1e-12) if k else 0.0
            # owner task-price update.  ``price_mode="owner_local"`` (advice/008
            # section 7, the P3.5-A baseline): each OWNER computes
            # ``pi_q = y_q/(D_q+eps)`` from its OWN belief and received
            # service, normalized on the owner's own undecided simplex
            # (single-target owner keeps ``y=1``).  ``price_mode=
            # "global_simplex"`` (P3.6, advice/009 section 7): the same
            # owner-local ``r_q = S_q/(D_q+eps)`` feeds the EXACT global
            # entropic mirror descent ``y_q^+ = y_q e^{-mu r_q} / Z``
            # over the WHOLE undecided simplex -- the global scalar ``Z``
            # is the only networked quantity (spanning-tree/gossip
            # reduction), so ``sum_q y_q = 1`` holds strictly and a
            # single-target owner's price can still change globally.
            ratios = np.array([S[qq] / max(D[qq] + eps, 1e-12)
                               for qq in range(q)])
            und = [qq for qq in range(q) if not decided[qq]]
            if price_mode == "global_simplex":
                y = _global_simplex(mu, ratios, y, und)
            else:
                # owner_local (P3.5-A baseline, byte-identical to the
                # advice/008 owner-anchored variant): per-owner simplex
                y_new = y.copy()
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
            # receiver airtime price: dual ascent on the EMA of the
            # OFFERED (pre-admission) load -- the congestion DEMAND the
            # price must steer, NOT the admitted load.  After a hard
            # admission the admitted load is always <= T_air (the fuse
            # holds pathwise), so the admitted-EMA ``rho`` would be pinned
            # at ~1 forever and lambda would never move (advice/009
            # sections 5, 10-11: lambda steers, admission is the fuse --
            # steering must see the demand that would overload).
            load_smooth = 0.8 * offered_load + 0.2 * load_smooth
            rho = load_smooth / max(t_air, 1e-30)
            # P4.1a: the airtime_price arm keeps the dual frozen at zero,
            # so only the task-price plane steers the best response.
            lam = (np.clip(lam + mu_c * (rho - 1.0), 0.0, lam_cap)
                   if airtime_price else lam)
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
        # pooled-J (advice/010 P0-5): raw per-target H1 delay statistics
        # pooled across ALL runs of this cell (geometry-level pooled
        # worst-target E[T_q | H1] = max_q sum_h1_delay[q]/n_h1[q]).
        "pool": {
            "n_h1": [int(np.sum(H_all[:, qq])) for qq in range(q)],
            "sum_h1_delay": [float(np.sum(delays[H_all[:, qq], qq]))
                             for qq in range(q)],
            "sum2_h1_delay": [float(np.sum(delays[H_all[:, qq], qq] ** 2))
                              for qq in range(q)],
        },
        "comm": {
            "airtime_per_cycle": float(np.mean(comm_airtime / active)),
            # P4.1 split ledger (advice/012 section 5): CANONICAL names.
            # ``offer_attempts_per_uav`` counts every cross-UAV OFFER; the
            # physical TX rate is ``admitted_tx_per_uav`` (the sends whose
            # airtime is charged even on link outage); the identities
            #   offers = admitted + capacity_dropped,
            #   admitted = delivered + link_dropped
            # hold exactly per active cycle.  Historically the offer count
            # was mislabelled ``tx_attempts_per_uav`` -- do NOT claim a
            # transmission-rate reduction from the offer count.
            "offer_attempts_per_uav": float(
                np.mean(comm_tx_attempts / active) / k),
            "admitted_tx_per_uav": float(
                np.mean(comm_admitted_tx / active) / k),
            "delivered_tx_per_uav": float(
                np.mean(comm_rx_delivered / active) / k),
            "link_dropped_tx_per_uav": float(
                np.mean(comm_rx_link_dropped / active) / k),
            "capacity_dropped_tx_per_uav": float(
                np.mean(comm_rx_capacity_dropped / active) / k),
            # deprecated aliases (kept for existing readers):
            "tx_attempts_per_uav": float(
                np.mean(comm_tx_attempts / active) / k),
            "rx_admitted_per_uav": float(
                np.mean(comm_admitted_tx / active) / k),
            "rx_delivered_per_uav": float(
                np.mean(comm_rx_delivered / active) / k),
            "rx_link_dropped_per_uav": float(
                np.mean(comm_rx_link_dropped / active) / k),
            "rx_capacity_dropped_per_uav": float(
                np.mean(comm_rx_capacity_dropped / active) / k),
            # rx_load per ACTIVE cycle (P3.5-B, advice/009 section 15: the
            # accumulator is accumulated per cycle, so it must be divided
            # by the active cycles, not only by n_runs)
            "rx_load_per_uav": float(np.mean(comm_rx_load / active)),
            "max_load_ratio": float(np.mean(comm_max_ratio)),
            "budget_feasible_fraction": float(
                np.mean(comm_feasible / active)),
            # P3.6-R (advice/010 section 4): the control plane cost per
            # cycle -- Q task prices, K airtime prices, plus one global
            # simplex scalar (global_simplex mode).  The evidence ledger
            # alone would under-report the SYSTEM communication cost.
            "control_bits_per_cycle": float(
                np.mean(comm_control_bits / active)),
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