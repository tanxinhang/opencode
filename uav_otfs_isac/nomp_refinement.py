"""NOMP-inspired online max-min refinement for joint power-bit allocation.

The greedy phase follows the winner-take-all marginal rule with an optional
mandatory per-target minimum cover, so every target keeps at least one active
communication/sensing link.  A discrete Newton-style refinement then searches
single power/bit exchanges, within-target atom merges, and redundant-atom
transfers.  A move is accepted only when it improves the lexicographic max-min
vector, so the worst target value never decreases and the loop terminates at a
finite local optimum or at a hard round cap.
"""

from __future__ import annotations

from functools import lru_cache
from heapq import nlargest

import numpy as np

from .power_split_theory import (
    power_gain_coefficient,
    proportional_target_pd,
)
from .robust_joint_power_bit import per_report_communication_target_pd


def qos_scores(values, floors, weights=None):
    """Normalized QoS slack ``w * (v - l) / l`` per target."""
    values = np.asarray(values, dtype=float)
    floors = np.asarray(floors, dtype=float)
    if weights is None:
        weights = np.ones_like(floors)
    weights = np.asarray(weights, dtype=float)
    return weights * (values - floors) / np.maximum(floors, 1e-12)


def _single_target_deflection(
    target,
    powers_row,
    bits_row,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
) -> float:
    """Deflection proxy of one target (the q-th row of a candidate)."""
    _, deltas, flips, successes = _parse_target(
        target, flip_probability, success_probability
    )
    total = 0.0
    for r in range(deltas.size):
        if bits_row[r] > 0:
            total += float(powers_row[r]) * float(
                power_gain_coefficient(
                    float(deltas[r]),
                    int(bits_row[r]),
                    float(flips[r]),
                    float(successes[r]),
                )
            )
    return total


def deflection_proxy(
    scenario,
    powers,
    bits,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
):
    """Cheap per-target deflection proxy for ranking refinement candidates."""
    values = []
    for q, target in enumerate(scenario):
        values.append(_single_target_deflection(
            target,
            powers[q],
            bits[q],
            flip_probability,
            success_probability,
        ))
    return values


def leximin_improves(
    old_values,
    new_values,
    *,
    floors=None,
    weights=None,
) -> bool:
    """True when the sorted target vector improves lexicographically."""
    if floors is not None:
        old_values = qos_scores(old_values, floors, weights)
        new_values = qos_scores(new_values, floors, weights)
    a = np.sort(np.asarray(old_values, dtype=float))
    b = np.sort(np.asarray(new_values, dtype=float))
    for x, y in zip(a, b):
        if not np.isclose(x, y, atol=1e-12, rtol=0.0):
            return y > x
    return False


def _parse_target(target, flip_probability=0.0, success_probability=1.0):
    """Return (owner, deltas, flips, successes) for either scenario format."""
    if isinstance(target, tuple) and len(target) == 4:
        return (
            float(target[0]),
            np.asarray(target[1], dtype=float),
            np.asarray(target[2], dtype=float),
            np.asarray(target[3], dtype=float),
        )
    row = np.asarray(target, dtype=float)
    owner = float(row[0])
    deltas = row[1:]
    flips = np.full(deltas.size, float(flip_probability))
    successes = np.full(deltas.size, float(success_probability))
    return owner, deltas, flips, successes


def parse_target(target, flip_probability=0.0, success_probability=1.0):
    """Public accessor for (owner, deltas, flips, successes)."""
    return _parse_target(
        target, flip_probability, success_probability
    )


def _report_count(target):
    return int(_parse_target(target)[1].size)


def initial_min_cover(
    scenario,
    budget,
    *,
    max_bits: int = 2,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    grid: int = 16,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng_seed: int = 0,
):
    """Activate one best report per target when the budget allows it."""
    reports = _report_count(scenario[0])
    powers = [np.zeros(reports, dtype=int) for _ in scenario]
    bits = [np.zeros(reports, dtype=int) for _ in scenario]
    used = 0
    if 2 * len(scenario) > budget:
        return powers, bits, used
    for q, target in enumerate(scenario):
        zero_powers = np.zeros(reports, dtype=float)
        zero_bits = np.zeros(reports, dtype=int)
        baseline = _target_pd(
            target,
            zero_powers,
            zero_bits,
            grid,
            flip_probability,
            success_probability,
            max_exact_reports,
            samples,
            rng_seed,
        )
        best_value = baseline
        best_winner = None
        for r in range(reports):
            candidate_powers = zero_powers.copy()
            candidate_bits = zero_bits.copy()
            candidate_powers[r] = 1
            candidate_bits[r] = 1
            candidate = _target_pd(
                target,
                candidate_powers,
                candidate_bits,
                grid,
                flip_probability,
                success_probability,
                max_exact_reports,
                samples,
                rng_seed,
            )
            if candidate > best_value:
                best_value = candidate
                best_winner = r
        if best_winner is not None and best_value > baseline + 1e-12:
            winner = best_winner
            powers[q][winner] = 1
            bits[q][winner] = 1
            used += 2
    return powers, bits, used


def _uncoverable_targets(
    scenario,
    *,
    grid: int,
    flip_probability: float,
    success_probability: float,
    max_exact_reports: int,
    samples: int,
    rng_seed: int,
) -> set[int]:
    """Targets no single activation can improve (no usable evidence).

    A target whose every report row fails to raise P_D above the all-zero
    baseline by the numeric tolerance cannot be improved by any allocation
    reachable from it: every activation path starts from a single
    ``(power, bit) = (1, 1)`` activation, and under the set-monotonicity of
    the fused P_D any superset of a non-improving activation stays at the
    baseline.  Every move that touches such a target therefore keeps its
    value at the baseline while consuming units other targets could use, so
    no lexicographic improvement ever accepts it.  Skipping these targets
    in candidate generation is exact up to the same numeric tolerance that
    ``leximin_improves`` uses, and mirrors the sensing principle that
    resources are only spent where detection can actually improve.
    """
    reports = _report_count(scenario[0])
    uncovered: set[int] = set()
    for q, target in enumerate(scenario):
        zero_powers = np.zeros(reports, dtype=float)
        zero_bits = np.zeros(reports, dtype=int)
        baseline = _target_pd(
            target, zero_powers, zero_bits, grid,
            flip_probability, success_probability,
            max_exact_reports, samples, rng_seed,
        )
        best_value = baseline
        for r in range(reports):
            candidate_powers = zero_powers.copy()
            candidate_bits = zero_bits.copy()
            candidate_powers[r] = 1
            candidate_bits[r] = 1
            candidate = _target_pd(
                target, candidate_powers, candidate_bits, grid,
                flip_probability, success_probability,
                max_exact_reports, samples, rng_seed,
            )
            if candidate > best_value:
                best_value = candidate
        if best_value <= baseline + 1e-12:
            uncovered.add(q)
    return uncovered


def target_scores(
    scenario,
    powers,
    bits,
    grid: int = 16,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng_seed: int = 0,
):
    """Per-target P_D under the current allocation."""
    return [
        float(_target_pd(
            target,
            powers[q],
            bits[q],
            grid,
            flip_probability,
            success_probability,
            max_exact_reports,
            samples,
            rng_seed,
        ))
        for q, target in enumerate(scenario)
    ]


def _target_score_at(
    scenario,
    q: int,
    powers,
    bits,
    grid: int = 16,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng_seed: int = 0,
):
    """P_D for a single target under the current allocation."""
    return float(_target_pd(
        scenario[q],
        powers[q],
        bits[q],
        grid,
        flip_probability,
        success_probability,
        max_exact_reports,
        samples,
        rng_seed,
    ))


def _target_pd(
    target,
    powers,
    bits,
    grid,
    flip_probability,
    success_probability,
    max_exact_reports=8,
    samples=2048,
    rng_seed=0,
):
    owner_delta, deltas, flips, successes = _parse_target(
        target, flip_probability, success_probability
    )
    if np.all(flips == 0.0) and np.all(successes == 1.0):
        return proportional_target_pd(owner_delta, deltas, powers, bits, grid)
    return _communication_target_pd_cached(
        float(owner_delta),
        tuple(float(d) for d in deltas),
        tuple(int(p) for p in np.asarray(powers, dtype=int).ravel()),
        tuple(int(b) for b in np.asarray(bits, dtype=int).ravel()),
        tuple(float(f) for f in flips),
        tuple(float(s) for s in successes),
        int(grid),
        int(max_exact_reports),
        int(samples),
        int(rng_seed),
    )


@lru_cache(maxsize=1 << 18)
def _communication_target_pd_cached(
    owner_delta,
    deltas,
    powers,
    bits,
    flips,
    successes,
    grid,
    max_exact_reports,
    samples,
    rng_seed,
):
    return per_report_communication_target_pd(
        owner_delta,
        np.asarray(deltas, dtype=float),
        np.asarray(powers, dtype=float),
        np.asarray(bits, dtype=int),
        np.asarray(flips, dtype=float),
        np.asarray(successes, dtype=float),
        grid,
        max_exact_reports=max_exact_reports,
        samples=samples,
        rng=np.random.default_rng(rng_seed),
    )


def _active_reports(bits):
    return [r for r in range(len(bits)) if bits[r] > 0]


def _rows_equal(row_a, row_b):
    a = np.asarray(row_a, dtype=int)
    b = np.asarray(row_b, dtype=int)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _winner_index(
    target,
    bits,
    powers,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    grid: int = 16,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng_seed: int = 0,
):
    active = _active_reports(bits)
    if not active:
        return None
    owner, deltas, flips, successes = _parse_target(
        target, flip_probability, success_probability
    )
    if not (np.all(flips == 0.0) and np.all(successes == 1.0)):
        base = _target_pd(
            (owner, deltas, flips, successes),
            powers,
            bits,
            grid,
            flip_probability,
            success_probability,
            max_exact_reports,
            samples,
            rng_seed,
        )
        best = None
        for r in active:
            candidate_p = np.asarray(powers, dtype=float).copy()
            candidate_b = np.asarray(bits, dtype=int).copy()
            candidate_p[r] += 1
            value = _target_pd(
                (owner, deltas, flips, successes),
                candidate_p,
                candidate_b,
                grid,
                flip_probability,
                success_probability,
                max_exact_reports,
                samples,
                rng_seed,
            ) - base
            if best is None or value > best[0]:
                best = (float(value), r)
        return int(best[1])
    return max(
        active,
        key=lambda r: power_gain_coefficient(
            float(deltas[r]),
            int(bits[r]),
            float(flips[r]),
            float(successes[r]),
        ),
    )


def _add_freed_units(
    power_row,
    bit_row,
    report_index,
    freed,
    *,
    max_power,
    max_bits,
):
    remaining = freed
    while remaining > 0 and power_row[report_index] < max_power:
        power_row[report_index] += 1
        remaining -= 1
    while remaining > 0 and bit_row[report_index] < max_bits:
        bit_row[report_index] += 1
        remaining -= 1
    return remaining


def _copy_rows(powers, bits, *row_ids):
    """Path-copy a candidate: share untouched rows, copy only mutated ones.

    Every exchange move only modifies the rows of the source target ``q``
    (and, for cross-target moves, the destination target ``d``).  Copying
    the full matrix per candidate costs O(Q*R) element copies; sharing the
    untouched rows is safe because the generator never mutates its input
    matrix and the consumer only reads yielded rows.  Untouched rows keep
    their identity (``candidate[q] is powers[q]``), which lets downstream
    key/evaluation code skip content comparison with reference equality.
    """
    new_p = list(powers)
    new_b = list(bits)
    for row_id in row_ids:
        new_p[row_id] = powers[row_id].copy()
        new_b[row_id] = bits[row_id].copy()
    return new_p, new_b


def _iter_candidates(
    scenario,
    powers,
    bits,
    *,
    max_power,
    max_bits,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    grid: int = 16,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng_seed: int = 0,
    probe_mask=None,
    target_only: int | None = None,
    skip_targets: set[int] | None = None,
):
    """Yield feasible single-exchange power/bit/atom moves.

    ``target_only`` restricts the generator to moves that only touch a
    single target's rows (its own report rows; cross-target transfers are
    skipped), which is the building block of the block-coordinate phase.
    ``skip_targets`` excludes targets that no move can improve (see
    :func:`_uncoverable_targets`); skipping is exact because any candidate
    touching such a target is never accepted.
    """
    q_count = len(scenario)
    reports = _report_count(scenario[0])
    for q in range(q_count):
        if target_only is not None and q != target_only:
            continue
        if skip_targets and q in skip_targets:
            continue
        target_q = scenario[q]
        active_q = _active_reports(bits[q])
        winner_q = _winner_index(
            target_q,
            bits[q],
            powers[q],
            flip_probability,
            success_probability,
            grid,
            max_exact_reports,
            samples,
            rng_seed,
        )
        for s in range(reports):
            for d in range(reports):
                if (
                    bits[q][d] > 0
                    or bits[q][s] <= 0
                    or powers[q][s] + bits[q][s] < 2
                ):
                    continue
                if probe_mask is not None and probe_mask[q][d] == 0:
                    continue
                if powers[q][s] >= 2:
                    new_p, new_b = _copy_rows(powers, bits, q)
                    new_p[q][s] -= 2
                    new_p[q][d] = 1
                    new_b[q][d] = 1
                    yield new_p, new_b
                elif bits[q][s] >= 2 and powers[q][s] >= 1:
                    new_p, new_b = _copy_rows(powers, bits, q)
                    new_p[q][s] -= 1
                    new_b[q][s] -= 1
                    new_p[q][d] = 1
                    new_b[q][d] = 1
                    yield new_p, new_b
            if powers[q][s] > 0 and bits[q][s] > 0:
                freed = int(powers[q][s] + bits[q][s])
                for d in range(reports):
                    if d == s:
                        continue
                    if probe_mask is not None and probe_mask[q][d] == 0:
                        continue
                    new_p, new_b = _copy_rows(powers, bits, q)
                    new_p[q][s] = 0
                    new_b[q][s] = 0
                    if new_b[q][d] > 0:
                        remaining = _add_freed_units(
                            new_p[q],
                            new_b[q],
                            d,
                            freed,
                            max_power=max_power,
                            max_bits=max_bits,
                        )
                    elif freed >= 2:
                        new_p[q][d] = 1
                        new_b[q][d] = 1
                        remaining = _add_freed_units(
                            new_p[q],
                            new_b[q],
                            d,
                            freed - 2,
                            max_power=max_power,
                            max_bits=max_bits,
                        )
                    else:
                        continue
                    if remaining == 0:
                        yield new_p, new_b
            if powers[q][s] > 0:
                for d in active_q:
                    if (
                        d == s
                        or powers[q][d] + powers[q][s] > max_power
                    ):
                        continue
                    new_p, new_b = _copy_rows(powers, bits, q)
                    new_p[q][d] += new_p[q][s]
                    new_p[q][s] = 0
                    yield new_p, new_b
            if winner_q is not None and s != winner_q:
                if powers[q][s] > 0 and powers[q][winner_q] < max_power:
                    new_p, new_b = _copy_rows(powers, bits, q)
                    new_p[q][s] -= 1
                    new_p[q][winner_q] += 1
                    yield new_p, new_b
                if (
                    bits[q][s] > 0
                    and bits[q][winner_q] < max_bits
                    and (len(active_q) > 1 or bits[q][s] > 1)
                ):
                    new_p, new_b = _copy_rows(powers, bits, q)
                    new_b[q][s] -= 1
                    new_b[q][winner_q] += 1
                    yield new_p, new_b
                if len(active_q) >= 2:
                    freed = int(powers[q][s] + bits[q][s])
                    new_p, new_b = _copy_rows(powers, bits, q)
                    new_p[q][s] = 0
                    new_b[q][s] = 0
                    remaining = _add_freed_units(
                        new_p[q],
                        new_b[q],
                        winner_q,
                        freed,
                        max_power=max_power,
                        max_bits=max_bits,
                    )
                    if remaining == 0:
                        yield new_p, new_b
            for d in range(q_count):
                if target_only is not None:
                    break
                if d == q:
                    continue
                if skip_targets and d in skip_targets:
                    continue
                active_d = _active_reports(bits[d])
                for dd in active_d:
                    if powers[q][s] > 0 and powers[d][dd] < max_power:
                        new_p, new_b = _copy_rows(powers, bits, q, d)
                        new_p[q][s] -= 1
                        new_p[d][dd] += 1
                        yield new_p, new_b
                    if (
                        bits[q][s] > 0
                        and bits[d][dd] < max_bits
                        and (len(active_q) > 1 or bits[q][s] > 1)
                    ):
                        new_p, new_b = _copy_rows(powers, bits, q, d)
                        new_b[q][s] -= 1
                        new_b[d][dd] += 1
                        yield new_p, new_b
                if len(active_q) >= 2:
                    freed = int(powers[q][s] + bits[q][s])
                    for dd in active_d:
                        new_p, new_b = _copy_rows(powers, bits, q, d)
                        new_p[q][s] = 0
                        new_b[q][s] = 0
                        remaining = _add_freed_units(
                            new_p[d],
                            new_b[d],
                            dd,
                            freed,
                            max_power=max_power,
                            max_bits=max_bits,
                        )
                        if remaining == 0:
                            yield new_p, new_b
                    for dd in range(reports):
                        if (
                            bits[d][dd] > 0
                            or (
                                probe_mask is not None
                                and probe_mask[d][dd] == 0
                            )
                        ):
                            continue
                        new_p, new_b = _copy_rows(powers, bits, q, d)
                        new_p[q][s] = 0
                        new_b[q][s] = 0
                        if freed >= 2:
                            new_p[d][dd] = 1
                            new_b[d][dd] = 1
                            remaining = freed - 2
                            remaining = _add_freed_units(
                                new_p[d],
                                new_b[d],
                                dd,
                                remaining,
                                max_power=max_power,
                                max_bits=max_bits,
                            )
                            if remaining == 0:
                                yield new_p, new_b


def maxmin_refine(
    scenario,
    powers,
    bits,
    *,
    max_power,
    max_bits: int = 2,
    max_rounds: int = 100,
    grid: int = 16,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    floors=None,
    weights=None,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng_seed: int = 0,
    candidate_budget: int = 32,
    probe_mask=None,
):
    """NOMP-style discrete refinement with a hard iteration cap.

    Each global round generates all single-exchange moves, prunes them to
    the top deflection variants per touched target (heuristic Top-L), and
    ranks the survivors by the lexicographic max-min deflection proxy.
    The top ``candidate_budget`` moves are then evaluated in that order and
    every move that strictly improves the round-start max-min vector is
    accepted, provided it does not touch a row already modified by an
    accepted move in the same round.  Because each accepted move raises its
    own target row (a single-row lexicographic improvement must increase
    the row value), the merged vector is componentwise non-decreasing in
    the sorted P_D vector, so the lexicographic max-min value never
    decreases and the loop terminates at a finite local optimum.  Each
    round costs one O(C log K) selection, while a single-move climb would
    need up to Q times more rounds.
    """
    rounds_used = 0
    q_count = len(scenario)
    # 无证据目标惰性: 单次激活都无法提升 P_D 的目标, 任何移动都不会被
    # 接受 (leximin 单调), 从候选生成中精确剔除.
    skip_targets = _uncoverable_targets(
        scenario,
        grid=grid,
        flip_probability=flip_probability,
        success_probability=success_probability,
        max_exact_reports=max_exact_reports,
        samples=samples,
        rng_seed=rng_seed,
    )
    # 跨轮共享的 deflection 缓存: 候选行内容在轮间高度重复 (path-copy
    # 只改少数元素), 跨轮缓存把 miss 计算压缩到每种行内容一次.
    proxy_cache: dict[tuple, float] = {}

    def _row_deflection(q, p_row, b_row):
        key = (
            q,
            tuple(int(v) for v in p_row),
            tuple(int(v) for v in b_row),
        )
        value = proxy_cache.get(key)
        if value is None:
            value = _single_target_deflection(
                scenario[q],
                p_row,
                b_row,
                flip_probability,
                success_probability,
            )
            proxy_cache[key] = value
        return value

    for _ in range(max_rounds):
        old_raw = target_scores(
            scenario,
            powers,
            bits,
            grid,
            flip_probability,
            success_probability,
            max_exact_reports,
            samples,
            rng_seed,
        )
        old_values = (
            np.sort(qos_scores(old_raw, floors, weights))
            if floors is not None
            else np.sort(old_raw)
        )
        candidates = list(_iter_candidates(
            scenario,
            powers,
            bits,
            max_power=max_power,
            max_bits=max_bits,
            flip_probability=flip_probability,
            success_probability=success_probability,
            grid=grid,
            max_exact_reports=max_exact_reports,
            samples=samples,
            rng_seed=rng_seed,
            probe_mask=probe_mask,
            skip_targets=skip_targets,
        ))
        if not candidates:
            break

        # 生成期 Top-L 预选: 每个候选按变化行的 deflection 最高的目标分组,
        # 每组只保留 deflection 最高的 L 个变体. deflection 是 P_D 的单调
        # 代理 (deflection 组合决定融合检测), 同组内 deflection 越高代理
        # 排序越靠前; 这是启发式剪枝, 其 worst-case 不劣化性质由
        # exact-frontier 测试 (test_nomp_refinement.py) 数值验证, 不是
        # 证明: 被剪掉的候选也可能携带与代理序相反的真实 P_D 提升.
        round_powers = powers
        round_bits = bits
        grouped: dict[int, list[tuple[float, int]]] = {}
        for idx, candidate in enumerate(candidates):
            best_q = -1
            best_d = -1.0
            for q in range(q_count):
                if (
                    candidate[0][q] is round_powers[q]
                    and candidate[1][q] is round_bits[q]
                ):
                    continue
                d = _row_deflection(q, candidate[0][q], candidate[1][q])
                if d > best_d:
                    best_q, best_d = q, d
            if best_q >= 0:
                grouped.setdefault(best_q, []).append((best_d, idx))
        keep_indices: set[int] = set()
        for q, entries in grouped.items():
            for _, idx in nlargest(candidate_budget, entries):
                keep_indices.add(idx)
        if keep_indices:
            candidates = [c for i, c in enumerate(candidates) if i in keep_indices]

        base_proxy = deflection_proxy(
            scenario,
            powers,
            bits,
            flip_probability,
            success_probability,
        )
        base_p_rows = [tuple(int(v) for v in row) for row in powers]
        base_b_rows = [tuple(int(v) for v in row) for row in bits]
        # 轮初基准: 候选行引用轮初行 (path-copy), 因此代理排序基准固定
        # 在轮初对象上; 候选行内容相同的部分用 base_p_rows/base_b_rows
        # 判定, 与 ``is`` 短路等价.

        def proxy_key(candidate):
            values = list(base_proxy)
            for q in range(q_count):
                if (
                    candidate[0][q] is round_powers[q]
                    or (
                        tuple(int(v) for v in candidate[0][q]) == base_p_rows[q]
                        and tuple(int(v) for v in candidate[1][q]) == base_b_rows[q]
                    )
                ):
                    continue
                values[q] = _row_deflection(
                    q, candidate[0][q], candidate[1][q]
                )
            return tuple(np.sort(values))

        # Stable Top-K: identical to ``sorted(key=..., reverse=True)[:K]``.
        # Sorting all C candidates costs O(C log C) tuple comparisons, while
        # partial selection only needs O(C log K); with reference equality
        # the (key, -index) tie-break reproduces the stable sort order.
        ranked = nlargest(
            candidate_budget,
            enumerate(candidates),
            key=lambda item: (proxy_key(item[1]), -item[0]),
        )
        accepted_any = False
        accepted_rows: set[int] = set()
        for _, candidate in ranked:
            # 本轮已接受移动修改过的目标行. 单目标候选的 deflection 只依赖
            # 自身行 (目标独立性), 因此轮初的代理排序对其他目标的接受保持
            # 有效; 冲突的候选 (改同一行) 被跳过, 保证轮内接受的移动互不
            # 重叠, 合并后的向量逐分量不小于轮初向量 -> lexicographic 单调.
            rows = {
                q
                for q in range(q_count)
                if not (
                    (candidate[0][q] is powers[q] or _rows_equal(candidate[0][q], powers[q]))
                    and (candidate[1][q] is bits[q] or _rows_equal(candidate[1][q], bits[q]))
                )
            }
            if not rows or (rows & accepted_rows):
                continue
            new_raw = list(old_raw)
            for q in rows:
                new_raw[q] = _target_score_at(
                    scenario, q, candidate[0], candidate[1], grid,
                    flip_probability, success_probability,
                    max_exact_reports, samples, rng_seed,
                )
            new_values = (
                np.sort(qos_scores(new_raw, floors, weights))
                if floors is not None
                else np.sort(new_raw)
            )
            if leximin_improves(old_values, new_values):
                powers, bits = candidate
                accepted_rows |= rows
                accepted_any = True
        if not accepted_any:
            break
        rounds_used += 1
    return powers, bits, rounds_used


def wta_greedy_joint_multi(
    scenario,
    budget,
    *,
    min_cover: bool = False,
    max_bits: int = 2,
    max_power=None,
    grid: int = 16,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    floors=None,
    weights=None,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng_seed: int = 0,
    probe_mask=None,
):
    """Online WTA greedy allocation with optional per-target minimum cover."""
    if max_power is None:
        max_power = int(budget)
    reports = _report_count(scenario[0])
    if probe_mask is not None:
        min_cover = False
    if min_cover:
        powers, bits, used = initial_min_cover(
            scenario,
            budget,
            max_bits=max_bits,
            flip_probability=flip_probability,
            success_probability=success_probability,
            grid=grid,
            max_exact_reports=max_exact_reports,
            samples=samples,
            rng_seed=rng_seed,
        )
    else:
        powers = [np.zeros(reports, dtype=int) for _ in scenario]
        bits = [np.zeros(reports, dtype=int) for _ in scenario]
        used = 0
    steps = 0
    gains: dict[tuple, float] = {}
    scores = target_scores(
        scenario,
        powers,
        bits,
        grid,
        flip_probability,
        success_probability,
        max_exact_reports,
        samples,
        rng_seed,
    )

    def mean_of(values):
        if floors is not None:
            return float(np.mean(qos_scores(values, floors, weights)))
        return float(np.mean(values))

    while True:
        mean_before = mean_of(scores)
        best = None
        # 增量 gain 缓存: 激活/bit/power 只改变目标 q 的 P_D, 其他目标的
        # 候选 gain 在相同 (powers, bits) 下不变 (P_D 目标独立), 因此
        # 每步只需重算被修改目标的候选, 其余候选的 gain 精确复用.
        # 这等价于每步全量重算 (数值逐位一致), 但融合评估次数从
        # O(steps * Q * R) 降到 O(steps * R + Q * R).
        if not gains:
            for q, target in enumerate(scenario):
                active = _active_reports(bits[q])
                for r in range(reports):
                    if (
                        bits[q][r] > 0
                        or used + 2 > budget
                        or (
                            probe_mask is not None
                            and probe_mask[q][r] == 0
                        )
                    ):
                        continue
                    old_b, old_p = bits[q].copy(), powers[q].copy()
                    bits[q][r] = 1
                    powers[q][r] = 1
                    new_score = _target_score_at(
                        scenario, q, powers, bits, grid,
                        flip_probability, success_probability,
                        max_exact_reports, samples, rng_seed,
                    )
                    bits[q], powers[q] = old_b, old_p
                    gain = mean_of([*scores[:q], new_score, *scores[q + 1:]]) - mean_before
                    gains[("activate", q, r)] = gain
                for r in active:
                    if bits[q][r] >= max_bits or used + 1 > budget:
                        continue
                    old_b = bits[q].copy()
                    bits[q][r] += 1
                    new_score = _target_score_at(
                        scenario, q, powers, bits, grid,
                        flip_probability, success_probability,
                        max_exact_reports, samples, rng_seed,
                    )
                    bits[q] = old_b
                    gain = mean_of([*scores[:q], new_score, *scores[q + 1:]]) - mean_before
                    gains[("bit", q, r)] = gain
                if active:
                    winner = _winner_index(
                        target,
                        bits[q],
                        powers[q],
                        flip_probability,
                        success_probability,
                        grid,
                        max_exact_reports,
                        samples,
                        rng_seed,
                    )
                    if powers[q][winner] < max_power and used + 1 <= budget:
                        old_p = powers[q].copy()
                        powers[q][winner] += 1
                        new_score = _target_score_at(
                            scenario, q, powers, bits, grid,
                            flip_probability, success_probability,
                            max_exact_reports, samples, rng_seed,
                        )
                        powers[q] = old_p
                        gain = mean_of([*scores[:q], new_score, *scores[q + 1:]]) - mean_before
                        gains[("power", q, winner)] = gain
        for (kind, q, index), gain in gains.items():
            if gain <= 0:
                continue
            if kind == "activate":
                if (
                    bits[q][index] > 0
                    or used + 2 > budget
                    or (
                        probe_mask is not None
                        and probe_mask[q][index] == 0
                    )
                ):
                    continue
                key = (gain / 2.0, gain, q, "activate", index)
            elif kind == "bit":
                if bits[q][index] < 1 or bits[q][index] >= max_bits or used + 1 > budget:
                    continue
                key = (gain, gain, q, "bit", index)
            else:
                if powers[q][index] >= max_power or used + 1 > budget:
                    continue
                key = (gain, gain, q, "power", index)
            if best is None or key > best[0]:
                best = (key, kind, q, index)
        if best is None:
            break
        _, kind, q, index = best
        if kind == "activate":
            bits[q][index] = 1
            powers[q][index] = 1
            used += 2
        elif kind == "bit":
            bits[q][index] += 1
            used += 1
        else:
            powers[q][index] += 1
            used += 1
        scores[q] = _target_score_at(
            scenario, q, powers, bits, grid,
            flip_probability, success_probability,
            max_exact_reports, samples, rng_seed,
        )
        # 目标 q 的状态已变: 其全部候选 gain 失效, 其余目标保持.
        # 重算必须以当前 scores 的均值作基准 (循环头的 mean_before 已过期).
        mean_now = mean_of(scores)
        for key in [key for key in gains if key[1] == q]:
            del gains[key]
        target_q = scenario[q]
        active_q = _active_reports(bits[q])
        for r in range(reports):
            if (
                bits[q][r] > 0
                or used + 2 > budget
                or (
                    probe_mask is not None
                    and probe_mask[q][r] == 0
                )
            ):
                continue
            old_b, old_p = bits[q].copy(), powers[q].copy()
            bits[q][r] = 1
            powers[q][r] = 1
            new_score = _target_score_at(
                scenario, q, powers, bits, grid,
                flip_probability, success_probability,
                max_exact_reports, samples, rng_seed,
            )
            bits[q], powers[q] = old_b, old_p
            gain = mean_of([*scores[:q], new_score, *scores[q + 1:]]) - mean_now
            gains[("activate", q, r)] = gain
        for r in active_q:
            if bits[q][r] >= max_bits or used + 1 > budget:
                continue
            old_b = bits[q].copy()
            bits[q][r] += 1
            new_score = _target_score_at(
                scenario, q, powers, bits, grid,
                flip_probability, success_probability,
                max_exact_reports, samples, rng_seed,
            )
            bits[q] = old_b
            gain = mean_of([*scores[:q], new_score, *scores[q + 1:]]) - mean_now
            gains[("bit", q, r)] = gain
        if active_q:
            winner = _winner_index(
                target_q,
                bits[q],
                powers[q],
                flip_probability,
                success_probability,
                grid,
                max_exact_reports,
                samples,
                rng_seed,
            )
            if powers[q][winner] < max_power and used + 1 <= budget:
                old_p = powers[q].copy()
                powers[q][winner] += 1
                new_score = _target_score_at(
                    scenario, q, powers, bits, grid,
                    flip_probability, success_probability,
                    max_exact_reports, samples, rng_seed,
                )
                powers[q] = old_p
                gain = mean_of([*scores[:q], new_score, *scores[q + 1:]]) - mean_now
                gains[("power", q, winner)] = gain
        steps += 1

    return {
        "powers": powers,
        "bits": bits,
        "worst_pd": float(min(target_scores(
            scenario,
            powers,
            bits,
            grid,
            flip_probability,
            success_probability,
            max_exact_reports,
            samples,
            rng_seed,
        ))),
        "used": int(sum(
            int(powers[q].sum()) + int(bits[q].sum())
            for q in range(len(scenario))
        )),
        "steps": steps,
    }


def nomp_wta_greedy_joint_multi(
    scenario,
    budget,
    *,
    max_bits: int = 2,
    max_power=None,
    max_rounds: int = 100,
    grid: int = 16,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    floors=None,
    weights=None,
    max_exact_reports: int = 8,
    samples: int = 2048,
    rng_seed: int = 0,
    candidate_budget: int = 32,
    probe_mask=None,
):
    """WTA greedy with minimum cover, followed by NOMP-style refinement."""
    if max_power is None:
        max_power = int(budget)
    greedy = wta_greedy_joint_multi(
        scenario,
        budget,
        min_cover=probe_mask is None,
        max_bits=max_bits,
        max_power=max_power,
        grid=grid,
        flip_probability=flip_probability,
        success_probability=success_probability,
        floors=floors,
        weights=weights,
        max_exact_reports=max_exact_reports,
        samples=samples,
        rng_seed=rng_seed,
        probe_mask=probe_mask,
    )
    powers, bits, refine_rounds = maxmin_refine(
        scenario,
        greedy["powers"],
        greedy["bits"],
        max_power=max_power,
        max_bits=max_bits,
        max_rounds=max_rounds,
        grid=grid,
        flip_probability=flip_probability,
        success_probability=success_probability,
        floors=floors,
        weights=weights,
        max_exact_reports=max_exact_reports,
        samples=samples,
        rng_seed=rng_seed,
        candidate_budget=candidate_budget,
        probe_mask=probe_mask,
    )
    raw = target_scores(
        scenario,
        powers,
        bits,
        grid,
        flip_probability,
        success_probability,
        max_exact_reports,
        samples,
        rng_seed,
    )
    return {
        "powers": powers,
        "bits": bits,
        "worst_pd": float(min(raw)),
        "qos_worst": (
            float(np.min(qos_scores(raw, floors, weights)))
            if floors is not None
            else None
        ),
        "used": int(sum(
            int(powers[q].sum()) + int(bits[q].sum())
            for q in range(len(scenario))
        )),
        "greedy_steps": greedy["steps"],
        "refine_rounds": refine_rounds,
    }
