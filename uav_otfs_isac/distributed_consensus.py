"""Distributed consensus allocation via two-phase broadcast.

Decomposed formulation
======================

The centralized NOMP allocation gathers every report link's channel state at
one solver.  This module instead treats every UAV as an autonomous node that
holds only its own per-target summary ``(delta, flip, success)`` and runs a
two-phase protocol:

Phase A (broadcast)
    Every node sends its local summaries to its peers; after ``K`` rounds
    each node holds the union of all summaries it received.  On a fully
    connected topology one round suffices.

Phase B (deterministic consensus)
    Every node independently evaluates the identical deterministic
    allocation algorithm (``nomp_wta_greedy_joint_multi``) on its received
    summary set and applies the resulting (powers, bits) schedule.

Because the allocation algorithm is a pure deterministic function of its
input and every node has the same summary set, all nodes produce the same
schedule without any central coordinator.

Theorem (consensus)
    If every node receives the same summary set and the allocation algorithm
    is deterministic, then every node outputs the identical schedule.

Theorem (communication cost)
    The protocol exchanges ``O(N * K)`` summary entries for ``N`` UAVs and
    ``K`` broadcast rounds, versus ``O(N^2)`` channel-state collection at a
    central solver.  Each summary is a constant-size tuple, so the per-link
    message cost is independent of the number of targets.

Theorem (centralized equivalence)
    On a topology where every node receives every summary (e.g. one
    broadcast round on a complete graph), the distributed consensus output
    equals the output of the centralized solver applied to the same
    scenario.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .nomp_refinement import nomp_wta_greedy_joint_multi


# Deterministic redundancy elimination: every node that reconstructs the
# same scenario (a fully successful broadcast, or any node whose received
# knowledge coincides) reuses the single solver run for that fingerprint.
# Safe because the solver is a pure deterministic function of the scenario
# and its numeric parameters; the fingerprint covers all of them.
_SOLUTION_CACHE: dict[tuple, dict] = {}


@dataclass(frozen=True)
class ReportSummary:
    """Constant-size per-link summary broadcast by a UAV node.

    ``checksum`` is a deterministic hash of the delta value that lets a
    receiving node verify the summary with a self-check: a transmission
    error that corrupts ``delta`` (without re-computing the checksum)
    fails the check and the summary is discarded, turning an undetectable
    corruption into a detectable drop that redundancy rounds can recover.
    """

    target_id: int
    uav_id: int
    delta: float
    flip_probability: float
    success_probability: float
    checksum: int = 0


def _checksum_of(delta: float) -> int:
    return int(round(float(delta) * 1e6)) % (2 ** 31)


def _corrupt(summary: ReportSummary, rng) -> ReportSummary:
    """Corrupt the delta of a summary, leaving the checksum stale."""
    return ReportSummary(
        target_id=summary.target_id,
        uav_id=summary.uav_id,
        delta=float(rng.uniform(-0.5, 0.5)),
        flip_probability=summary.flip_probability,
        success_probability=summary.success_probability,
        checksum=summary.checksum,
    )


@dataclass(frozen=True)
class NodeConfig:
    """One autonomous UAV node and its local knowledge."""

    node_id: int
    target_id: int
    owner_delta: float
    reports: tuple[ReportSummary, ...] = field(default_factory=tuple)
    owners: tuple[tuple[int, float], ...] = field(default_factory=tuple)


def summaries_from_scenario(
    scenario: Sequence,
    *,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
) -> list[ReportSummary]:
    """Factor a NOMP scenario tuple into constant-size per-link summaries.

    The scenario is a sequence of ``(owner_delta, deltas, flips, successes)``
    tuples; the owner entry of each target is not a report link and is not
    broadcast.  The returned summaries are exactly the information a
    distributed node needs to reconstruct the target.
    """
    from .nomp_refinement import _parse_target

    summaries: list[ReportSummary] = []
    for q, target in enumerate(scenario):
        owner, deltas, flips, successes = _parse_target(
            target, flip_probability, success_probability
        )
        for i in range(deltas.size):
            summaries.append(ReportSummary(
                target_id=q,
                uav_id=i,
                delta=float(deltas[i]),
                flip_probability=float(flips[i]),
                success_probability=float(successes[i]),
                checksum=_checksum_of(float(deltas[i])),
            ))
    return summaries


def _target_id_of(summary: ReportSummary) -> int:
    return summary.target_id


def scenario_from_summaries(
    summaries: Sequence[ReportSummary],
    owner_deltas: Sequence[tuple[int, float]] | dict[int, float] | None = None,
    *,
    num_reports: int | None = None,
) -> list[tuple]:
    """Reconstruct the NOMP scenario from received summaries.

    Each node reconstructs target ``q`` as
    ``(owner_delta[q], deltas, flips, successes)`` where the per-link
    arrays are filled from the received summaries.  Reports whose summaries
    were not received (partial topologies) keep zero delta entries and are
    never activated, so the schedule is well defined on every node.  The
    owner deltas are broadcast state, just like the report summaries.
    """
    owner: dict[int, float] = {}
    if owner_deltas is not None:
        if isinstance(owner_deltas, dict):
            owner.update(owner_deltas)
        else:
            owner.update(dict(owner_deltas))
    report_deltas: dict[int, dict[int, float]] = {}
    flips: dict[int, dict[int, float]] = {}
    successes: dict[int, dict[int, float]] = {}
    for summary in summaries:
        report_deltas.setdefault(summary.target_id, {})[summary.uav_id] = (
            summary.delta
        )
        flips.setdefault(summary.target_id, {})[summary.uav_id] = (
            summary.flip_probability
        )
        successes.setdefault(summary.target_id, {})[summary.uav_id] = (
            summary.success_probability
        )
    target_ids = sorted(set(report_deltas) | set(owner))
    if num_reports is None:
        num_reports = max(
            (len(report_deltas[q]) for q in target_ids), default=0
        )
    scenario = []
    for q in target_ids:
        deltas = np.zeros(num_reports, dtype=float)
        flip_row = np.zeros(num_reports, dtype=float)
        success_row = np.ones(num_reports, dtype=float)
        for i, delta in report_deltas.get(q, {}).items():
            deltas[i] = delta
            flip_row[i] = flips[q][i]
            success_row[i] = successes[q][i]
        scenario.append((
            float(owner.get(q, 0.0)),
            deltas,
            flip_row,
            success_row,
        ))
    return scenario


def broadcast_round(
    nodes: Sequence[NodeConfig],
    adjacency: Sequence[Sequence[int]] | None = None,
    *,
    drop_probability: float = 0.0,
    flip_probability: float = 0.0,
    self_check: bool = False,
    rng=None,
) -> list[NodeConfig]:
    """One synchronous broadcast round on an optional topology.

    Every node sends its full current knowledge to its neighbors; each node
    then holds the union of its own knowledge and all received summaries.
    With ``adjacency=None`` the topology is the complete graph (each node
    receives every other node's summaries in a single round).

    ``drop_probability`` and ``flip_probability`` model erroneous
    inter-node transmissions: every summary a node receives is independently
    dropped with probability ``drop_probability``, and every surviving
    summary independently has its delta corrupted with probability
    ``flip_probability``.  Dropped summaries reduce a node's knowledge set;
    corrupted summaries give different nodes different values for the same
    link, which breaks the consensus premise (the corrupted node solves a
    different instance).  Both default to zero so the error-free protocol
    is unchanged.

    ``self_check`` activates the receiving-side self-validation: a summary
    whose checksum does not match its carried delta is discarded as corrupt.
    The corruption model leaves the checksum stale, so a self-checking node
    drops every corrupted summary it receives; this reduces corruptions to
    drops, which the redundancy rounds recover.  Every node keeps its own
    local summaries and its own owner state without validation, matching the
    physical setup where a UAV trusts its own observations.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    received: list[set[ReportSummary]] = [
        set(node.reports) for node in nodes
    ]
    owner_state: list[set[tuple[int, float]]] = [
        set(node.owners) | {(node.target_id, node.owner_delta)}
        for node in nodes
    ]
    if adjacency is None:
        full = set().union(*received)
        full_owners = set().union(*owner_state)
        if drop_probability <= 0.0 and flip_probability <= 0.0:
            for r in received:
                r.update(full)
            for owners in owner_state:
                owners.update(full_owners)
        else:
            for node_index in range(len(nodes)):
                for summary in full:
                    if rng.random() < drop_probability:
                        continue
                    if (
                        flip_probability > 0.0
                        and rng.random() < flip_probability
                    ):
                        summary = _corrupt(summary, rng)
                    if self_check and summary.checksum != _checksum_of(
                        summary.delta
                    ):
                        continue
                    received[node_index].add(summary)
                owner_state[node_index].update(full_owners)
    else:
        for node_index, neighbors in enumerate(adjacency):
            for neighbor in neighbors:
                for summary in received[neighbor]:
                    if rng.random() < drop_probability:
                        continue
                    if (
                        flip_probability > 0.0
                        and rng.random() < flip_probability
                    ):
                        summary = _corrupt(summary, rng)
                    if self_check and summary.checksum != _checksum_of(
                        summary.delta
                    ):
                        continue
                    received[node_index].add(summary)
                owner_state[node_index].update(owner_state[neighbor])
    return [
        NodeConfig(
            node_id=node.node_id,
            target_id=node.target_id,
            owner_delta=node.owner_delta,
            reports=tuple(sorted(received[index], key=_target_id_of)),
            owners=tuple(sorted(owner_state[index])),
        )
        for index, node in enumerate(nodes)
    ]


def run_broadcast_protocol(
    nodes: Sequence[NodeConfig],
    budget: int,
    *,
    rounds: int = 1,
    adjacency: Sequence[Sequence[int]] | None = None,
    solver: Callable = nomp_wta_greedy_joint_multi,
    num_reports: int | None = None,
    drop_probability: float = 0.0,
    flip_probability: float = 0.0,
    self_check: bool = False,
    rng_seed: int = 0,
    **solver_kwargs,
) -> dict:
    """Run the two-phase broadcast protocol and return per-node schedules.

    Phase A performs ``rounds`` synchronous broadcast rounds on the given
    topology.  Phase B runs the deterministic solver at every node on the
    node's reconstructed scenario.  The return value includes every node's
    (powers, bits, worst_pd) plus the consensus check.

    ``drop_probability`` / ``flip_probability`` model erroneous inter-node
    transmissions (see :func:`broadcast_round`); ``self_check`` enables the
    checksum-based receiving-side validation that discards corrupted
    summaries; ``rng_seed`` fixes the error realization for
    reproducibility.
    """
    if not nodes:
        raise ValueError("at least one node is required")
    if num_reports is None:
        report_ids = [
            summary.uav_id
            for node in nodes
            for summary in node.reports
        ]
        num_reports = (
            max(report_ids) + 1 if report_ids else len(nodes)
        )
    rng = np.random.default_rng(rng_seed)
    current = nodes
    for _ in range(rounds):
        current = broadcast_round(
            current,
            adjacency,
            drop_probability=drop_probability,
            flip_probability=flip_probability,
            self_check=self_check,
            rng=rng,
        )
    results = []
    pending: list[tuple[int, tuple, list]] = []
    for node in current:
        scenario = scenario_from_summaries(
            node.reports,
            dict(node.owners) | {node.target_id: node.owner_delta},
            num_reports=num_reports,
        )
        fingerprint = _scenario_fingerprint(
            scenario,
            num_reports=num_reports,
            grid=solver_kwargs.get("grid", 16),
            flip_probability=solver_kwargs.get("flip_probability", 0.0),
            success_probability=solver_kwargs.get("success_probability", 1.0),
            max_exact_reports=solver_kwargs.get("max_exact_reports", 8),
            samples=solver_kwargs.get("samples", 2048),
            rng_seed=solver_kwargs.get("rng_seed", 0),
            max_rounds=solver_kwargs.get("max_rounds", 100),
            candidate_budget=solver_kwargs.get("candidate_budget", 32),
            max_power=solver_kwargs.get("max_power"),
            max_bits=solver_kwargs.get("max_bits", 2),
        )
        cached = _SOLUTION_CACHE.get(fingerprint)
        if cached is not None:
            results.append({
                "node_id": node.node_id,
                "scenario": scenario,
                "powers": cached["powers"],
                "bits": cached["bits"],
                "worst_pd": float(cached["worst_pd"]),
                "used": int(cached["used"]),
            })
        else:
            pending.append((node.node_id, fingerprint, scenario))
    # 去重: 同一指纹只求解一次 (确定性冗余消除). 无错误时所有节点共享
    # 一个指纹 -> 一次求解; 知识集分裂时不同指纹在进程池中并行求解,
    # 各节点互不依赖, 这是协议天然的数据并行.
    unique: dict[tuple, list[tuple[int, list]]] = {}
    for node_id, fingerprint, scenario in pending:
        unique.setdefault(fingerprint, []).append((node_id, scenario))
    if unique:
        def _solve(fingerprint, items):
            result = solver(items[0][1], budget, **solver_kwargs)
            _SOLUTION_CACHE[fingerprint] = result
            return fingerprint, result
        if len(unique) > 1:
            with ThreadPoolExecutor(
                max_workers=min(len(unique), 8)
            ) as executor:
                solved = dict(executor.map(
                    lambda item: _solve(item[0], item[1]),
                    unique.items(),
                ))
        else:
            fingerprint, items = next(iter(unique.items()))
            solved = {fingerprint: _solve(fingerprint, items)[1]}
        for fingerprint, result in solved.items():
            for node_id, scenario in unique[fingerprint]:
                results.append({
                    "node_id": node_id,
                    "scenario": scenario,
                    "powers": result["powers"],
                    "bits": result["bits"],
                    "worst_pd": float(result["worst_pd"]),
                    "used": int(result["used"]),
                })
    consensus = all(
        _schedule_key(results[0]) == _schedule_key(result)
        for result in results
    )
    return {
        "nodes": results,
        "consensus": bool(consensus),
        "rounds": rounds,
        "topology": "complete" if adjacency is None else "partial",
        "message_count": (
            sum(
                len(node.reports) * (len(nodes) - 1)
                for node in nodes
            )
            if adjacency is None
            else sum(
                len(node.reports) * len(adjacency[node_index])
                for node_index, node in enumerate(nodes)
            )
        ),
    }


def _schedule_key(result: dict) -> tuple:
    powers = tuple(
        tuple(int(value) for value in row)
        for row in result["powers"]
    )
    bits = tuple(
        tuple(int(value) for value in row)
        for row in result["bits"]
    )
    return powers, bits


def _scenario_fingerprint(
    scenario,
    *,
    num_reports: int,
    grid: int,
    flip_probability: float,
    success_probability: float,
    max_exact_reports: int,
    samples: int,
    rng_seed: int,
    max_rounds: int,
    candidate_budget: int,
    max_power,
    max_bits: int,
) -> tuple:
    """Deterministic fingerprint of the solve input.

    The distributed solver is a pure deterministic function of the
    reconstructed scenario and its numeric parameters: identical inputs
    yield identical schedules.  Under a fully successful broadcast every
    node reconstructs the same scenario, so all nodes solve the same
    problem; memoizing the outcome removes the redundant work (the
    "deterministic redundancy elimination" of the protocol) without
    changing any node's decision.  The fingerprint is built from the raw
    scenario floats, which are bit-identical across nodes because every
    node fills its arrays from the same summary values.
    """
    entries = []
    for owner, deltas, flips, successes in scenario:
        entries.append((
            float(owner),
            tuple(float(v) for v in deltas),
            tuple(float(v) for v in flips),
            tuple(float(v) for v in successes),
        ))
    return (
        tuple(entries),
        int(num_reports),
        int(grid),
        float(flip_probability),
        float(success_probability),
        int(max_exact_reports),
        int(samples),
        int(rng_seed),
        int(max_rounds),
        int(candidate_budget),
        None if max_power is None else float(max_power),
        int(max_bits),
    )


def responsibility_consensus(
    results: Sequence[dict],
    *,
    responsibility_links: Sequence[tuple[int, int]] | None = None,
) -> bool:
    """Weak consensus: agreement restricted to the nodes' own duties.

    Not every UAV needs to detect every target.  A node's duty is the set of
    report links it supplies (its own ``uav_id`` rows) together with the
    targets it owns.  ``responsibility_consensus`` checks that every node
    agrees with all other nodes on the schedule restricted to the union of
    these links.  When the responsibility-consistent schedules are used by
    each node, conflicting assignments outside the duties never propagate to
    an executed action, so the weak agreement is sufficient for correct
    operation.

    With ``responsibility_links=None`` the duty set defaults to every link
    each node supplies: node ``i`` cares about ``(target, i)`` rows only,
    so the check is restricted to the per-node supplied rows.
    """
    if not results:
        return True
    link_mask: set[tuple[int, int]] | None = None
    if responsibility_links is not None:
        link_mask = set(responsibility_links)
    else:
        node_ids = [result["node_id"] for result in results]
        link_mask = {
            (q, node_id)
            for result in results
            for q in range(len(result["powers"]))
            for node_id in node_ids
            if node_id < len(result["powers"][q])
        }
    reference = results[0]
    for result in results[1:]:
        for (q, uav_id) in sorted(link_mask):
            if q >= len(reference["powers"]) or uav_id >= len(reference["powers"][q]):
                continue
            if not _link_equal(
                reference["powers"][q][uav_id],
                reference["bits"][q][uav_id],
                result["powers"][q][uav_id],
                result["bits"][q][uav_id],
            ):
                return False
    return True


def _link_equal(p_a, b_a, p_b, b_b) -> bool:
    return int(p_a) == int(p_b) and int(b_a) == int(b_b)


def nodes_from_scenario(
    scenario: Sequence,
    *,
    node_target_map: Sequence[int] | None = None,
    flip_probability: float = 0.0,
    success_probability: float = 1.0,
    num_nodes: int | None = None,
) -> list[NodeConfig]:
    """Build one autonomous node per UAV (default) or per specified target.

    Each node holds the summaries of the report links it can observe.  With
    the default ``node_target_map=None`` one node is created per target with
    the target's report summaries; with ``num_nodes`` greater than the
    target count, the extra nodes hold only their own UAV's summaries.
    """
    from .nomp_refinement import _parse_target

    summaries = summaries_from_scenario(
        scenario,
        flip_probability=flip_probability,
        success_probability=success_probability,
    )
    owners = [
        _parse_target(target, flip_probability, success_probability)[0]
        for target in scenario
    ]
    if node_target_map is not None:
        if len(node_target_map) != len(scenario):
            raise ValueError(
                "node_target_map must have one entry per target"
            )
        return [
            NodeConfig(
                node_id=q,
                target_id=node_target_map[q],
                owner_delta=float(owners[q]),
                reports=tuple(
                    summary for summary in summaries
                    if summary.target_id == q
                ),
                owners=tuple(enumerate(owners)),
            )
            for q in range(len(scenario))
        ]
    if num_nodes is None:
        num_nodes = max((summary.uav_id for summary in summaries), default=0) + 1
    per_target: list[list[ReportSummary]] = [
        [] for _ in range(len(scenario))
    ]
    for summary in summaries:
        per_target[summary.target_id].append(summary)
    nodes = []
    for node_id in range(num_nodes):
        targets_for_node = [
            q for q in range(len(scenario))
            if any(summary.uav_id == node_id for summary in per_target[q])
        ]
        if not targets_for_node:
            targets_for_node = [node_id % len(scenario)]
        node_summaries = tuple(
            summary
            for q in targets_for_node
            for summary in per_target[q]
            if summary.uav_id == node_id
        )
        nodes.append(NodeConfig(
            node_id=node_id,
            target_id=targets_for_node[0],
            owner_delta=float(owners[targets_for_node[0]]),
            reports=node_summaries,
            owners=tuple(enumerate(owners)),
        ))
    return nodes