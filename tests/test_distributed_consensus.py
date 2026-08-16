import numpy as np

from uav_otfs_isac.distributed_consensus import (
    NodeConfig,
    ReportSummary,
    broadcast_round,
    nodes_from_scenario,
    run_broadcast_protocol,
    scenario_from_summaries,
    summaries_from_scenario,
)
from uav_otfs_isac.nomp_refinement import nomp_wta_greedy_joint_multi


def _scenario(q_count=3, reports=3, seed=0):
    rng = np.random.default_rng(seed)
    scenario = []
    for q in range(q_count):
        owner = float(rng.uniform(0.3, 1.2))
        deltas = rng.uniform(0.6, 2.0, reports)
        flips = rng.uniform(0.01, 0.05, reports)
        succ = rng.uniform(0.8, 0.98, reports)
        scenario.append((owner, deltas, flips, succ))
    return scenario


def test_summaries_roundtrip_preserves_scenario():
    scenario = _scenario(q_count=3, reports=3, seed=1)
    summaries = summaries_from_scenario(scenario)
    owners = list(enumerate(t[0] for t in scenario))
    rebuilt = scenario_from_summaries(summaries, owners)
    for original, reconstructed in zip(scenario, rebuilt):
        assert abs(original[0] - reconstructed[0]) < 1e-12
        np.testing.assert_allclose(original[1], reconstructed[1], atol=1e-12)
        np.testing.assert_allclose(original[2], reconstructed[2], atol=1e-12)
        np.testing.assert_allclose(original[3], reconstructed[3], atol=1e-12)


def test_complete_graph_consensus_matches_centralized():
    scenario = _scenario(q_count=4, reports=3, seed=2)
    budget = 8 * len(scenario)
    centralized = nomp_wta_greedy_joint_multi(
        scenario, budget, grid=32, max_rounds=30,
    )
    nodes = nodes_from_scenario(scenario)
    assert len(nodes) == 3  # UAV 数
    result = run_broadcast_protocol(
        nodes, budget, rounds=1, grid=32, max_rounds=30,
    )
    assert result["consensus"]
    # 每个节点的调度都与集中式一致 (节点持有相同摘要)
    reference = centralized["worst_pd"]
    for node_result in result["nodes"]:
        assert abs(node_result["worst_pd"] - reference) < 1e-12


def test_broadcast_round_union_on_complete_graph():
    node_a = NodeConfig(0, 0, 0.5, (ReportSummary(0, 0, 1.0, 0.02, 0.9),))
    node_b = NodeConfig(1, 0, 0.5, (ReportSummary(0, 1, 2.0, 0.03, 0.85),))
    after = broadcast_round([node_a, node_b])
    assert len(after[0].reports) == 2
    assert len(after[1].reports) == 2


def test_partial_topology_rounds_accumulate_knowledge():
    scenario = _scenario(q_count=2, reports=2, seed=3)
    nodes = nodes_from_scenario(scenario, num_nodes=2)
    adjacency = [[1], [0]]  # 每个节点只有一跳邻居
    one_round = broadcast_round(nodes, adjacency)
    two_rounds = broadcast_round(one_round, adjacency)
    # 两轮后每个节点都学到全部摘要
    total = len(set(nodes[0].reports) | set(nodes[1].reports))
    assert len(two_rounds[0].reports) == total
    assert len(two_rounds[1].reports) == total


def test_partial_topology_still_consensus_if_all_learn_all():
    scenario = _scenario(q_count=2, reports=3, seed=4)
    budget = 8 * len(scenario)
    nodes = nodes_from_scenario(scenario, num_nodes=2)
    adjacency = [[1], [0]]
    result = run_broadcast_protocol(
        nodes, budget, rounds=2, adjacency=adjacency,
        grid=32, max_rounds=30,
    )
    assert result["consensus"]
    assert result["topology"] == "partial"


def test_message_count_is_linear_in_nodes():
    scenario = _scenario(q_count=3, reports=2, seed=5)
    nodes = nodes_from_scenario(scenario, num_nodes=3)
    result = run_broadcast_protocol(nodes, 16, rounds=1, grid=16)
    # 完全图一轮: 每节点把它持有的摘要发给其余 2 个节点
    held = sum(len(node.reports) for node in nodes)
    expected = held * (3 - 1)
    assert result["message_count"] == expected