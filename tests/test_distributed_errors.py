import numpy as np

from uav_otfs_isac.distributed_consensus import (
    NodeConfig,
    ReportSummary,
    broadcast_round,
    nodes_from_scenario,
    run_broadcast_protocol,
)


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


def test_error_free_broadcast_unchanged():
    scenario = _scenario(q_count=3, reports=2, seed=5)
    nodes = nodes_from_scenario(scenario, num_nodes=3)
    clean = broadcast_round(nodes)
    with_errors = broadcast_round(
        nodes, drop_probability=0.0, flip_probability=0.0, rng=np.random.default_rng(1),
    )
    assert clean[0].reports == with_errors[0].reports


def test_drop_probability_reduces_knowledge():
    scenario = _scenario(q_count=3, reports=2, seed=5)
    nodes = nodes_from_scenario(scenario, num_nodes=3)
    full_size = len(broadcast_round(nodes)[0].reports)
    heavy = broadcast_round(
        nodes, drop_probability=0.5, rng=np.random.default_rng(3),
    )
    assert all(len(node.reports) < full_size for node in heavy)


def test_drop_consensus_degrades_then_recovers_with_rounds():
    scenario = _scenario(q_count=4, reports=3, seed=2)
    nodes = nodes_from_scenario(scenario)
    budget = 8 * 4
    one_round = run_broadcast_protocol(
        nodes, budget, rounds=1, grid=16, max_rounds=10,
        drop_probability=0.2, rng_seed=0,
    )
    assert not one_round["consensus"]
    # 丢包只减知识集，最差节点 worst_pd 退化有限
    three_rounds = run_broadcast_protocol(
        nodes, budget, rounds=3, grid=16, max_rounds=10,
        drop_probability=0.05, rng_seed=0,
    )
    assert three_rounds["consensus"]


def test_flip_probability_corrupts_deltas():
    scenario = _scenario(q_count=3, reports=2, seed=5)
    nodes = nodes_from_scenario(scenario, num_nodes=3)
    clean = broadcast_round(nodes)
    corrupted = broadcast_round(
        nodes, flip_probability=1.0, rng=np.random.default_rng(0),
    )
    # 全翻转后至少一个节点的摘要 delta 与干净不同
    clean_deltas = {s.uav_id: s.delta for s in clean[0].reports}
    corrupt_deltas = {s.uav_id: s.delta for s in corrupted[0].reports}
    assert any(
        corrupt_deltas.get(i) != clean_deltas.get(i)
        for i in set(clean_deltas) | set(corrupt_deltas)
    )


def test_flip_consensus_degrades_with_redundancy_limited():
    scenario = _scenario(q_count=4, reports=3, seed=2)
    nodes = nodes_from_scenario(scenario)
    budget = 8 * 4
    result = run_broadcast_protocol(
        nodes, budget, rounds=3, grid=16, max_rounds=10,
        flip_probability=0.2, rng_seed=0,
    )
    # 损坏数据（拜占庭式）无法靠冗余轮次修复
    assert not result["consensus"]


def test_num_reports_inference_survives_total_drop():
    scenario = _scenario(q_count=3, reports=2, seed=5)
    nodes = nodes_from_scenario(scenario, num_nodes=3)
    budget = 8 * 3
    result = run_broadcast_protocol(
        nodes, budget, rounds=1, grid=16, max_rounds=10,
        drop_probability=0.95, rng_seed=0,
    )
    # 不崩溃且所有节点仍返回 2 报告行
    for node_result in result["nodes"]:
        assert len(node_result["powers"]) == 3
        assert all(len(row) == 2 for row in node_result["powers"])