import numpy as np

from uav_otfs_isac.distributed_consensus import (
    _checksum_of,
    nodes_from_scenario,
    responsibility_consensus,
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


def test_checksum_detects_corruption():
    assert _checksum_of(1.234567) == _checksum_of(1.234567)
    assert _checksum_of(1.234567) != _checksum_of(-0.25)


def test_self_check_discards_corrupted_summaries():
    scenario = _scenario(q_count=3, reports=2, seed=5)
    nodes = nodes_from_scenario(scenario, num_nodes=3)
    unchecked = run_broadcast_protocol(
        nodes, 16, rounds=3, grid=16, max_rounds=10,
        flip_probability=0.5, rng_seed=0,
    )
    checked = run_broadcast_protocol(
        nodes, 16, rounds=3, grid=16, max_rounds=10,
        flip_probability=0.5, self_check=True, rng_seed=0,
    )
    # 自校验丢弃损坏摘要后, 各节点知识集只含真值摘要
    # 所有自校验节点的摘要 delta 均在真值集合内 (损坏值 -0.5..0.5 被丢弃)
    true_deltas = {
        round(float(s.delta), 6)
        for node in nodes
        for s in node.reports
    }
    for node_result in checked["nodes"]:
        for target in node_result["scenario"]:
            for value in target[1]:
                assert round(float(value), 6) in true_deltas


def test_self_check_recovers_flip_consensus():
    scenario = _scenario(q_count=4, reports=3, seed=2)
    nodes = nodes_from_scenario(scenario)
    budget = 8 * 4
    unchecked = run_broadcast_protocol(
        nodes, budget, rounds=3, grid=16, max_rounds=10,
        flip_probability=0.2, rng_seed=0,
    )
    checked = run_broadcast_protocol(
        nodes, budget, rounds=3, grid=16, max_rounds=10,
        flip_probability=0.2, self_check=True, rng_seed=0,
    )
    assert not unchecked["consensus"]
    assert checked["consensus"]


def test_responsibility_consensus_trivially_true_for_one_node():
    scenario = _scenario(q_count=2, reports=2, seed=1)
    nodes = nodes_from_scenario(scenario, num_nodes=2)
    result = run_broadcast_protocol(nodes, 12, rounds=1, grid=16)
    single = [result["nodes"][0]]
    assert responsibility_consensus(single)


def test_responsibility_consensus_implies_less_than_full():
    # 职责只覆盖每节点供给的行: 一个节点与参考在职责行不一致 -> False
    scenario = _scenario(q_count=2, reports=2, seed=1)
    nodes = nodes_from_scenario(scenario, num_nodes=2)
    result = run_broadcast_protocol(nodes, 12, rounds=1, grid=16)
    reference = result["nodes"][0]
    other = result["nodes"][1]
    # 篡改另一节点的职责行, 职责共识必须变 False
    tampered = {
        "node_id": other["node_id"],
        "powers": [row.copy() for row in other["powers"]],
        "bits": [row.copy() for row in other["bits"]],
    }
    tampered["powers"][0][0] = (tampered["powers"][0][0] + 1) % 2
    assert not responsibility_consensus([reference, tampered])