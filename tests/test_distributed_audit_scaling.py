"""Gate F0-S tests: K/Q scaling audit (advice/006)."""

import json

import numpy as np
import pytest

from scripts.run_distributed_audit_scaling import (
    SCALES,
    measure_decision_cost,
    run_scaling_audit,
)


@pytest.fixture(scope="module")
def audit():
    rows = run_scaling_audit(
        scales=((6, 3), (8, 4)), n_runs=40, seeds=2,
        calib_verify_runs=500,
    )
    return rows


def test_scaling_rows_structure(audit):
    verdicts, rows = audit
    assert set(rows) == {"6_3", "8_4"}
    for key, row in rows.items():
        assert row["k"] == 2 * row["q"]
        assert set(row["modes"]) == {
            "centralized", "full_message", "compact_token", "local_only"}
        for mode, m in row["modes"].items():
            assert 0.0 < m["J"] <= 40.0
            assert 0.0 <= m["p_md_max"] <= 1.0
            assert 0.0 <= m["p_fa_max"] <= 1.0
        assert row["u2u_bits_per_uav_per_cycle"]["transmit"] == 19.0
        assert row["u2u_bits_per_uav_per_cycle"]["receive_full_mesh"] \
            == 19.0 * (row["k"] - 1)
        assert row["decision_us_per_uav"] > 0.0


def test_decision_cost_grows_gradually(audit):
    verdicts, rows = audit
    dec = [rows[key]["decision_us_per_uav"] for key in rows]
    # sub-linear-in-Q growth is required; at these tiny runs only the
    # ordering sanity is checked (formal bound in gate_b)
    assert dec[1] < dec[0] * (4.0 / 3.0) * 3.5


def test_verdict_keys_and_frozen_params(audit):
    verdicts, rows = audit
    for gate in ("gate_a_detection_scales", "gate_b_local_compute",
                 "gate_c_communication", "first_bottleneck"):
        assert gate in verdicts
    assert isinstance(verdicts["gate_a_detection_scales"]["passed"], bool)
    assert isinstance(verdicts["gate_b_local_compute"]["passed"], bool)
    assert verdicts["gate_c_communication"]["transmit_constant"] is True
    assert verdicts["gate_c_communication"]["receive_linear_in_k"] is True
    assert "target allocation" in verdicts["first_bottleneck"] or \
        "scales (Gate A passed)" in verdicts["first_bottleneck"]


def test_scaling_gate_deterministic():
    rows1 = run_scaling_audit(scales=((6, 3),), n_runs=40, seeds=2,
                              calib_verify_runs=500)
    rows2 = run_scaling_audit(scales=((6, 3),), n_runs=40, seeds=2,
                              calib_verify_runs=500)
    assert rows1[1]["6_3"]["modes"]["compact_token"]["J"] \
        == rows2[1]["6_3"]["modes"]["compact_token"]["J"]
