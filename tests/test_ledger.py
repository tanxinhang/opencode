"""P4 unified ledger accountant tests (advice/011 section 6)."""

import numpy as np
import pytest

from uav_otfs_isac.config import load_config
from uav_otfs_isac.ledger import (
    report_budget_from_total,
    report_cost_bits,
    ris_control_overhead_bits,
    scheduled_report_bits,
)
from uav_otfs_isac.scenario import build_models


def test_report_cost_bits_matches_model_ledger():
    cfg = load_config("config/demo.yaml")
    model = build_models(cfg, np.random.default_rng(cfg.seed))[0]
    assert model.report_bits.shape == (model.num_uavs,)
    # owner entry free; every non-owner is payload + 2
    assert report_cost_bits(model, model.owner) == 0
    for uav in range(model.num_uavs):
        if uav == model.owner:
            continue
        assert report_cost_bits(model, uav) == int(model.report_bits[uav])
        assert report_cost_bits(model, uav) == cfg.quantizer_bits + 2


def test_scheduled_report_bits_sums_non_owner_only():
    cfg = load_config("config/demo.yaml")
    model = build_models(cfg, np.random.default_rng(cfg.seed))[0]
    scheduled = {u for u in range(model.num_uavs) if u != model.owner}
    expected = sum(int(model.report_bits[u]) for u in scheduled)
    assert scheduled_report_bits(model, scheduled) == expected
    assert scheduled_report_bits(model, {model.owner}) == 0


def test_ris_control_overhead_bits():
    assert ris_control_overhead_bits(16, None, 100) == 0.0
    assert ris_control_overhead_bits(16, 2, 100) == 0.32
    assert ris_control_overhead_bits(128, 3, 64) == 6.0


def test_report_budget_from_total_keeps_fractional_overhead():
    # total 40 with a fractional control overhead 0.32 must NOT truncate
    # the fraction out of the ledger: B_total = report + overhead + residual
    total = 40.0
    report, residual, overhead = report_budget_from_total(total, 0.32)
    assert report == 39
    assert overhead == 0.32
    assert abs(total - (report + overhead + residual)) < 1e-12
    assert residual == pytest.approx(0.68)  # the untruncated fractional part


def test_report_budget_from_total_rejects_overhead_overflow():
    from uav_otfs_isac.ledger import report_budget_from_total
    with pytest.raises(ValueError):
        report_budget_from_total(10.0, 12.0)