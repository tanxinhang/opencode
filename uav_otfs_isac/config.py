from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OTFSConfig:
    doppler_bins: int = 16
    delay_bins: int = 32
    accumulation: int = 8
    snr_db_range: tuple[float, float] = (-7.0, 5.0)
    fractional_doppler_range: tuple[float, float] = (-0.48, 0.48)
    residual_interference: float = 0.12
    common_factor_strength: float = 0.30


@dataclass(frozen=True)
class ReportingConfig:
    success_probability_range: tuple[float, float] = (0.72, 0.98)
    bit_flip_probability_range: tuple[float, float] = (0.005, 0.08)
    calibration_std: float = 0.05


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260802
    num_uavs: int = 8
    num_targets: int = 3
    owners: tuple[int, ...] = (0, 3, 6)
    target_present: tuple[bool, ...] = (True, True, True)
    quantizer_bits: int = 3
    report_budget_bits: int = 42
    false_alarm_rate: float = 0.05
    qos_min_deflection: tuple[float, ...] = (3.0, 3.0, 3.0)
    qos_weights: tuple[float, ...] = (1.0, 1.0, 1.3)
    performance_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    monte_carlo_trials: int = 5000
    expected_mode: str = "exact"
    max_exact_reports: int = 14
    covariance_shrinkage: float = 0.08
    covariance_epsilon: float = 1e-8
    otfs: OTFSConfig = field(default_factory=OTFSConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def validate(self) -> None:
        if len(self.owners) != self.num_targets:
            raise ValueError("owners must contain one fusion node per target")
        if any(x < 0 or x >= self.num_uavs for x in self.owners):
            raise ValueError("owner index is outside the UAV set")
        for values, name in [
            (self.qos_min_deflection, "qos_min_deflection"),
            (self.qos_weights, "qos_weights"),
            (self.performance_weights, "performance_weights"),
        ]:
            if len(values) != self.num_targets:
                raise ValueError(f"{name} must have num_targets entries")
        if not 0 < self.false_alarm_rate < 1:
            raise ValueError("false_alarm_rate must be in (0, 1)")
        if self.quantizer_bits < 1:
            raise ValueError("quantizer_bits must be positive")


def _tupleize(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    otfs_raw = {k: _tupleize(v) for k, v in raw.pop("otfs", {}).items()}
    reporting_raw = {k: _tupleize(v) for k, v in raw.pop("reporting", {}).items()}
    root = {k: _tupleize(v) for k, v in raw.items()}
    cfg = ExperimentConfig(
        **root,
        otfs=OTFSConfig(**otfs_raw),
        reporting=ReportingConfig(**reporting_raw),
    )
    cfg.validate()
    return cfg

