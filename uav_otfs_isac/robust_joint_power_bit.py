"""Robust joint sensing-power and communication-bit allocation.

Each ``(power, bit)`` option is evaluated both at the clean communication
point and at the worst endpoint ``(flip_hi, success_lo)``.  By Lemma 4.70,
the endpoint is the worst case over the communication ambiguity rectangle,
so the exact DP over robust options is the exact worst-case allocation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .fusion import optimal_gaussian_detection_probability
from .joint_allocation import model_from_bits


@dataclass(frozen=True)
class JointPowerBitOption:
    cost_bits: int
    powers: tuple[float, ...]
    bits: tuple[int, ...]
    clean_pd: float
    robust_pd: float


def _target_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probability: float,
    success_probability: float,
    grid: int,
) -> float:
    full_deltas = np.concatenate((
        [float(owner_delta)],
        np.asarray(report_deltas, dtype=float)
        * np.sqrt(np.maximum(np.asarray(powers, dtype=float), 0.0)),
    ))
    full_bits = np.concatenate((
        [0],
        np.asarray(bits, dtype=int),
    ))
    model = model_from_bits(
        full_deltas, full_bits, bit_flip_probability=float(flip_probability)
    )
    model = dataclasses_replace_success(
        model, report_deltas.size, float(success_probability)
    )
    return float(optimal_gaussian_detection_probability(
        model.mu0, model.mu1, model.sigma0, model.sigma1,
        set(range(model.num_uavs)), 0.05, grid=grid,
    ))


def communication_target_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probability: float,
    success_probability: float,
    grid: int = 32,
) -> float:
    """P_D of one target with a shared communication channel state."""
    return _target_pd(
        owner_delta,
        report_deltas,
        powers,
        bits,
        flip_probability,
        success_probability,
        grid,
    )


def dataclasses_replace_success(model, report_count: int, success: float):
    from dataclasses import replace
    return replace(
        model,
        success_prob=np.array([1.0] + [success] * report_count),
    )


def enumerate_robust_power_bit_options(
    owner_delta: float,
    report_deltas: np.ndarray,
    *,
    power_levels: np.ndarray,
    bit_options: np.ndarray,
    budget: int,
    flip_interval: tuple[float, float],
    success_interval: tuple[float, float],
    power_cost: float = 1.0,
    bit_cost: float = 1.0,
    grid: int = 32,
) -> list[JointPowerBitOption]:
    """Enumerate joint options with clean and worst-endpoint P_D."""
    deltas = np.asarray(report_deltas, dtype=float)
    flip_lo, flip_hi = flip_interval
    success_lo, success_hi = success_interval
    per_report_choices = list(itertools.product(power_levels, bit_options))
    options = []
    for combo in itertools.product(
        per_report_choices, repeat=deltas.size
    ):
        powers = np.asarray([item[0] for item in combo], dtype=float)
        bits = np.asarray([item[1] for item in combo], dtype=int)
        cost = int(round(
            power_cost * float(powers.sum()) + bit_cost * float(bits.sum())
        ))
        if cost > budget:
            continue
        clean_pd = _target_pd(
            owner_delta, deltas, powers, bits, flip_lo, success_hi, grid
        )
        robust_pd = _target_pd(
            owner_delta, deltas, powers, bits, flip_hi, success_lo, grid
        )
        options.append(JointPowerBitOption(
            cost_bits=cost,
            powers=tuple(float(value) for value in powers),
            bits=tuple(int(value) for value in bits),
            clean_pd=float(clean_pd),
            robust_pd=float(robust_pd),
        ))
    return options


def pareto_options(
    options: list[JointPowerBitOption],
    value_field: str,
) -> list[tuple[int, float]]:
    result = []
    best_value = -1.0
    last_cost = None
    for option in sorted(options, key=lambda item: (item.cost_bits, -getattr(item, value_field))):
        value = float(getattr(option, value_field))
        if option.cost_bits == last_cost:
            continue
        last_cost = option.cost_bits
        if value > best_value + 1e-12:
            result.append((option.cost_bits, value))
            best_value = value
    return result
