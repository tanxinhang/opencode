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
from .joint_allocation import moments


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
    return _expected_communication_pd(
        owner_delta,
        report_deltas,
        powers,
        bits,
        np.full(len(report_deltas), float(flip_probability)),
        np.full(len(report_deltas), float(success_probability)),
        grid,
    )


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


def _expected_communication_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probabilities: np.ndarray,
    success_probabilities: np.ndarray,
    grid: int,
) -> float:
    """Expected P_D over independent report erasures."""
    deltas = np.asarray(report_deltas, dtype=float)
    powers = np.asarray(powers, dtype=float)
    bits = np.asarray(bits, dtype=int)
    flips = np.asarray(flip_probabilities, dtype=float)
    successes = np.asarray(success_probabilities, dtype=float)
    entries = []
    for i in range(deltas.size):
        if bits[i] <= 0:
            continue
        scaled = float(deltas[i]) * np.sqrt(max(float(powers[i]), 0.0))
        m0, m1, v0, v1 = moments(scaled, int(bits[i]), float(flips[i]))
        entries.append((m0, m1, v0, v1, float(successes[i])))
    total = 0.0
    for mask in range(1 << len(entries)):
        probability = 1.0
        mu0 = [0.0]
        mu1 = [float(owner_delta)]
        var0 = [1.0]
        var1 = [1.0]
        for j, (m0, m1, v0, v1, success) in enumerate(entries):
            if mask >> j & 1:
                probability *= success
                mu0.append(m0)
                mu1.append(m1)
                var0.append(v0)
                var1.append(v1)
            else:
                probability *= 1.0 - success
        if probability <= 0.0:
            continue
        pd = optimal_gaussian_detection_probability(
            np.asarray(mu0),
            np.asarray(mu1),
            np.diag(var0),
            np.diag(var1),
            set(range(len(mu0))),
            0.05,
            grid=grid,
        )
        total += probability * float(pd)
    return float(total)


def per_report_communication_target_pd(
    owner_delta: float,
    report_deltas: np.ndarray,
    powers: np.ndarray,
    bits: np.ndarray,
    flip_probabilities: np.ndarray,
    success_probabilities: np.ndarray,
    grid: int = 32,
) -> float:
    """Expected P_D with per-report BSC flip and erasure."""
    return _expected_communication_pd(
        owner_delta,
        report_deltas,
        powers,
        bits,
        np.asarray(flip_probabilities, dtype=float),
        np.asarray(success_probabilities, dtype=float),
        grid,
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


def enumerate_heterogeneous_robust_power_bit_options(
    owner_delta: float,
    report_deltas: np.ndarray,
    flip_intervals: list[tuple[float, float]],
    success_intervals: list[tuple[float, float]],
    *,
    power_levels: np.ndarray,
    bit_options: np.ndarray,
    budget: int,
    power_cost: float = 1.0,
    bit_cost: float = 1.0,
    grid: int = 32,
) -> list[JointPowerBitOption]:
    """Enumerate joint options with per-report communication intervals."""
    deltas = np.asarray(report_deltas, dtype=float)
    flip_lo = np.asarray([item[0] for item in flip_intervals], dtype=float)
    flip_hi = np.asarray([item[1] for item in flip_intervals], dtype=float)
    success_lo = np.asarray(
        [item[0] for item in success_intervals], dtype=float
    )
    success_hi = np.asarray(
        [item[1] for item in success_intervals], dtype=float
    )
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
        clean_pd = per_report_communication_target_pd(
            owner_delta,
            deltas,
            powers,
            bits,
            flip_lo,
            success_hi,
            grid,
        )
        robust_pd = per_report_communication_target_pd(
            owner_delta,
            deltas,
            powers,
            bits,
            flip_hi,
            success_lo,
            grid,
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
