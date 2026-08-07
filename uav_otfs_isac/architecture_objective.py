"""Derived architecture objective for RIS aperture design.

The design variables enter through a physical objective, not through a blind
search.  Under the subarray approximation, the normalized array gain of a
block of ``a_q`` elements toward target ``q`` is ``a_q / N``, so the RIS-to-
direct path-power ratio is

``K_q a_q^2 sinc^2(1/2^b)``.

The evidence SNR is multiplied by ``1 + K_q a_q^2 sinc^2(1/2^b)``, and the
local deflection scales with the square of the evidence SNR.  For an equal
aperture allocation ``a_q = N/3`` the weak-target deflection surrogate is

``J(N) = beta (1 + kappa N^2)^2 (R - L N)``,

with ``kappa = K_weak sinc^2(1/2^b) / 9``, ``L = b / C`` and
``R = B_total``.  The first-order condition is the quadratic

``5 kappa L N^2 - 4 kappa R N + L = 0``,

so the optimal aperture has a closed form.  The same quadratic explains why
``N``, ``b``, and ``C`` matter jointly: ``N`` controls the quartic deflection
gain, while ``b/C`` controls the linear control-overhead loss.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ris_scenario import RisConfig, ris_quantized_gain_loss


def aperture_constants(
    config: RisConfig,
    transmitter_positions: Sequence[Sequence[float]],
    target_positions: Sequence[Sequence[float]],
    receiver_position: Sequence[float],
    aperture_scale: float,
    direct_blockage: float = 0.01,
) -> np.ndarray:
    """Per-target constants ``K_q = P_ris / P_dir / a_q^2``."""
    transmitters = [np.asarray(position, dtype=float) for position in transmitter_positions]
    targets = [np.asarray(position, dtype=float) for position in target_positions]
    receiver = np.asarray(receiver_position, dtype=float)
    result = np.zeros(len(targets), dtype=float)
    for q, target in enumerate(targets):
        values = []
        for transmitter in transmitters:
            tx_target = float(np.linalg.norm(transmitter - target))
            target_rx = float(np.linalg.norm(target - receiver))
            tx_ris = float(np.linalg.norm(transmitter - config.position))
            ris_target = float(np.linalg.norm(config.position - target))
            direct = 1.0 / (tx_target**2 * target_rx**2)
            if config.weak_target_id == q:
                direct *= direct_blockage
            ris_power = aperture_scale / (
                tx_ris**2 * ris_target**2 * target_rx**2
            )
            values.append(ris_power / max(direct, 1e-30))
        result[q] = float(np.mean(values))
    return result


def deflection_surrogate(
    aperture_allocation: Sequence[int],
    phase_bits: int,
    aperture_constants: Sequence[float],
    base_deflections: Sequence[float],
) -> np.ndarray:
    """Owner-only deflection surrogate ``beta (1 + K a^2 g_q)^2``."""
    allocation = np.asarray(aperture_allocation, dtype=float)
    constants = np.asarray(aperture_constants, dtype=float)
    base = np.asarray(base_deflections, dtype=float)
    quantization = ris_quantized_gain_loss(phase_bits)
    return base * (1.0 + constants * allocation**2 * quantization) ** 2


def waterfilling_allocation(
    total_elements: int,
    aperture_constants: Sequence[float],
    base_deflections: Sequence[float],
) -> tuple[int, ...]:
    """Max-min deflection-equalizing aperture allocation.

    The surrogate is ``D_q(a_q) = beta_q (1 + kappa_q a_q^2)^2`` with
    monotone increasing convex terms.  The max-min allocation is reached by
    moving aperture from the currently largest-D target to the currently
    smallest-D target while this strictly increases the minimum.  The step
    size halves from a coarse value to one, yielding a deterministic
    water-filling fixed point without enumerating allocations.
    """
    constants = np.asarray(aperture_constants, dtype=float)
    base = np.asarray(base_deflections, dtype=float)
    count = constants.size
    if base.shape != (count,):
        raise ValueError("one base deflection is required per target")
    if np.any(constants <= 0.0) or np.any(base <= 0.0):
        raise ValueError("constants and base deflections must be positive")
    if total_elements <= 0:
        raise ValueError("total_elements must be positive")

    quotient, remainder = divmod(total_elements, count)
    allocation = [quotient] * count
    for index in range(remainder):
        allocation[index] += 1

    def values(candidate):
        candidate = np.asarray(candidate, dtype=float)
        return base * (1.0 + constants * candidate**2) ** 2

    step = max(1, total_elements // 32)
    while step >= 1:
        improved = True
        while improved:
            improved = False
            current = values(allocation)
            source = int(np.argmax(current))
            target = int(np.argmin(current))
            if source == target or allocation[source] < step:
                break
            trial = list(allocation)
            trial[source] -= step
            trial[target] += step
            trial_values = values(trial)
            if float(np.min(trial_values)) > float(np.min(current)) + 1e-12:
                allocation = trial
                improved = True
        step //= 2
    return tuple(allocation)


def optimal_aperture_formula(
    total_budget: float,
    phase_bits: int,
    coherence_frames: int,
    kappa: float,
) -> float | None:
    """Larger root of the first-order quadratic, or None if no finite optimum."""
    if total_budget <= 0.0:
        raise ValueError("total_budget must be positive")
    if phase_bits <= 0:
        raise ValueError("phase_bits must be positive")
    if coherence_frames <= 0:
        raise ValueError("coherence_frames must be positive")
    if kappa <= 0.0:
        return None
    rate = phase_bits / coherence_frames
    discriminant = (
        16.0 * kappa * kappa * total_budget * total_budget
        - 20.0 * kappa * rate * rate
    )
    if discriminant < 0.0:
        return None
    root = 4.0 * kappa * total_budget + np.sqrt(discriminant)
    return float(root / (10.0 * kappa * rate))


def derived_surrogate_objective(
    aperture: float,
    total_budget: float,
    phase_bits: int,
    coherence_frames: int,
    kappa: float,
    base_deflection: float = 1.0,
) -> float:
    """Weak-target surrogate ``beta (1 + kappa N^2)^2 (R - LN)``."""
    rate = phase_bits / coherence_frames
    if aperture * rate >= total_budget:
        return -np.inf
    return float(
        base_deflection
        * (1.0 + kappa * aperture * aperture) ** 2
        * (total_budget - rate * aperture)
    )
