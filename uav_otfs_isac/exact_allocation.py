"""Exact-array-factor water-filling for subarray aperture allocation.

The separable surrogate used by G13 ignores cross-block interference.  For a
phase profile built from target-aligned blocks, the exact squared array gain
toward target ``q`` is

``G_q(a) = | (1/N) sum_b sum_{n in block b}
             exp(j(phi_n - ideal_nq)) |^2``,

so the owner-only deflection surrogate including cross-block interference is

``D_q(a) = beta_q (1 + K0_q N^2 G_q(a))^2``,

where ``K0_q`` is the aperture-scale and geometry constant and ``G_q(a)`` is
computed by :func:`aperture_allocation_gains`.  The allocation is then
obtained by the same max-min water-filling iteration, but with the exact
objective instead of the separable approximation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ris_scenario import RisConfig
from .ris_subarray import aperture_allocation_gains


def exact_block_surrogate(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    allocation: Sequence[int],
    aperture_constants: Sequence[float],
    base_deflections: Sequence[float],
) -> np.ndarray:
    """Exact owner-only deflection surrogate including cross-block terms."""
    constants = np.asarray(aperture_constants, dtype=float)
    base = np.asarray(base_deflections, dtype=float)
    if constants.shape != (len(target_positions),) or base.shape != constants.shape:
        raise ValueError("constants and base deflections must match targets")
    gains = aperture_allocation_gains(config, target_positions, allocation)
    aperture_factor = float(config.num_elements**2)
    return base * (1.0 + constants * aperture_factor * gains) ** 2


def exact_waterfilling_allocation(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    aperture_constants: Sequence[float],
    base_deflections: Sequence[float],
) -> tuple[int, ...]:
    """Max-min water-filling on the exact array-factor surrogate."""
    count = len(target_positions)
    total_elements = config.num_elements
    quotient, remainder = divmod(total_elements, count)
    allocation = [quotient] * count
    for index in range(remainder):
        allocation[index] += 1

    def values(candidate):
        return exact_block_surrogate(
            config, target_positions, candidate,
            aperture_constants, base_deflections,
        )

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
