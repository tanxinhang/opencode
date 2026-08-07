"""Aperture-conserved multi-beam RIS with discrete aperture-gradient search.

A single shared phase profile can only point one ULA beam.  To serve several
targets simultaneously without per-target time multiplexing, the aperture is
partitioned into disjoint subarrays and every subarray is steered to one
target.  The total number of elements is conserved, so the optimization is a
zero-sum allocation of aperture between targets.

For target ``q`` and a phase profile composed of blocks ``b``, the squared
array gain is

``G_q = | (1/N) sum_b sum_{n in block b} exp(j(phi_n - ideal_nq)) |^2``.

The self-block contribution of an aligned block is ``N_b / N``, so moving
aperture from one target to another changes every target's gain through both
the self-block term and cross-block interference.  The module exposes the
exact phase construction and a discrete coordinate-ascent search over the
integer aperture allocation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .ris_optimization import (
    ris_beam_phase_from_cosine,
    target_direction_cosines,
)
from .ris_scenario import RisConfig, ris_array_gain, ris_beam_phase


def multi_beam_phase(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    allocation: Sequence[int],
    *,
    steering_cosines: Sequence[float] | None = None,
) -> np.ndarray:
    """Build one phase vector from disjoint target-aligned subarrays."""
    allocation = [int(value) for value in allocation]
    if len(allocation) != len(target_positions):
        raise ValueError("one allocation entry is required per target")
    if any(value < 0 for value in allocation):
        raise ValueError("allocation entries must be nonnegative")
    if sum(allocation) != config.num_elements:
        raise ValueError("allocation must sum to num_elements")
    if steering_cosines is None:
        block_phases = [
            ris_beam_phase(target, config) for target in target_positions
        ]
    else:
        steering_cosines = [float(value) for value in steering_cosines]
        if len(steering_cosines) != len(target_positions):
            raise ValueError("one steering cosine is required per target block")
        block_phases = [
            ris_beam_phase_from_cosine(config, cosine)
            for cosine in steering_cosines
        ]
    phase = np.zeros(config.num_elements, dtype=float)
    start = 0
    for block_phase, size in zip(block_phases, allocation):
        if size == 0:
            continue
        phase[start:start + size] = block_phase[start:start + size]
        start += size
    return phase


def aperture_allocation_gains(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    allocation: Sequence[int],
) -> np.ndarray:
    """Exact squared array gains of a subarray phase profile."""
    phase = multi_beam_phase(config, target_positions, allocation)
    return np.asarray([
        ris_array_gain(phase, target, config) ** 2
        for target in target_positions
    ])


def coordinate_aperture_ascent(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    objective: Callable[[tuple[int, ...]], float],
    *,
    step_sizes: Sequence[int] = (32, 16, 8),
    max_rounds_per_step: int = 4,
    initial_allocation: Sequence[int] | None = None,
) -> dict:
    """Coordinate ascent over integer aperture allocations.

    Each round evaluates all moves that transfer ``step`` elements from one
    target block to another, accepts the best feasible improvement, and stops
    when no move improves the objective.  The step size is reduced through
    ``step_sizes`` so the search first finds the coarse regime and then
    refines locally.
    """
    count = len(target_positions)
    if initial_allocation is None:
        quotient, remainder = divmod(config.num_elements, count)
        allocation = np.asarray([quotient] * count, dtype=int)
        allocation[:remainder] += 1
        best_allocation = tuple(int(value) for value in allocation)
    else:
        initial = [int(value) for value in initial_allocation]
        if len(initial) != count:
            raise ValueError("initial_allocation must have one entry per target")
        if sum(initial) != config.num_elements:
            raise ValueError("initial_allocation must sum to num_elements")
        if any(value < 0 for value in initial):
            raise ValueError("initial_allocation entries must be nonnegative")
        best_allocation = tuple(initial)
    best_value = float(objective(best_allocation))
    history = [{
        "allocation": best_allocation,
        "value": best_value,
    }]
    for step in step_sizes:
        improved = True
        rounds = 0
        while improved and rounds < max_rounds_per_step:
            improved = False
            candidates = []
            for source in range(count):
                for target in range(count):
                    if source == target:
                        continue
                    trial = list(best_allocation)
                    if trial[source] < step:
                        continue
                    trial[source] -= step
                    trial[target] += step
                    trial_tuple = tuple(int(value) for value in trial)
                    value = float(objective(trial_tuple))
                    candidates.append((value, trial_tuple))
            if not candidates:
                break
            candidates.sort(reverse=True)
            value, trial = candidates[0]
            if value > best_value + 1e-12:
                best_value = value
                best_allocation = trial
                history.append({
                    "allocation": best_allocation,
                    "value": best_value,
                })
                improved = True
            rounds += 1
    return {
        "allocation": best_allocation,
        "value": best_value,
        "history": history,
        "target_cosines": target_direction_cosines(
            config, target_positions
        ).tolist(),
    }


def coordinate_block_steering_ascent(
    config: RisConfig,
    target_positions: Sequence[Sequence[float]],
    allocation: Sequence[int],
    objective: Callable[[tuple[float, ...]], float],
    *,
    step: float = 0.1,
    grid_points: int = 9,
    max_rounds: int = 3,
    initial_cosines: Sequence[float] | None = None,
) -> dict:
    """Coordinate ascent over per-subarray steering cosines.

    The aperture allocation is fixed, so total aperture and control overhead
    are conserved.  Each coordinate sweep perturbs one block steering cosine
    on a bounded grid around its current value and accepts the best feasible
    improvement.  The result is a local optimum over single-block steering
    changes for the supplied system objective.
    """
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    if step <= 0.0:
        raise ValueError("step must be positive")
    count = len(target_positions)
    if initial_cosines is None:
        cosines = target_direction_cosines(config, target_positions).tolist()
    else:
        cosines = [float(value) for value in initial_cosines]
        if len(cosines) != count:
            raise ValueError("one initial cosine is required per block")
    best_value = float(objective(tuple(cosines)))
    history = [{
        "steering_cosines": list(cosines),
        "value": best_value,
    }]
    offsets = np.linspace(-step, step, grid_points)
    for _ in range(max_rounds):
        improved = False
        for block in range(count):
            best_cosine = cosines[block]
            best_local = best_value
            for offset in offsets:
                candidate = float(np.clip(cosines[block] + offset, -1.0, 1.0))
                trial = list(cosines)
                trial[block] = candidate
                value = float(objective(tuple(trial)))
                if value > best_local + 1e-12:
                    best_local = value
                    best_cosine = candidate
            if best_cosine != cosines[block]:
                cosines[block] = best_cosine
                best_value = best_local
                history.append({
                    "steering_cosines": list(cosines),
                    "value": best_value,
                })
                improved = True
        if not improved:
            break
    return {
        "allocation": [int(value) for value in allocation],
        "steering_cosines": cosines,
        "value": best_value,
        "history": history,
    }


def exact_single_move_gradients(
    objective: Callable[[tuple[int, ...]], float],
    allocation: Sequence[int],
) -> dict:
    """Exact one-element transfer gradients of an allocation objective.

    The function evaluates ``F(a + e_q - e_r) - F(a)`` for every ordered pair
    of distinct blocks ``(r -> q)`` with ``a_r > 0``.  If the maximum is
    nonpositive, the allocation is locally optimal with respect to
    single-element transfers of the exact objective.
    """
    allocation = tuple(int(value) for value in allocation)
    current = float(objective(allocation))
    moves = []
    for source in range(len(allocation)):
        if allocation[source] == 0:
            continue
        for target in range(len(allocation)):
            if target == source:
                continue
            trial = list(allocation)
            trial[source] -= 1
            trial[target] += 1
            value = float(objective(tuple(trial)))
            moves.append({
                "source": source,
                "target": target,
                "value": value,
                "gradient": value - current,
            })
    moves.sort(key=lambda move: move["gradient"], reverse=True)
    maximum = float(max((move["gradient"] for move in moves), default=0.0))
    return {
        "value": current,
        "maximum_gradient": maximum,
        "local_optimal": maximum <= 1e-9,
        "moves": moves,
    }


def bounded_multi_move_certificate(
    objective: Callable[[tuple[int, ...]], float],
    allocation: Sequence[int],
    *,
    max_transfer: int = 3,
) -> dict:
    """Exact local certificate over a bounded multi-block transfer set.

    Every integer net-change vector ``n`` with ``sum n = 0`` and
    ``sum max(n, 0) <= max_transfer`` is evaluated exactly.  If no trial
    improves ``F``, the allocation is locally optimal with respect to all
    multi-block moves moving at most ``max_transfer`` elements in total.
    """
    allocation = tuple(int(value) for value in allocation)
    count = len(allocation)
    if max_transfer <= 0:
        raise ValueError("max_transfer must be positive")
    current = float(objective(allocation))
    best_allocation = allocation
    best_value = current
    evaluated = 0
    for net in _net_change_vectors(count, max_transfer):
        if all(value == 0 for value in net):
            continue
        trial = [allocation[q] + net[q] for q in range(count)]
        if any(value < 0 for value in trial):
            continue
        value = float(objective(tuple(trial)))
        evaluated += 1
        if value > best_value + 1e-12:
            best_value = value
            best_allocation = tuple(trial)
    return {
        "value": best_value,
        "allocation": best_allocation,
        "improved": best_value > current + 1e-12,
        "local_optimal": best_value <= current + 1e-12,
        "evaluated": evaluated,
        "max_transfer": max_transfer,
    }


def _net_change_vectors(count: int, max_transfer: int):
    """All zero-sum net vectors with positive mass at most max_transfer."""
    if count > 4:
        raise ValueError("bounded multi-move enumeration supports up to 4 blocks")
    grid = range(-max_transfer, max_transfer + 1)
    for values in np.ndindex(*([2 * max_transfer + 1] * count)):
        net = [grid[index] for index in values]
        if sum(net) != 0:
            continue
        if sum(max(value, 0) for value in net) > max_transfer:
            continue
        yield net
