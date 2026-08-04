"""Physics-constrained local model-order selection for multistatic targets.

Within a density-connected collision component, candidates are explained by
an unknown number of target states plus a clutter state. Assignment enforces
at most one path from a transmitter to a target. A calibrated cross-UAV test
admits the local order; a penalized physical likelihood fits that order.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .multistatic_association import (
    PathCandidate, TargetGroup, _fit_group, bistatic_position_covariance,
)
from .multistatic_baselines import _dbscan_labels, _project
from .multistatic_targets import KinematicNode


@dataclass(frozen=True)
class LocalModel:
    groups: tuple[TargetGroup, ...]
    clutter: tuple[PathCandidate, ...]
    bic: float


def _clutter_baseline_assignment(
    target_costs: np.ndarray, clutter_costs: np.ndarray
) -> tuple[np.ndarray, float]:
    """Solve one UAV's target-or-clutter assignment exactly by subset DP.

    Every candidate may be assigned to clutter, while each physical target
    accepts at most one candidate from this UAV. Subtracting the all-clutter
    cost leaves a minimum-weight partial matching with increments
    ``target_costs[k, q] - clutter_costs[k]``. A target-occupancy mask is a
    sufficient state, giving O(K q 2^q) time and O(2^q) value storage instead
    of padding the problem with K interchangeable clutter columns.
    """
    target_costs = np.asarray(target_costs, dtype=float)
    clutter_costs = np.asarray(clutter_costs, dtype=float)
    if target_costs.ndim != 2 or clutter_costs.shape != (len(target_costs),):
        raise ValueError("assignment costs have incompatible shapes")
    if np.any(~np.isfinite(target_costs)) or np.any(~np.isfinite(clutter_costs)):
        raise ValueError("assignment costs must be finite")
    candidate_count, target_count = target_costs.shape
    state_count = 1 << target_count
    values = np.full(state_count, np.inf)
    values[0] = 0.0
    parents = []
    increments = target_costs - clutter_costs[:, None]
    for candidate in range(candidate_count):
        updated = values.copy()  # Candidate remains in clutter.
        parent = [(mask, -1) for mask in range(state_count)]
        for mask in range(state_count):
            if not np.isfinite(values[mask]):
                continue
            for target in range(target_count):
                bit = 1 << target
                if mask & bit:
                    continue
                new_mask = mask | bit
                proposal = values[mask] + increments[candidate, target]
                if proposal < updated[new_mask] - 1e-12:
                    updated[new_mask] = proposal
                    parent[new_mask] = (mask, target)
        values = updated
        parents.append(parent)
    final_mask = int(np.argmin(values))
    assignment = np.full(candidate_count, -1, dtype=int)
    mask = final_mask
    for candidate in range(candidate_count - 1, -1, -1):
        previous_mask, target = parents[candidate][mask]
        assignment[candidate] = target
        mask = previous_mask
    return assignment, float(np.sum(clutter_costs) + values[final_mask])


def poisson_binomial_tail(
    probabilities: Iterable[float], minimum_successes: int
) -> float:
    """Return ``P(sum(Z_m) >= minimum_successes)`` for Bernoulli views.

    The dynamic program treats different UAV views as conditionally
    independent experimental units. It deliberately does *not* treat peaks
    from the same UAV as independent, because they share front-end noise and
    waveform sidelobes.
    """
    values = np.asarray(tuple(probabilities), dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("view false-extra probabilities must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("view false-extra probabilities must lie in [0, 1]")
    if minimum_successes <= 0:
        return 1.0
    if minimum_successes > len(values):
        return 0.0
    mass = np.zeros(len(values) + 1)
    mass[0] = 1.0
    for probability in values:
        mass[1:] = (
            mass[1:] * (1.0 - probability)
            + mass[:-1] * probability
        )
        mass[0] *= 1.0 - probability
    return float(np.clip(np.sum(mass[minimum_successes:]), 0.0, 1.0))


def collision_support_threshold(
    probabilities: Iterable[float], false_alarm_probability: float
) -> int | None:
    """Smallest cross-UAV support giving a collision false alarm <= alpha."""
    values = tuple(probabilities)
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("collision false-alarm probability must lie in (0, 1)")
    # Validate even the empty input through the tail routine.
    poisson_binomial_tail(values, 1)
    for support in range(1, len(values) + 1):
        if poisson_binomial_tail(values, support) <= false_alarm_probability:
            return support
    return None


def _joint_model_cost(
    groups: list[list[tuple[PathCandidate, np.ndarray]]],
    clutter: list[tuple[PathCandidate, np.ndarray]],
    fitted: list[TargetGroup],
    nodes: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    position_sigma_m: float,
    doppler_sigma_hz: float,
    clutter_log_density_ratio: float,
    propagation_speed: float,
) -> float:
    cost = 0.0
    for group_members, state in zip(groups, fitted):
        for candidate, position in group_members:
            probability = float(np.clip(candidate.confidence, 1e-6, 1 - 1e-6))
            predicted = _predict_doppler(
                state.position, state.velocity, nodes[candidate.transmitter_id],
                receiver, carrier_hz, propagation_speed,
            )
            cost += (np.linalg.norm(position - state.position) / position_sigma_m) ** 2
            cost += ((candidate.doppler_hz - predicted) / doppler_sigma_hz) ** 2
            cost += -2.0 * np.log(probability)
    for candidate, _ in clutter:
        probability = float(np.clip(candidate.confidence, 1e-6, 1 - 1e-6))
        cost += -2.0 * np.log1p(-probability) + clutter_log_density_ratio
    return float(cost)


def _joint_refine(
    members: list[tuple[PathCandidate, np.ndarray]],
    initial_groups: list[list[tuple[PathCandidate, np.ndarray]]],
    initial_clutter: list[tuple[PathCandidate, np.ndarray]],
    nodes: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    position_sigma_m: float,
    doppler_sigma_hz: float,
    clutter_log_density_ratio: float,
    minimum_transmitters: int,
    maximum_velocity_condition: float,
    maximum_iterations: int,
    propagation_speed: float,
) -> tuple[list[list[tuple[PathCandidate, np.ndarray]]], list[tuple[PathCandidate, np.ndarray]], list[TargetGroup]]:
    """Monotone constrained coordinate descent after position initialization."""
    groups = [list(group) for group in initial_groups]
    clutter = list(initial_clutter)
    fitted = [_fit_group(
        tuple(item[0] for item in group), tuple(item[1] for item in group),
        nodes, receiver, carrier_hz, propagation_speed,
    ) for group in groups]
    best_cost = _joint_model_cost(
        groups, clutter, fitted, nodes, receiver, carrier_hz,
        position_sigma_m, doppler_sigma_hz, clutter_log_density_ratio,
        propagation_speed,
    )
    order = len(groups)
    by_transmitter: dict[int, list[tuple[PathCandidate, np.ndarray]]] = {}
    for member in members:
        by_transmitter.setdefault(member[0].transmitter_id, []).append(member)
    for _ in range(maximum_iterations):
        proposed = [[] for _ in range(order)]
        proposed_clutter = []
        for transmitter_id, transmitter_members in sorted(by_transmitter.items()):
            costs = np.empty((len(transmitter_members), order + len(transmitter_members)))
            for row, (candidate, position) in enumerate(transmitter_members):
                probability = float(np.clip(candidate.confidence, 1e-6, 1 - 1e-6))
                costs[row, order:] = (
                    -2.0 * np.log1p(-probability) + clutter_log_density_ratio
                )
                for column, state in enumerate(fitted):
                    predicted = _predict_doppler(
                        state.position, state.velocity, nodes[transmitter_id],
                        receiver, carrier_hz, propagation_speed,
                    )
                    costs[row, column] = (
                        (np.linalg.norm(position - state.position) / position_sigma_m) ** 2
                        + ((candidate.doppler_hz - predicted) / doppler_sigma_hz) ** 2
                        - 2.0 * np.log(probability)
                    )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                member = transmitter_members[row]
                if column < order and costs[row, column] < costs[row, order]:
                    proposed[int(column)].append(member)
                else:
                    proposed_clutter.append(member)
        if any(len({item[0].transmitter_id for item in group}) < minimum_transmitters
               for group in proposed):
            break
        proposed_fitted = [_fit_group(
            tuple(item[0] for item in group), tuple(item[1] for item in group),
            nodes, receiver, carrier_hz, propagation_speed,
        ) for group in proposed]
        if any(_velocity_geometry_condition(
            state.position, state.paths, nodes, receiver
        ) > maximum_velocity_condition for state in proposed_fitted):
            break
        proposed_cost = _joint_model_cost(
            proposed, proposed_clutter, proposed_fitted, nodes, receiver,
            carrier_hz, position_sigma_m, doppler_sigma_hz,
            clutter_log_density_ratio, propagation_speed,
        )
        if proposed_cost >= best_cost - 1e-9:
            break
        groups, clutter, fitted = proposed, proposed_clutter, proposed_fitted
        best_cost = proposed_cost
    return groups, clutter, fitted


def _farthest_centers(
    members: list[tuple[PathCandidate, np.ndarray]], count: int
) -> np.ndarray:
    first = max(range(len(members)), key=lambda i: members[i][0].confidence)
    selected = [first]
    while len(selected) < count:
        distances = np.asarray([
            min(np.linalg.norm(item[1] - members[index][1]) for index in selected)
            for item in members
        ])
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return np.asarray([members[index][1] for index in selected])


def _predict_doppler(
    position: np.ndarray,
    velocity: np.ndarray,
    transmitter: KinematicNode,
    receiver: KinematicNode,
    carrier_hz: float,
    propagation_speed: float,
) -> float:
    tx_leg = position - transmitter.position
    rx_leg = position - receiver.position
    tx_unit = tx_leg / np.linalg.norm(tx_leg)
    rx_unit = rx_leg / np.linalg.norm(rx_leg)
    range_rate = (
        np.dot(tx_unit, velocity - transmitter.velocity)
        + np.dot(rx_unit, velocity - receiver.velocity)
    )
    return float(carrier_hz * range_rate / propagation_speed)


def _velocity_geometry_condition(
    position: np.ndarray,
    paths: tuple[PathCandidate, ...],
    transmitters: tuple[KinematicNode, ...],
    receiver: KinematicNode,
) -> float:
    """Condition number of the 2-D bistatic range-rate observation matrix."""
    rows = []
    for path in paths:
        transmitter = transmitters[path.transmitter_id]
        tx_leg = position - transmitter.position
        rx_leg = position - receiver.position
        tx_norm = np.linalg.norm(tx_leg)
        rx_norm = np.linalg.norm(rx_leg)
        if tx_norm <= 1e-12 or rx_norm <= 1e-12:
            return np.inf
        rows.append(tx_leg / tx_norm + rx_leg / rx_norm)
    singular_values = np.linalg.svd(np.asarray(rows), compute_uv=False)
    if len(singular_values) < 2 or singular_values[-1] <= 1e-10:
        return np.inf
    return float(singular_values[0] / singular_values[-1])


def _robust_velocity_refinement(
    group: TargetGroup,
    transmitters: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    propagation_speed: float,
    doppler_sigma_hz: float | Sequence[float],
    position_covariance: np.ndarray | None = None,
    huber_delta: float = 1.345,
    maximum_iterations: int = 20,
) -> TargetGroup:
    """Refit velocity by confidence-weighted Huber IRLS at fixed position.

    Bistatic Doppler is linear in target velocity once position and path
    association are fixed.  The bounded Huber score limits the influence of a
    wrongly associated sidelobe while retaining quadratic efficiency for
    inlier Doppler noise.  This post-fit does not alter model order or paths.
    """
    if huber_delta <= 0 or maximum_iterations <= 0:
        raise ValueError("robust velocity parameters must be positive")
    if np.isscalar(doppler_sigma_hz):
        per_path_sigma = np.full(
            len(group.paths), float(doppler_sigma_hz)
        )
    else:
        per_path_sigma = np.asarray(tuple(doppler_sigma_hz), dtype=float)
        if per_path_sigma.shape != (len(group.paths),):
            raise ValueError(
                "one Doppler sigma is required per retained path"
            )
    if np.any(~np.isfinite(per_path_sigma)) or np.any(per_path_sigma <= 0.0):
        raise ValueError("Doppler sigmas must be positive and finite")
    rows, right, confidence = [], [], []
    for path in group.paths:
        transmitter = transmitters[path.transmitter_id]
        tx_leg = group.position - transmitter.position
        rx_leg = group.position - receiver.position
        tx_unit = tx_leg / np.linalg.norm(tx_leg)
        rx_unit = rx_leg / np.linalg.norm(rx_leg)
        rows.append(tx_unit + rx_unit)
        offset = (np.dot(tx_unit, transmitter.velocity)
                  + np.dot(rx_unit, receiver.velocity))
        right.append(path.doppler_hz * propagation_speed / carrier_hz + offset)
        confidence.append(np.clip(path.confidence, 1e-6, 1.0))
    matrix = np.asarray(rows)
    right = np.asarray(right)
    base_weights = np.asarray(confidence)
    velocity = np.asarray(group.velocity, dtype=float).copy()
    # Convert the front-end Doppler scale to bistatic range-rate units.  When a
    # position covariance is available, first-order propagation adds the
    # uncertainty of the geometry-dependent bistatic look vectors.
    base_scale_squared = (
        per_path_sigma * propagation_speed / carrier_hz
    ) ** 2
    for _ in range(maximum_iterations):
        scale_squared = base_scale_squared.copy()
        if position_covariance is not None:
            covariance = np.asarray(position_covariance, dtype=float)
            if covariance.shape != (2, 2):
                raise ValueError("position covariance must be 2 by 2")
            for index, path in enumerate(group.paths):
                transmitter = transmitters[path.transmitter_id]
                tx_leg = group.position - transmitter.position
                rx_leg = group.position - receiver.position
                tx_norm = np.linalg.norm(tx_leg)
                rx_norm = np.linalg.norm(rx_leg)
                tx_unit = tx_leg / tx_norm
                rx_unit = rx_leg / rx_norm
                tx_relative_velocity = velocity - transmitter.velocity
                rx_relative_velocity = velocity - receiver.velocity
                gradient = (
                    (np.eye(2) - np.outer(tx_unit, tx_unit))
                    @ tx_relative_velocity / tx_norm
                    + (np.eye(2) - np.outer(rx_unit, rx_unit))
                    @ rx_relative_velocity / rx_norm
                )
                scale_squared[index] += max(
                    float(gradient @ covariance @ gradient), 0.0
                )
        scales = np.sqrt(scale_squared)
        standardized = (matrix @ velocity - right) / scales
        robust = np.ones_like(standardized)
        large = np.abs(standardized) > huber_delta
        robust[large] = huber_delta / np.abs(standardized[large])
        weights = base_weights * robust / scale_squared
        proposed = np.linalg.lstsq(
            matrix * np.sqrt(weights)[:, None], right * np.sqrt(weights),
            rcond=None,
        )[0]
        if np.linalg.norm(proposed - velocity) <= 1e-8 * (
            1.0 + np.linalg.norm(velocity)
        ):
            velocity = proposed
            break
        velocity = proposed
    doppler_residual_hz = (
        (matrix @ velocity - right) * carrier_hz / propagation_speed
    )
    residual = float(np.sqrt(np.average(
        doppler_residual_hz ** 2, weights=base_weights
    )))
    return TargetGroup(group.paths, group.position, velocity, residual)


def _covariance_weighted_position_refinement(
    group: TargetGroup,
    transmitters: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    propagation_speed: float,
) -> TargetGroup:
    """GLS position refit using range/bearing geometry-derived covariances."""
    precisions = []
    weighted_positions = []
    for path in group.paths:
        transmitter = transmitters[path.transmitter_id]
        range_sigma_m = (
            1.5 if path.range_sigma_m is None else path.range_sigma_m
        )
        angle_sigma_rad = (
            np.deg2rad(0.4)
            if path.angle_sigma_rad is None else path.angle_sigma_rad
        )
        covariance = bistatic_position_covariance(
            transmitter.position, receiver.position,
            path.delay_s * propagation_speed, path.receive_azimuth_rad,
            range_sigma_m=range_sigma_m,
            angle_sigma_rad=angle_sigma_rad,
        )
        precision = np.linalg.inv(covariance)
        # Confidence is an existence weight only; covariance describes the
        # conditional measurement precision once the path is retained.
        weight = float(np.clip(path.confidence, 1e-6, 1.0))
        precisions.append(weight * precision)
        projected = _project((path,), transmitters, receiver, propagation_speed)
        if not projected:
            return group
        weighted_positions.append(weight * precision @ projected[0][1])
    information = np.sum(precisions, axis=0)
    if np.linalg.cond(information) > 1e10:
        return group
    position = np.linalg.solve(information, np.sum(weighted_positions, axis=0))
    position_covariance = np.linalg.inv(information)
    provisional = TargetGroup(
        group.paths, position, group.velocity, group.residual
    )
    doppler_sigmas = [
        3.0 if path.doppler_sigma_hz is None else path.doppler_sigma_hz
        for path in group.paths
    ]
    return _robust_velocity_refinement(
        provisional, transmitters, receiver, carrier_hz,
        propagation_speed, doppler_sigmas, position_covariance,
    )


def _final_state_refinement(
    groups: Iterable[TargetGroup],
    transmitters: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    propagation_speed: float,
    doppler_sigma_hz: float,
    robust_velocity: bool,
    covariance_weighted_position: bool,
) -> tuple[TargetGroup, ...]:
    """Apply one order-independent, post-decision state estimator."""
    refined = tuple(groups)
    if covariance_weighted_position:
        # The covariance refit already recomputes velocity robustly at the new
        # position.  Running the old-position velocity refit first is redundant.
        return tuple(_covariance_weighted_position_refinement(
            group, transmitters, receiver, carrier_hz, propagation_speed,
        ) for group in refined)
    if robust_velocity:
        refined = tuple(_robust_velocity_refinement(
            group, transmitters, receiver, carrier_hz, propagation_speed,
            doppler_sigma_hz,
        ) for group in refined)
    return refined


def _fit_order(
    members: list[tuple[PathCandidate, np.ndarray]],
    order: int,
    nodes: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    position_sigma_m: float,
    doppler_sigma_hz: float,
    clutter_log_density_ratio: float,
    minimum_transmitters: int,
    maximum_velocity_condition: float,
    maximum_iterations: int,
    propagation_speed: float,
    initial_centers: np.ndarray | None = None,
    joint_refinement_iterations: int = 3,
) -> LocalModel | None:
    centers = (
        _farthest_centers(members, order)
        if initial_centers is None else np.asarray(initial_centers, dtype=float).copy()
    )
    velocities = np.zeros((order, 2))
    previous_signature = None
    assigned_groups: list[list[tuple[PathCandidate, np.ndarray]]] = []
    clutter: list[tuple[PathCandidate, np.ndarray]] = []
    for iteration in range(maximum_iterations):
        assigned_groups = [[] for _ in range(order)]
        clutter = []
        by_transmitter: dict[int, list[tuple[PathCandidate, np.ndarray]]] = {}
        for member in members:
            by_transmitter.setdefault(member[0].transmitter_id, []).append(member)
        signature = []
        for transmitter_id, transmitter_members in sorted(by_transmitter.items()):
            candidate_count = len(transmitter_members)
            costs = np.empty((candidate_count, order + candidate_count))
            for row, (candidate, position) in enumerate(transmitter_members):
                probability = float(np.clip(candidate.confidence, 1e-6, 1 - 1e-6))
                target_prior_cost = -2.0 * np.log(probability)
                clutter_prior_cost = (
                    -2.0 * np.log1p(-probability) + clutter_log_density_ratio
                )
                costs[row, order:] = clutter_prior_cost
                for column in range(order):
                    position_cost = (
                        np.linalg.norm(position - centers[column]) / position_sigma_m
                    ) ** 2
                    costs[row, column] = (
                        position_cost + target_prior_cost
                    )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                member = transmitter_members[row]
                clutter_threshold = costs[row, order]
                if column < order and costs[row, column] < clutter_threshold:
                    assigned_groups[int(column)].append(member)
                    signature.append((transmitter_id, id(member[0]), int(column)))
                else:
                    clutter.append(member)
                    signature.append((transmitter_id, id(member[0]), -1))
        fitted = []
        valid_groups = []
        for group_index, group in enumerate(assigned_groups):
            if len({item[0].transmitter_id for item in group}) < minimum_transmitters:
                fitted.append(None)
                continue
            estimate = _fit_group(
                tuple(item[0] for item in group),
                tuple(item[1] for item in group),
                nodes, receiver, carrier_hz, propagation_speed,
            )
            fitted.append(estimate)
            valid_groups.append(group_index)
            if _velocity_geometry_condition(
                estimate.position, estimate.paths, nodes, receiver
            ) > maximum_velocity_condition:
                fitted[-1] = None
                valid_groups.pop()
        for group_index in valid_groups:
            centers[group_index] = fitted[group_index].position
            velocities[group_index] = fitted[group_index].velocity
        current_signature = tuple(sorted(signature))
        if current_signature == previous_signature:
            break
        previous_signature = current_signature

    # A q-state model is physically infeasible unless every state has enough
    # distinct bistatic views before joint refinement. Do not manufacture an
    # empty state or let np.average normalize zero weight.
    if any(len({item[0].transmitter_id for item in group}) < minimum_transmitters
           for group in assigned_groups):
        return None
    if joint_refinement_iterations > 0:
        assigned_groups, clutter, refined_groups = _joint_refine(
            members, assigned_groups, clutter, nodes, receiver, carrier_hz,
            position_sigma_m, doppler_sigma_hz, clutter_log_density_ratio,
            minimum_transmitters, maximum_velocity_condition,
            min(maximum_iterations, joint_refinement_iterations), propagation_speed,
        )
    else:
        refined_groups = [_fit_group(
            tuple(item[0] for item in group), tuple(item[1] for item in group),
            nodes, receiver, carrier_hz, propagation_speed,
        ) for group in assigned_groups]
    likelihood_cost = sum(
        -2.0 * np.log1p(-np.clip(item[0].confidence, 1e-6, 1 - 1e-6))
        + clutter_log_density_ratio
        for item in clutter
    )
    final_groups = []
    for group_members, group in zip(assigned_groups, refined_groups):
        if len({item[0].transmitter_id for item in group_members}) < minimum_transmitters:
            return None
        if _velocity_geometry_condition(
            group.position, group.paths, nodes, receiver
        ) > maximum_velocity_condition:
            return None
        final_groups.append(group)
        for candidate, position in group_members:
            likelihood_cost += -2.0 * np.log(np.clip(
                candidate.confidence, 1e-6, 1 - 1e-6
            ))
            likelihood_cost += (
                np.linalg.norm(position - group.position) / position_sigma_m
            ) ** 2
            predicted = _predict_doppler(
                group.position, group.velocity, nodes[candidate.transmitter_id],
                receiver, carrier_hz, propagation_speed,
            )
            likelihood_cost += (
                (candidate.doppler_hz - predicted) / doppler_sigma_hz
            ) ** 2
    parameter_count = 4 * order
    # Peaks produced by one UAV share waveform sidelobes and front-end noise;
    # treating them as independent samples over-penalizes higher model orders.
    # Distinct transmitter views are the independent experimental units here.
    effective_samples = len({item[0].transmitter_id for item in members})
    bic = float(
        likelihood_cost + parameter_count * np.log(max(effective_samples, 2))
    )
    return LocalModel(
        tuple(final_groups), tuple(item[0] for item in clutter), bic
    )


def _initializations(
    members: list[tuple[PathCandidate, np.ndarray]], order: int,
    maximum_starts: int = 4,
) -> tuple[np.ndarray, ...]:
    """Deterministic transmitter-anchored multi-start centers.

    These starts exploit the fact that one UAV can contribute at most one path per target:
    the strongest ``order`` peaks from one UAV form a physically meaningful
    local state hypothesis. A global farthest-first start is retained as a
    fallback when no single UAV observes every local target.
    """
    if order == 1:
        strongest = max(members, key=lambda item: item[0].confidence)
        return (np.asarray([strongest[1]]),)
    starts = [_farthest_centers(members, order)]
    by_transmitter: dict[int, list[tuple[PathCandidate, np.ndarray]]] = {}
    for member in members:
        by_transmitter.setdefault(member[0].transmitter_id, []).append(member)
    anchors = []
    for transmitter_members in by_transmitter.values():
        ranked = sorted(
            transmitter_members,
            key=lambda item: item[0].confidence,
            reverse=True,
        )
        if len(ranked) < order:
            continue
        selected = ranked[:order]
        if order > 1 and min(
            np.linalg.norm(selected[left][1] - selected[right][1])
            for left in range(order) for right in range(left + 1, order)
        ) < 0.5:
            continue
        anchors.append((
            sum(item[0].confidence for item in selected),
            np.asarray([item[1] for item in selected]),
        ))
    starts.extend(centers for _, centers in sorted(
        anchors, key=lambda item: item[0], reverse=True
    )[:maximum_starts - 1])
    unique = []
    signatures = set()
    for centers in starts:
        signature = tuple(np.round(np.sort(centers, axis=0).ravel(), 6))
        if signature not in signatures:
            signatures.add(signature)
            unique.append(centers)
    return tuple(unique)


def _initial_center_profile(
    members: list[tuple[PathCandidate, np.ndarray]],
    centers: np.ndarray,
    position_sigma_m: float,
    clutter_log_density_ratio: float,
) -> tuple[float, tuple]:
    """One-step profile cost for ranking physical multi-start hypotheses.

    Velocity is deliberately omitted because it is unidentified before path
    grouping. For each UAV, the score exactly profiles candidate assignments
    over the proposed position centers and clutter. It is therefore a
    likelihood-consistent screening statistic, not a geometric heuristic.
    """
    total = 0.0
    target_members = [[] for _ in range(len(centers))]
    clutter_members = []
    order = len(centers)
    by_transmitter: dict[int, list[tuple[PathCandidate, np.ndarray]]] = {}
    for member in members:
        by_transmitter.setdefault(member[0].transmitter_id, []).append(member)
    for transmitter_members in by_transmitter.values():
        costs = np.empty((len(transmitter_members), order + len(transmitter_members)))
        for row, (candidate, position) in enumerate(transmitter_members):
            probability = float(np.clip(candidate.confidence, 1e-6, 1 - 1e-6))
            costs[row, order:] = (
                -2.0 * np.log1p(-probability) + clutter_log_density_ratio
            )
            costs[row, :order] = (
                np.linalg.norm(centers - position, axis=1) / position_sigma_m
            ) ** 2 - 2.0 * np.log(probability)
        rows, columns = linear_sum_assignment(costs)
        total += float(np.sum(costs[rows, columns]))
        for row, column in zip(rows, columns):
            candidate_key = id(transmitter_members[int(row)][0])
            if column < order and costs[row, column] < costs[row, order]:
                target_members[int(column)].append(candidate_key)
            else:
                clutter_members.append(candidate_key)
    # Target labels are exchangeable. Canonicalization makes starts that only
    # permute target columns share one assignment basin signature.
    canonical_targets = tuple(sorted(
        tuple(sorted(group)) for group in target_members
    ))
    signature = (canonical_targets, tuple(sorted(clutter_members)))
    return total, signature


def _initial_center_score(
    members: list[tuple[PathCandidate, np.ndarray]],
    centers: np.ndarray,
    position_sigma_m: float,
    clutter_log_density_ratio: float,
) -> float:
    return _initial_center_profile(
        members, centers, position_sigma_m, clutter_log_density_ratio
    )[0]


def physics_order_evidence(
    members: list[tuple[PathCandidate, np.ndarray]],
    nodes: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    position_sigma_m: float,
    doppler_sigma_hz: float,
    clutter_log_density_ratio: float,
    minimum_transmitters: int,
    maximum_velocity_condition: float,
    maximum_iterations: int,
    propagation_speed: float,
    joint_refinement_iterations: int = 0,
) -> tuple[float, LocalModel | None]:
    """Penalized generalized-likelihood gain from one to two targets.

    Both hypotheses use the same path probabilities, clutter density,
    per-UAV capacity constraint, and bistatic observability checks. The gain is
    positive only when the two-state physical fit overcomes its extra BIC-like
    parameter penalty. Positive ``joint_refinement_iterations`` applies the
    same monotone constrained coordinate descent under both hypotheses.
    """
    if joint_refinement_iterations < 0:
        raise ValueError("joint refinement iterations cannot be negative")
    best = {}
    for order in (1, 2):
        raw_starts = _initializations(members, order)
        starts = _screen_distinct_initializations(
            members, raw_starts, position_sigma_m,
            clutter_log_density_ratio,
            maximum_refined_starts=(1 if order == 1 else 4),
        )
        models = [
            model for centers in starts
            if (model := _fit_order(
                members, order, nodes, receiver, carrier_hz,
                position_sigma_m, doppler_sigma_hz,
                clutter_log_density_ratio, minimum_transmitters,
                maximum_velocity_condition, maximum_iterations,
                propagation_speed, centers,
                joint_refinement_iterations=joint_refinement_iterations,
            )) is not None
        ]
        if not models:
            return -np.inf, None
        best[order] = min(model.bic for model in models)
        if order == 2:
            best_two_model = min(models, key=lambda model: model.bic)
    return float(best[1] - best[2]), best_two_model


def physics_order_gain(
    members: list[tuple[PathCandidate, np.ndarray]],
    nodes: tuple[KinematicNode, ...],
    receiver: KinematicNode,
    carrier_hz: float,
    position_sigma_m: float,
    doppler_sigma_hz: float,
    clutter_log_density_ratio: float,
    minimum_transmitters: int,
    maximum_velocity_condition: float,
    maximum_iterations: int,
    propagation_speed: float,
    joint_refinement_iterations: int = 0,
) -> float:
    """Backward-compatible scalar view of :func:`physics_order_evidence`."""
    return physics_order_evidence(
        members, nodes, receiver, carrier_hz, position_sigma_m,
        doppler_sigma_hz, clutter_log_density_ratio, minimum_transmitters,
        maximum_velocity_condition, maximum_iterations, propagation_speed,
        joint_refinement_iterations,
    )[0]


def _screen_distinct_initializations(
    members: list[tuple[PathCandidate, np.ndarray]],
    starts: Iterable[np.ndarray],
    position_sigma_m: float,
    clutter_log_density_ratio: float,
    maximum_refined_starts: int = 2,
) -> tuple[np.ndarray, ...]:
    """Keep the cheapest representative of each first-assignment basin."""
    if maximum_refined_starts <= 0:
        raise ValueError("maximum refined starts must be positive")
    representatives = {}
    for centers in starts:
        score, signature = _initial_center_profile(
            members, centers, position_sigma_m, clutter_log_density_ratio
        )
        incumbent = representatives.get(signature)
        if incumbent is None or score < incumbent[0]:
            representatives[signature] = (score, centers)
    return tuple(centers for _, centers in sorted(
        representatives.values(), key=lambda item: item[0]
    )[:maximum_refined_starts])


def bic_conflict_association(
    candidates: Iterable[PathCandidate],
    transmitters: Iterable[KinematicNode],
    receiver: KinematicNode,
    carrier_hz: float,
    *,
    position_tolerance_m: float,
    position_sigma_m: float = 3.0,
    doppler_sigma_hz: float = 3.0,
    clutter_doppler_span_hz: float = 1800.0,
    minimum_transmitters: int = 2,
    view_false_extra_probability: float | Iterable[float] = 0.1,
    collision_false_alarm_probability: float = 0.05,
    view_false_target_probability: float | Iterable[float] = 0.1,
    target_false_alarm_probability: float = 0.05,
    required_target_support_override: int | None = None,
    required_collision_support_override: int | None = None,
    maximum_velocity_condition: float = 100.0,
    maximum_local_targets: int = 4,
    order_confidence_threshold: float = 0.6,
    collision_gate_mode: str = "hard_null",
    collision_posterior_failure_probability: float = 0.05,
    collision_support_calibrators: dict[int, object] | None = None,
    collision_statistic_thresholds: dict[int, float] | None = None,
    physics_collision_threshold: float | None = None,
    physics_frame_thresholds: dict[int, float] | None = None,
    physics_stepdown_thresholds: tuple[float, ...] | None = None,
    physics_stepdown_activation_count: int = 1,
    physics_glrt_refinement_iterations: int = 0,
    physics_cascade_thresholds: tuple[float, float, float] | None = None,
    final_joint_refinement_iterations: int = 3,
    covariance_weighted_final_state: bool = False,
    robust_final_velocity: bool = False,
    physics_conformal_null=None,
    physics_conformal_p_threshold: float | None = None,
    maximum_iterations: int = 20,
    propagation_speed: float = 299_792_458.0,
) -> tuple[TargetGroup, ...]:
    """Select local target count by a calibrated test, then fit its states.

    For order ``q``, UAV ``m`` supports ``H_q`` when it provides at least
    ``q`` candidates above ``order_confidence_threshold``. Under ``H_{q-1}``,
    an additional supported peak is modeled as Bernoulli with calibrated
    probability ``view_false_extra_probability[m]``. The Poisson-binomial
    tail therefore controls the probability of falsely opening order ``q``.
    """
    nodes = tuple(transmitters)
    if position_sigma_m <= 0 or doppler_sigma_hz <= 0:
        raise ValueError("position and Doppler standard deviations must be positive")
    if clutter_doppler_span_hz <= 0:
        raise ValueError("clutter Doppler span must be positive")
    if collision_gate_mode not in {
        "hard_null", "posterior_support", "empirical_null", "physics_glrt",
        "physics_conformal", "physics_frame_stratified", "physics_stepdown",
        "physics_cascade"
    }:
        raise ValueError("unsupported collision gate mode")
    if final_joint_refinement_iterations < 0:
        raise ValueError("final joint refinement iterations cannot be negative")
    if not 0 < collision_posterior_failure_probability < 1:
        raise ValueError("posterior failure probability must lie in (0, 1)")
    if np.isscalar(view_false_extra_probability):
        false_extra_probabilities = np.full(
            len(nodes), float(view_false_extra_probability)
        )
    else:
        false_extra_probabilities = np.asarray(
            tuple(view_false_extra_probability), dtype=float
        )
        if false_extra_probabilities.shape != (len(nodes),):
            raise ValueError(
                "one view false-extra probability is required per transmitter"
            )
    if required_collision_support_override is not None:
        if required_collision_support_override < 1:
            raise ValueError(
                "required collision support must be positive"
            )
        required_collision_support = required_collision_support_override
    else:
        required_collision_support = collision_support_threshold(
            false_extra_probabilities, collision_false_alarm_probability
        )
    if np.isscalar(view_false_target_probability):
        false_target_probabilities = np.full(
            len(nodes), float(view_false_target_probability)
        )
    else:
        false_target_probabilities = np.asarray(
            tuple(view_false_target_probability), dtype=float
        )
        if false_target_probabilities.shape != (len(nodes),):
            raise ValueError(
                "one view false-target probability is required per transmitter"
            )
    if required_target_support_override is not None:
        if required_target_support_override < 1:
            raise ValueError(
                "required target support must be positive"
            )
        required_target_support = required_target_support_override
    else:
        required_target_support = collision_support_threshold(
            false_target_probabilities, target_false_alarm_probability
        )
    if required_target_support is None:
        return ()
    effective_minimum_transmitters = max(
        minimum_transmitters, required_target_support
    )
    clutter_volume = (
        np.pi * position_tolerance_m ** 2 * clutter_doppler_span_hz
    )
    target_noise_volume = (
        (2.0 * np.pi) ** 1.5 * position_sigma_m ** 2 * doppler_sigma_hz
    )
    clutter_log_density_ratio = 2.0 * np.log(
        clutter_volume / target_noise_volume
    )
    projected = _project(candidates, nodes, receiver, propagation_speed)
    if not projected:
        return ()
    positions = np.asarray([entry[1] for entry in projected])
    labels = _dbscan_labels(
        positions / position_tolerance_m, radius=1.0, min_samples=2
    )
    component_members = [
        [projected[index] for index in np.flatnonzero(labels == label)]
        for label in range(int(labels.max()) + 1)
    ]
    # This nuisance stratum is computed before any GLRT decision.  It measures
    # whether at least one spatial component contains two or more repeated
    # same-UAV peaks, a principal source of the separated-frame GLRT tail.
    frame_excess_stratum = int(any(
        len(members) - len({candidate.transmitter_id
                            for candidate, _ in members}) >= 2
        for members in component_members
    ))
    stepdown_confirmed = set()
    stepdown_gains = {}
    stepdown_models = {}
    if collision_gate_mode in {"physics_stepdown", "physics_cascade"}:
        if collision_gate_mode == "physics_cascade":
            if (physics_cascade_thresholds is None or
                    len(physics_cascade_thresholds) != 3):
                raise ValueError("three cascade thresholds are required")
            active_thresholds = physics_cascade_thresholds[:2]
        else:
            active_thresholds = physics_stepdown_thresholds
        if not physics_stepdown_thresholds:
            if collision_gate_mode == "physics_stepdown":
                raise ValueError("step-down thresholds are required")
        if physics_stepdown_activation_count <= 0:
            raise ValueError("step-down activation count must be positive")
        for component_index, members in enumerate(component_members):
            gain, model = physics_order_evidence(
                members, nodes, receiver, carrier_hz, position_sigma_m,
                doppler_sigma_hz, clutter_log_density_ratio,
                effective_minimum_transmitters, maximum_velocity_condition,
                maximum_iterations, propagation_speed,
                physics_glrt_refinement_iterations,
            )
            stepdown_gains[component_index] = gain
            stepdown_models[component_index] = model
        ranked = sorted(stepdown_gains, key=stepdown_gains.get, reverse=True)
        strong_count = sum(
            stepdown_gains[index] > active_thresholds[0]
            for index in ranked
        )
        if strong_count >= physics_stepdown_activation_count:
            for rank, component_index in enumerate(ranked):
                if rank >= len(active_thresholds):
                    break
                if stepdown_gains[component_index] > active_thresholds[rank]:
                    stepdown_confirmed.add(component_index)
                else:
                    break
        else:
            stepdown_confirmed.update(
                index for index in ranked
                if stepdown_gains[index] > active_thresholds[0]
            )
        if collision_gate_mode == "physics_cascade" and len(stepdown_confirmed) >= 2:
            remaining = [index for index in ranked
                         if index not in stepdown_confirmed]
            refined_candidates = []
            for index in remaining:
                gain, model = physics_order_evidence(
                    component_members[index], nodes, receiver, carrier_hz,
                    position_sigma_m, doppler_sigma_hz,
                    clutter_log_density_ratio, effective_minimum_transmitters,
                    maximum_velocity_condition, maximum_iterations,
                    propagation_speed, 3,
                )
                refined_candidates.append((gain, index, model))
            if refined_candidates:
                gain, index, model = max(refined_candidates, key=lambda item: item[0])
                if gain > physics_cascade_thresholds[2]:
                    stepdown_confirmed.add(index)
                    stepdown_models[index] = model
    output = []
    for component_index, members in enumerate(component_members):
        high_confidence_counts = np.zeros(len(nodes), dtype=int)
        for candidate, _ in members:
            if candidate.confidence >= order_confidence_threshold:
                high_confidence_counts[candidate.transmitter_id] += 1
        confirmed_order = 1
        if collision_gate_mode in {
            "physics_glrt", "physics_conformal", "physics_frame_stratified"
        }:
            gain = physics_order_gain(
                members, nodes, receiver, carrier_hz, position_sigma_m,
                doppler_sigma_hz, clutter_log_density_ratio,
                effective_minimum_transmitters, maximum_velocity_condition,
                maximum_iterations, propagation_speed,
                physics_glrt_refinement_iterations,
            )
            if collision_gate_mode == "physics_glrt":
                if physics_collision_threshold is None:
                    raise ValueError("physics collision threshold is required")
                if gain > physics_collision_threshold:
                    confirmed_order = 2
            elif collision_gate_mode == "physics_conformal":
                if (physics_conformal_null is None or
                        physics_conformal_p_threshold is None):
                    raise ValueError("conformal null and p threshold are required")
                distinct_views = len({
                    candidate.transmitter_id for candidate, _ in members
                })
                p_value = physics_conformal_null.p_value(
                    gain, len(members), distinct_views
                )
                if p_value < physics_conformal_p_threshold:
                    confirmed_order = 2
            else:
                if (physics_frame_thresholds is None or
                        set(physics_frame_thresholds) != {0, 1}):
                    raise ValueError("both frame-stratum thresholds are required")
                if gain > physics_frame_thresholds[frame_excess_stratum]:
                    confirmed_order = 2
        elif collision_gate_mode in {"physics_stepdown", "physics_cascade"}:
            if component_index in stepdown_confirmed:
                confirmed_order = 2
        elif required_collision_support is not None:
            for order in range(2, maximum_local_targets + 1):
                if collision_gate_mode == "hard_null":
                    observed_support = int(np.sum(high_confidence_counts >= order))
                    if observed_support < required_collision_support:
                        break
                else:
                    order_probabilities = []
                    for transmitter_id in range(len(nodes)):
                        scores = sorted((
                            candidate.confidence
                            for candidate, _ in members
                            if candidate.transmitter_id == transmitter_id
                        ), reverse=True)
                        if len(scores) < order:
                            probability = 0.0
                        else:
                            score = scores[order - 1]
                            calibrator = (
                                None if collision_support_calibrators is None
                                else collision_support_calibrators.get(order)
                            )
                            probability = (
                                score if calibrator is None
                                else float(calibrator(score))
                            )
                        order_probabilities.append(probability)
                    aggregate_statistic = poisson_binomial_tail(
                        order_probabilities, required_collision_support
                    )
                    if collision_gate_mode == "posterior_support":
                        if aggregate_statistic < (
                            1.0 - collision_posterior_failure_probability
                        ):
                            break
                    else:
                        if (collision_statistic_thresholds is None or
                                order not in collision_statistic_thresholds):
                            break
                        if aggregate_statistic <= collision_statistic_thresholds[order]:
                            break
                confirmed_order = order
        if confirmed_order == 1:
            # No independent multi-UAV evidence of a collision: use the
            # minimum-complexity identity-filtered estimator. This avoids
            # letting weak same-UAV sidelobes trigger unnecessary model search.
            best_by_transmitter = {}
            for member in members:
                transmitter_id = member[0].transmitter_id
                incumbent = best_by_transmitter.get(transmitter_id)
                if incumbent is None or member[0].confidence > incumbent[0].confidence:
                    best_by_transmitter[transmitter_id] = member
            selected = list(best_by_transmitter.values())
            if len(selected) < effective_minimum_transmitters:
                continue
            group = _fit_group(
                tuple(item[0] for item in selected),
                tuple(item[1] for item in selected),
                nodes, receiver, carrier_hz, propagation_speed,
            )
            if _velocity_geometry_condition(
                group.position, group.paths, nodes, receiver
            ) <= maximum_velocity_condition:
                output.extend(_final_state_refinement(
                    (group,), nodes, receiver, carrier_hz, propagation_speed,
                    doppler_sigma_hz, robust_final_velocity,
                    covariance_weighted_final_state,
                ))
            continue
        models = []
        # Orders not admitted by the sequential false-alarm test are not
        # searched. This keeps weak same-UAV sidelobes from increasing both
        # model order and runtime.
        for order in (confirmed_order,):
            cached_model = (
                stepdown_models.get(component_index)
                if collision_gate_mode in {"physics_stepdown", "physics_cascade"}
                and order == 2
                else None
            )
            if cached_model is not None:
                refined = _fit_order(
                    members, order, nodes, receiver, carrier_hz,
                    position_sigma_m, doppler_sigma_hz,
                    clutter_log_density_ratio,
                    effective_minimum_transmitters,
                    maximum_velocity_condition, maximum_iterations,
                    propagation_speed,
                    np.asarray([group.position for group in cached_model.groups]),
                    joint_refinement_iterations=final_joint_refinement_iterations,
                )
                if refined is not None:
                    models.append(refined)
                continue
            starts = _screen_distinct_initializations(
                members, _initializations(members, order),
                position_sigma_m, clutter_log_density_ratio,
                maximum_refined_starts=2,
            )
            screened = [
                (model, initial_centers)
                for initial_centers in starts
                if (model := _fit_order(
                    members, order, nodes, receiver, carrier_hz,
                    position_sigma_m, doppler_sigma_hz,
                    clutter_log_density_ratio,
                    effective_minimum_transmitters, maximum_velocity_condition,
                    maximum_iterations, propagation_speed,
                    initial_centers,
                    joint_refinement_iterations=0,
                )) is not None
            ]
            if screened:
                coarse_model, _ = min(screened, key=lambda item: item[0].bic)
                refined = _fit_order(
                    members, order, nodes, receiver, carrier_hz,
                    position_sigma_m, doppler_sigma_hz,
                    clutter_log_density_ratio,
                    effective_minimum_transmitters, maximum_velocity_condition,
                    maximum_iterations, propagation_speed,
                    np.asarray([group.position for group in coarse_model.groups]),
                    joint_refinement_iterations=final_joint_refinement_iterations,
                )
                if refined is not None:
                    models.append(refined)
        if models:
            # The public name is retained for compatibility with completed
            # audits. Cross-order choice is now made by the calibrated gate;
            # this score only chooses the best fit within the admitted order.
            chosen_groups = min(models, key=lambda model: model.bic).groups
            chosen_groups = _final_state_refinement(
                chosen_groups, nodes, receiver, carrier_hz, propagation_speed,
                doppler_sigma_hz, robust_final_velocity,
                covariance_weighted_final_state,
            )
            output.extend(chosen_groups)
    output.sort(key=lambda group: float(np.arctan2(
        group.position[1] - receiver.position[1],
        group.position[0] - receiver.position[0],
    )))
    return tuple(output)
