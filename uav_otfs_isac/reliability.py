from __future__ import annotations

from dataclasses import replace
from itertools import product
from collections.abc import Sequence

import numpy as np

from .models import TargetEvidenceModel


def common_state_pattern_distribution(
    success_prob: np.ndarray,
    owner: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Bernoulli mixture with exactly preserved link marginals.

    The equally likely common states use conditional probabilities p+d and
    p-d, where d=strength*min(p,1-p). This preserves E[gamma_i]=p_i while
    inducing Cov(gamma_i,gamma_j)=d_i*d_j for non-owner reporting links.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    marginal = np.asarray(success_prob, dtype=float)
    if np.any((marginal < 0.0) | (marginal > 1.0)):
        raise ValueError("success probabilities must lie in [0, 1]")
    displacement = strength * np.minimum(marginal, 1.0 - marginal)
    displacement[owner] = 0.0
    conditional = (marginal + displacement, marginal - displacement)
    patterns = np.asarray(list(product((0, 1), repeat=marginal.size)), dtype=np.int8)
    probabilities = np.zeros(patterns.shape[0], dtype=float)
    for state_probability, state_success in zip((0.5, 0.5), conditional):
        likelihood = np.prod(
            np.where(patterns == 1, state_success[None, :], 1.0 - state_success[None, :]),
            axis=1,
        )
        probabilities += state_probability * likelihood
    positive = probabilities > 1e-15
    patterns = patterns[positive]
    probabilities = probabilities[positive]
    probabilities /= probabilities.sum()
    return patterns, probabilities


def common_state_parameters(
    success_prob: np.ndarray, owner: int, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    marginal = np.asarray(success_prob, dtype=float)
    displacement = strength * np.minimum(marginal, 1.0 - marginal)
    displacement[owner] = 0.0
    return np.array([0.5, 0.5]), np.vstack((marginal + displacement, marginal - displacement))


def with_common_state_erasures(
    models: Sequence[TargetEvidenceModel], strength: float
) -> list[TargetEvidenceModel]:
    result = []
    for model in models:
        state_probabilities, conditional_success = common_state_parameters(
            model.success_prob, model.owner, strength
        )
        patterns, probabilities = common_state_pattern_distribution(
            model.success_prob, model.owner, strength
        )
        correlated = replace(
            model,
            reception_patterns=patterns,
            pattern_probabilities=probabilities,
            reception_state_probabilities=state_probabilities,
            conditional_success_probabilities=conditional_success,
        )
        correlated.validate()
        result.append(correlated)
    return result


def grouped_common_state_parameters(
    success_prob: np.ndarray,
    owner: int,
    groups: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent binary common states per failure group, preserving marginals."""
    marginal = np.asarray(success_prob, dtype=float)
    groups = np.asarray(groups, dtype=int)
    if groups.shape != marginal.shape:
        raise ValueError("groups must have one entry per UAV")
    group_ids = sorted(set(groups[i] for i in range(marginal.size) if i != owner))
    state_bits = np.asarray(list(product((0, 1), repeat=len(group_ids))), dtype=int)
    state_probabilities = np.full(state_bits.shape[0], 1.0 / state_bits.shape[0])
    displacement = strength * np.minimum(marginal, 1.0 - marginal)
    displacement[owner] = 0.0
    conditional = np.tile(marginal, (state_bits.shape[0], 1))
    group_column = {group: column for column, group in enumerate(group_ids)}
    for state_index, state in enumerate(state_bits):
        for i in range(marginal.size):
            if i == owner:
                continue
            sign = 1.0 if state[group_column[groups[i]]] else -1.0
            conditional[state_index, i] += sign * displacement[i]
    return state_probabilities, conditional


def with_grouped_common_state_erasures(
    models: Sequence[TargetEvidenceModel],
    strength: float,
    failure_groups: Sequence[np.ndarray] | np.ndarray,
) -> list[TargetEvidenceModel]:
    if isinstance(failure_groups, np.ndarray) and failure_groups.ndim == 1:
        groups_per_model = [failure_groups for _ in models]
    else:
        groups_per_model = list(failure_groups)
    if len(groups_per_model) != len(models):
        raise ValueError("failure_groups must provide labels for every model")
    result = []
    for model, groups in zip(models, groups_per_model):
        groups = np.asarray(groups, dtype=int)
        if groups.shape != (model.num_uavs,):
            raise ValueError("each failure-group vector must match model.num_uavs")
        if len(set(groups[i] for i in range(model.num_uavs) if i != model.owner)) < 2:
            raise ValueError("each model must contain at least two reporting failure groups")
        state_probabilities, conditional = grouped_common_state_parameters(
            model.success_prob, model.owner, groups, strength
        )
        # Materialize full patterns only for validation and diagnostics.
        patterns = np.asarray(list(product((0, 1), repeat=model.num_uavs)), dtype=np.int8)
        probabilities = np.zeros(patterns.shape[0])
        for state_weight, state_success in zip(state_probabilities, conditional):
            probabilities += state_weight * np.prod(
                np.where(patterns == 1, state_success[None, :], 1.0 - state_success[None, :]),
                axis=1,
            )
        positive = probabilities > 1e-15
        grouped = replace(
            model,
            reception_patterns=patterns[positive],
            pattern_probabilities=probabilities[positive] / probabilities[positive].sum(),
            reception_state_probabilities=state_probabilities,
            conditional_success_probabilities=conditional,
        )
        grouped.validate(); result.append(grouped)
    return result


def alternating_failure_groups(models: Sequence[TargetEvidenceModel], num_groups: int = 2) -> list[np.ndarray]:
    if num_groups < 2:
        raise ValueError("num_groups must be at least two")
    return [np.arange(model.num_uavs, dtype=int) % num_groups for model in models]


def _deterministic_kmeans(features: np.ndarray, num_groups: int) -> np.ndarray:
    """Small dependency-free clustering fixed entirely by physical features."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] < num_groups:
        raise ValueError("features must contain at least num_groups rows")
    if num_groups < 2:
        raise ValueError("num_groups must be at least two")
    centroid = features.mean(axis=0)
    center_indices = [int(np.argmax(np.linalg.norm(features - centroid, axis=1)))]
    while len(center_indices) < num_groups:
        distance = np.min(
            np.stack([
                np.linalg.norm(features - features[index], axis=1)
                for index in center_indices
            ]),
            axis=0,
        )
        distance[center_indices] = -1.0
        center_indices.append(int(np.argmax(distance)))
    centers = features[center_indices].copy()
    labels = np.zeros(features.shape[0], dtype=int)
    for _ in range(100):
        new_labels = np.argmin(
            np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2),
            axis=1,
        )
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for group in range(num_groups):
            members = features[labels == group]
            if members.size:
                centers[group] = members.mean(axis=0)
    if len(set(labels.tolist())) != num_groups:
        raise RuntimeError("physical-feature clustering produced an empty group")
    return labels


def physical_failure_groups(
    models: Sequence[TargetEvidenceModel],
    positions: np.ndarray,
    scheme: str,
    num_groups: int = 2,
) -> list[np.ndarray]:
    """Generate target-aware proxy failure domains from geometry only.

    Schemes never inspect sensing quality, link success probability, or an
    optimizer result.  Therefore the grouping cannot be tuned retrospectively
    to improve scheduling performance.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [num_uavs, 3]")
    result = []
    for model in models:
        if positions.shape[0] != model.num_uavs:
            raise ValueError("positions must match model.num_uavs")
        candidates = [i for i in range(model.num_uavs) if i != model.owner]
        owner_position = positions[model.owner]
        if scheme == "formation_position":
            features = positions[candidates]
        elif scheme == "link_midpoint":
            features = (positions[candidates] + owner_position) / 2.0
        elif scheme == "owner_angle_path":
            link = positions[candidates] - owner_position
            distance = np.linalg.norm(link, axis=1)
            horizontal = np.linalg.norm(link[:, :2], axis=1)
            azimuth = np.arctan2(link[:, 1], link[:, 0])
            elevation = np.arctan2(link[:, 2], horizontal)
            features = np.column_stack((
                np.cos(azimuth), np.sin(azimuth), elevation,
                distance / max(float(distance.max()), 1e-12),
            ))
        else:
            raise ValueError(
                "scheme must be 'formation_position', 'link_midpoint', or "
                "'owner_angle_path'"
            )
        candidate_labels = _deterministic_kmeans(features, num_groups)
        labels = np.full(model.num_uavs, -1, dtype=int)
        labels[candidates] = candidate_labels
        result.append(labels)
    return result


def grouped_failure_correlation(
    model: TargetEvidenceModel, groups: Sequence[int]
) -> tuple[float, float]:
    """Return mean within-domain and between-domain failure correlation."""
    if model.reception_patterns is None:
        return 0.0, 0.0
    labels = np.asarray(groups, dtype=int)
    if labels.shape != (model.num_uavs,):
        raise ValueError("groups must have one entry per UAV")
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    patterns = np.asarray(model.reception_patterns)[:, candidates]
    probabilities = np.asarray(model.pattern_probabilities)
    failures = 1.0 - patterns
    means = probabilities @ failures
    centered = failures - means
    covariance = (centered * probabilities[:, None]).T @ centered
    variance = np.diag(covariance)
    within = []; between = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            denominator = np.sqrt(variance[left] * variance[right])
            if denominator <= 1e-14:
                continue
            correlation = covariance[left, right] / denominator
            target = (
                within if labels[candidates[left]] == labels[candidates[right]]
                else between
            )
            target.append(correlation)
    return (
        float(np.mean(within)) if within else 0.0,
        float(np.mean(between)) if between else 0.0,
    )


def mean_off_diagonal_failure_correlation(model: TargetEvidenceModel) -> float:
    if model.reception_patterns is None:
        return 0.0
    candidates = [i for i in range(model.num_uavs) if i != model.owner]
    if len(candidates) < 2:
        return 0.0
    patterns = np.asarray(model.reception_patterns)[:, candidates]
    probabilities = np.asarray(model.pattern_probabilities)
    failures = 1.0 - patterns
    means = probabilities @ failures
    centered = failures - means
    covariance = (centered * probabilities[:, None]).T @ centered
    variance = np.diag(covariance)
    correlations = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            denominator = np.sqrt(variance[i] * variance[j])
            if denominator > 1e-14:
                correlations.append(covariance[i, j] / denominator)
    return float(np.mean(correlations)) if correlations else 0.0
