"""Grid-search suboptimality bounds for deployment optimization.

Let ``f : R^d -> R`` be ``L``-Lipschitz and let ``G_h`` be a grid with
spacing ``h`` in every coordinate.  Every point lies within ``l_inf``
distance ``h / 2`` of a grid point, hence within Euclidean distance
``h sqrt(d) / 2``.  Lipschitz continuity then gives the deployment-loss bound

``max_x f(x) - max_{g in G_h} f(g) <= L h sqrt(d) / 2``.

The module also estimates an empirical Lipschitz constant from evaluated
deployments, which is a valid (possibly loose) substitute for ``L`` when the
objective is not analytically available.
"""

from __future__ import annotations

import numpy as np


def grid_search_suboptimality_bound(
    lipschitz: float,
    spacing: float,
    dimension: int,
) -> float:
    """Return ``L h sqrt(d) / 2`` for an ``L``-Lipschitz deployment objective."""
    if lipschitz < 0.0:
        raise ValueError("lipschitz must be nonnegative")
    if spacing < 0.0:
        raise ValueError("spacing must be nonnegative")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return float(lipschitz * spacing * np.sqrt(dimension) / 2.0)


def estimate_lipschitz(
    values: np.ndarray,
    positions: np.ndarray,
) -> float:
    """Empirical Lipschitz constant over all evaluated deployment pairs."""
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if values.ndim != 1 or positions.ndim != 2:
        raise ValueError("values must be 1-D and positions must be 2-D")
    if positions.shape[0] != values.size:
        raise ValueError("one value is required per deployment position")
    if values.size < 2:
        return 0.0
    maximum = 0.0
    for left in range(values.size):
        for right in range(left + 1, values.size):
            distance = float(np.linalg.norm(
                positions[left] - positions[right]
            ))
            if distance == 0.0:
                continue
            maximum = max(
                maximum,
                abs(values[left] - values[right]) / distance,
            )
    return float(maximum)


def estimate_coordinate_lipschitz(
    values: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Empirical per-coordinate Lipschitz constants over evaluated pairs.

    For each coordinate ``j``, the returned constant is the maximum of
    ``|f(x) - f(y)| / |x_j - y_j|`` over evaluated pairs that differ only in
    coordinate ``j``.  Pairs with simultaneous multi-coordinate changes are
    excluded because they mix several directional slopes.  Like
    :func:`estimate_lipschitz`, the result is an empirical constant, not a
    proven global one; the search doubles it as a safety factor before using
    it in a certificate.
    """
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if values.ndim != 1 or positions.ndim != 2:
        raise ValueError("values must be 1-D and positions must be 2-D")
    if positions.shape[0] != values.size:
        raise ValueError("one value is required per deployment position")
    dimension = positions.shape[1]
    result = np.zeros(dimension, dtype=float)
    if values.size < 2:
        return result
    for left in range(values.size):
        for right in range(left + 1, values.size):
            value_difference = abs(values[left] - values[right])
            coordinate_difference = np.abs(
                positions[left] - positions[right]
            )
            differing = coordinate_difference > 1e-12
            if np.count_nonzero(differing) != 1:
                continue
            j = int(np.argmax(differing))
            result[j] = max(
                result[j], value_difference / coordinate_difference[j]
            )
    return result


def lipschitz_adaptive_search(
    objective,
    bounds,
    lipschitz: float,
    epsilon: float,
    max_evaluations: int = 500,
    coordinate_lipschitz: np.ndarray | None = None,
    return_boxes: bool = False,
) -> dict:
    """Piyavskii-style Lipschitz branch-and-bound search.

Each box stores its evaluated center ``c``, half-side vector ``h`` and
upper bound ``min(f(c) + L ||h||_2, f(c) + sum_j L_j h_j)`` when positive
coordinate-wise Lipschitz constants ``L_j`` are available.  The box with the
largest upper bound is split along its longest axis, and both child centers
are evaluated.
    The loop terminates when ``global_upper - best <= epsilon``, which is an
    epsilon-optimality certificate whenever ``L`` is a valid Lipschitz
    constant for the objective over the box.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (dimension, 2)")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be below its upper bound")
    if lipschitz < 0.0:
        raise ValueError("lipschitz must be nonnegative")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    if coordinate_lipschitz is not None:
        coordinate_lipschitz = np.asarray(
            coordinate_lipschitz, dtype=float
        )
        if coordinate_lipschitz.shape != (bounds.shape[0],):
            raise ValueError("coordinate_lipschitz must match bounds")
        if np.any(coordinate_lipschitz < 0.0):
            raise ValueError("coordinate_lipschitz entries must be nonnegative")

    center = 0.5 * (bounds[:, 0] + bounds[:, 1])
    half = 0.5 * (bounds[:, 1] - bounds[:, 0])
    value = float(objective(center))
    radius = float(np.linalg.norm(half))
    best = value
    best_point = center.copy()
    radial_upper = value + lipschitz * radius
    coordinate_upper = radial_upper
    if coordinate_lipschitz is not None and np.all(coordinate_lipschitz > 0.0):
        coordinate_upper = value + float(np.sum(
            coordinate_lipschitz * half
        ))
    boxes = [{
        "center": center,
        "half": half,
        "value": value,
        "upper": min(radial_upper, coordinate_upper),
    }]
    evaluations = 1
    global_upper = boxes[0]["upper"]

    while global_upper - best > epsilon and evaluations < max_evaluations:
        boxes.sort(key=lambda box: box["upper"], reverse=True)
        box = boxes.pop(0)
        axis = int(np.argmax(box["half"]))
        half = box["half"].copy()
        children = []
        for sign in (-1.0, 1.0):
            child_center = box["center"].copy()
            child_center[axis] += sign * half[axis] / 2.0
            child_half = half.copy()
            child_half[axis] /= 2.0
            child_value = float(objective(child_center))
            evaluations += 1
            child_radius = float(np.linalg.norm(child_half))
            radial_upper = child_value + lipschitz * child_radius
            coordinate_upper = radial_upper
            if (
                coordinate_lipschitz is not None
                and np.all(coordinate_lipschitz > 0.0)
            ):
                coordinate_upper = child_value + float(np.sum(
                    coordinate_lipschitz * child_half
                ))
            children.append({
                "center": child_center,
                "half": child_half,
                "value": child_value,
                "upper": min(radial_upper, coordinate_upper),
            })
            if child_value > best:
                best = child_value
                best_point = child_center.copy()
        boxes.extend(children)
        global_upper = max(box["upper"] for box in boxes)

    payload = {
        "best_point": best_point,
        "best_value": float(best),
        "global_upper": float(global_upper),
        "evaluations": int(evaluations),
        "certificate_gap": float(global_upper - best),
        "terminated_by_budget": bool(evaluations >= max_evaluations),
    }
    if return_boxes:
        payload["boxes"] = boxes
    return payload
