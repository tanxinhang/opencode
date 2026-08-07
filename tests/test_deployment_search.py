import numpy as np

from uav_otfs_isac.deployment_search import (
    estimate_coordinate_lipschitz,
    estimate_lipschitz,
    grid_search_suboptimality_bound,
    lipschitz_adaptive_search,
)


def test_lipschitz_constant_of_distance_function():
    center = np.array([0.0, 0.0])
    positions = np.array([
        [-10.0, -10.0], [-10.0, 0.0], [0.0, 0.0],
        [10.0, 0.0], [10.0, 10.0], [-10.0, 10.0],
    ])
    values = np.linalg.norm(positions - center, axis=1)
    assert np.isclose(estimate_lipschitz(values, positions), 1.0, atol=1e-12)


def test_grid_bound_holds_for_distance_function():
    center = np.array([0.0, 0.0])
    spacing = 2.0
    axis = np.arange(-10.0, 10.01, spacing)
    grid = np.array([[x, y] for x in axis for y in axis])
    values = np.linalg.norm(grid - center, axis=1)
    true_max = float(np.max(np.linalg.norm(
        np.array([[-10.0, -10.0], [10.0, 10.0]]) - center, axis=1
    )))
    grid_max = float(np.max(values))
    bound = grid_search_suboptimality_bound(1.0, spacing, 2)
    assert true_max - grid_max <= bound + 1e-12
    assert np.isclose(bound, spacing * np.sqrt(2.0) / 2.0)


def test_bound_and_estimate_validate_inputs():
    assert grid_search_suboptimality_bound(0.0, 2.0, 3) == 0.0
    assert estimate_lipschitz(np.array([1.0]), np.zeros((1, 2))) == 0.0
    for invalid in (
        lambda: grid_search_suboptimality_bound(-1.0, 2.0, 3),
        lambda: grid_search_suboptimality_bound(1.0, -1.0, 3),
        lambda: grid_search_suboptimality_bound(1.0, 2.0, 0),
    ):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_coordinate_lipschitz_matches_separable_function():
    positions = np.array([
        [-2.0, 0.0],
        [0.0, 0.0],
        [-2.0, -2.0],
        [0.0, -2.0],
    ])
    values = np.array([3.0, 0.0, 4.0, 1.0])
    estimate = estimate_coordinate_lipschitz(values, positions)
    assert np.allclose(estimate, [1.5, 0.5])


def test_adaptive_search_epsilon_optimal_for_distance_function():
    center = np.array([0.0, 0.0])
    bounds = np.array([[-10.0, 10.0], [-10.0, 10.0]])

    def objective(point):
        return float(np.linalg.norm(np.asarray(point) - center))

    result = lipschitz_adaptive_search(
        objective, bounds, lipschitz=1.0, epsilon=0.1,
        max_evaluations=500,
    )
    true_max = float(np.sqrt(2.0) * 10.0)
    assert true_max - result["best_value"] <= 0.1 + 1e-12
    assert result["certificate_gap"] <= 0.1 + 1e-12
    assert result["evaluations"] < 500
    assert not result["terminated_by_budget"]


def test_adaptive_search_with_coordinate_lipschitz():
    bounds = np.array([[-10.0, 10.0], [-10.0, 10.0]])

    def objective(point):
        point = np.asarray(point)
        return 1.5 * abs(point[0]) + 0.5 * abs(point[1])

    result = lipschitz_adaptive_search(
        objective, bounds, lipschitz=1.5, epsilon=0.1,
        max_evaluations=500, coordinate_lipschitz=np.array([1.5, 0.5]),
    )
    true_max = 20.0
    assert true_max - result["best_value"] <= 0.1 + 1e-12
    assert result["certificate_gap"] <= 0.1 + 1e-12
    assert result["evaluations"] < 500


def test_adaptive_search_can_return_boxes():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    result = lipschitz_adaptive_search(
        lambda x: float(np.sum(np.asarray(x))),
        bounds,
        lipschitz=1.0,
        epsilon=0.5,
        max_evaluations=5,
        return_boxes=True,
    )
    assert "boxes" in result
    assert len(result["boxes"]) >= 1


def test_adaptive_search_validates_inputs():
    for invalid in (
        lambda: lipschitz_adaptive_search(
            lambda x: 0.0, np.zeros((2, 3)), 1.0, 0.1
        ),
        lambda: lipschitz_adaptive_search(
            lambda x: 0.0, np.array([[1.0, 0.0]]), 1.0, 0.1
        ),
        lambda: lipschitz_adaptive_search(
            lambda x: 0.0, np.array([[0.0, 1.0]]), -1.0, 0.1
        ),
        lambda: lipschitz_adaptive_search(
            lambda x: 0.0, np.array([[0.0, 1.0]]), 1.0, 0.1,
            coordinate_lipschitz=np.array([1.0, 2.0]),
        ),
    ):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
