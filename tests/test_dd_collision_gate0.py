from scripts.run_dd_collision_gate0 import cyclic_peak_errors


def test_peak_matching_is_one_to_one_and_cyclic():
    errors = cyclic_peak_errors(
        [(0, 0), (2, 4)], [(2, 4), (7, 15)], (8, 16)
    )
    assert errors == [(1, 1), (0, 0)]
