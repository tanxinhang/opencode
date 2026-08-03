from scripts.run_dd_spatial_gate import local_maxima, spatial_resolution_case


def test_local_maxima_excludes_edges_and_plateau_duplicates():
    assert local_maxima([5.0, 1.0, 3.0, 3.0, 1.0, 5.0]) == [2]


def test_large_angle_gap_is_resolved_in_spatial_cube():
    result = spatial_resolution_case(40.0)
    assert result["resolved"]
    assert result["two_dimensional_peak_count"] == 1


def test_shallow_ten_degree_split_is_not_counted_as_resolved():
    result = spatial_resolution_case(10.0)
    assert not result["resolved"]
    assert result["valley_to_weaker_peak_ratio"] > 0.8


def test_spatial_code_needs_identity_contrast_not_only_angle_tolerance():
    from uav_otfs_isac.otfs_physical import cazac_sequence
    result = spatial_resolution_case(
        2.0, [cazac_sequence(8, 1), cazac_sequence(8, 3)]
    )
    assert not result["resolved"]
    assert result["mean_desired_to_other_angle_ratio"] < 2.0
