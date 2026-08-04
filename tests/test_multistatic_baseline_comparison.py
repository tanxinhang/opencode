from scripts.run_multistatic_baseline_comparison import run_comparison


def test_paired_comparison_contains_all_unknown_n_methods():
    result = run_comparison(trials=2, seed=7, transmitters=4, targets=2)
    methods = {row["method"] for row in result["summaries"]}
    assert methods == {
        "position_dbscan", "angle_position_dbscan",
        "identity_dbscan", "gated_identity_dbscan",
        "conflict_aware_dbscan", "bic_conflict",
        "geometry_doppler",
    }
    assert result["trials"] == 2
    for comparison in result["paired_proposed_bic_minus_baseline"].values():
        assert comparison["proposed_wins"] + comparison["baseline_wins"] + comparison["ties"] == 2
