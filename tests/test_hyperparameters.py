import keypoint_moseq as kpms


def test_tutorial_sigmasq_loc(
    tutorial_sigmasq_loc, tutorial_outlier_removed, num_regression
):
    project_dir, _, _ = tutorial_outlier_removed
    config = lambda: kpms.load_config(project_dir)

    assert config()["cen_hypparams"]["sigmasq_loc"] == tutorial_sigmasq_loc
    num_regression.check(
        {"sigmasq_loc": [tutorial_sigmasq_loc]},
        default_tolerance=dict(rtol=1e-6),
    )
