import os

import numpy as np

import keypoint_moseq as kpms

PDF_SIGNATURE = b"%PDF"


def test_tutorial_pca_variances_and_mean(tutorial_pca, ndarrays_regression):
    pca, _ = tutorial_pca

    ndarrays_regression.check(
        {
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "mean": pca.mean_,
        },
        default_tolerance=dict(rtol=1e-6),
        tolerances={"mean": dict(rtol=1e-6, atol=1e-6)},
    )


def test_tutorial_pca_saved(tutorial_pca, tutorial_outlier_removed):
    pca, _ = tutorial_pca
    project_dir, _, _ = tutorial_outlier_removed

    saved_pca = kpms.load_pca(project_dir)

    np.testing.assert_array_equal(
        saved_pca.explained_variance_, pca.explained_variance_
    )
    np.testing.assert_array_equal(saved_pca.mean_, pca.mean_)


def test_tutorial_pca_dims_match_notebook_latent_dim(
    tutorial_pca, tutorial_outlier_removed
):
    _, printed = tutorial_pca
    project_dir, _, _ = tutorial_outlier_removed
    config = lambda: kpms.load_config(project_dir)

    assert printed == ">=90.0% of variance exlained by 4 components.\n"
    assert config()["ar_hypparams"]["latent_dim"] == 4


def test_tutorial_pca_plots(tutorial_pca, tutorial_outlier_removed):
    project_dir, _, _ = tutorial_outlier_removed

    for filename in ["pca_scree.pdf", "pcs-xy.pdf"]:
        with open(os.path.join(project_dir, filename), "rb") as f:
            assert f.read(len(PDF_SIGNATURE)) == PDF_SIGNATURE
