import contextlib
import io
import os
import shutil
import zipfile

import matplotlib.pyplot as plt
import pooch
import pytest

OPEN_FIELD_2D_URL = (
    "https://zenodo.org/api/records/10636983/files/open_field_2D.zip/content"
)
OPEN_FIELD_2D_HASH = "md5:22ec317b33039875cb21c76dca027017"
OPEN_FIELD_2D_DLC_PROJECT = "22_12_11_pipeline_dev/keypoint-moseq/testing/dlc_project"


@pytest.fixture(scope="session")
def dlc_project_dir():
    """DeepLabCut project from the tutorial's 2D open field dataset (Zenodo
    record 10636983), downloaded and unpacked once per machine."""
    archive = pooch.retrieve(
        OPEN_FIELD_2D_URL,
        known_hash=OPEN_FIELD_2D_HASH,
        fname="open_field_2D.zip",
        path=pooch.os_cache("keypoint-moseq"),
        progressbar=False,
    )
    # The archive stores every file under "../", which zipfile drops on
    # extraction. pooch.Unzip looks for the files at their "../" paths, never
    # finds them, and extracts again on every run, so extract here instead.
    extract_dir = archive + ".unzip"
    extracted_marker = os.path.join(extract_dir, ".extracted")
    if not os.path.exists(extracted_marker):
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extractall(extract_dir)
        open(extracted_marker, "w").close()
    return os.path.join(extract_dir, OPEN_FIELD_2D_DLC_PROJECT)


@pytest.fixture(scope="session")
def tutorial_config_edits():
    """Keyword arguments of the modeling notebook's `update_config` call for
    the example DeepLabCut dataset."""
    return dict(
        video_dir="dlc_project/videos/",
        anterior_bodyparts=["nose"],
        posterior_bodyparts=["spine4"],
        use_bodyparts=[
            "spine4",
            "spine3",
            "spine2",
            "spine1",
            "head",
            "nose",
            "right ear",
            "left ear",
        ],
        fps=30,
    )


def setup_tutorial_project(project_dir, dlc_project_dir, config_edits):
    """Set up a project from the example DeepLabCut config and edit it as in
    the modeling notebook."""
    import keypoint_moseq as kpms

    dlc_config = os.path.join(dlc_project_dir, "config.yaml")
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config)
    kpms.update_config(project_dir, **config_edits)


@pytest.fixture
def tutorial_project_dir(tmp_path, dlc_project_dir, tutorial_config_edits):
    """Project set up from the example DeepLabCut config and edited as in the
    modeling notebook."""
    project_dir = str(tmp_path / "demo_project")
    setup_tutorial_project(project_dir, dlc_project_dir, tutorial_config_edits)
    return project_dir


@pytest.fixture(scope="session")
def snapshot_recording():
    """Recording whose arrays are snapshotted; the shortest in the example
    dataset."""
    return "21_12_10_def6a_3.top.irDLC_resnet50_moseq_exampleAug21shuffle1_500000"


@pytest.fixture(scope="session")
def tutorial_keypoints(dlc_project_dir):
    """Coordinates, confidences and bodyparts loaded the way the modeling
    notebook loads the example DeepLabCut dataset."""
    import keypoint_moseq as kpms

    keypoint_data_path = os.path.join(dlc_project_dir, "videos")
    return kpms.load_keypoints(keypoint_data_path, "deeplabcut")


@pytest.fixture(scope="session")
def tutorial_outlier_removed(
    tmp_path_factory, dlc_project_dir, tutorial_config_edits, tutorial_keypoints
):
    """Project directory, coordinates and confidences after the modeling
    notebook's outlier removal cell, run on every recording in the example
    dataset."""
    import keypoint_moseq as kpms

    project_dir = str(tmp_path_factory.mktemp("outliers") / "demo_project")
    setup_tutorial_project(project_dir, dlc_project_dir, tutorial_config_edits)
    config = lambda: kpms.load_config(project_dir)
    coordinates, confidences, _ = tutorial_keypoints

    kpms.update_config(project_dir, outlier_scale_factor=6.0)
    coordinates, confidences = kpms.outlier_removal(
        dict(coordinates), dict(confidences), project_dir, overwrite=False, **config()
    )
    return project_dir, coordinates, confidences


@pytest.fixture(scope="session")
def tutorial_formatted_data(tutorial_outlier_removed):
    """Data and metadata from the modeling notebook's format_data cell, run on
    every recording in the example dataset."""
    import keypoint_moseq as kpms

    project_dir, coordinates, confidences = tutorial_outlier_removed
    config = lambda: kpms.load_config(project_dir)
    return kpms.format_data(dict(coordinates), dict(confidences), **config())


@pytest.fixture(scope="session")
def tutorial_pca(tutorial_outlier_removed, tutorial_formatted_data):
    """PCA and the printed variance summary from the modeling notebook's PCA
    cell, run in the shared tutorial project, followed by the notebook's
    `latent_dim` edit."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    data, _ = tutorial_formatted_data
    config = lambda: kpms.load_config(project_dir)

    pca = kpms.fit_pca(**data, **config())
    kpms.save_pca(pca, project_dir)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        kpms.print_dims_to_explain_variance(pca, 0.9)
    kpms.plot_scree(pca, project_dir=project_dir)
    kpms.plot_pcs(pca, project_dir=project_dir, **config())
    plt.close("all")

    kpms.update_config(project_dir, latent_dim=4)
    return pca, printed.getvalue()


@pytest.fixture(scope="session")
def tutorial_sigmasq_loc(tutorial_outlier_removed, tutorial_formatted_data, tutorial_pca):
    """`sigmasq_loc` estimated by the modeling notebook's hyperparameter cell
    and written to the shared tutorial project's config."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    data, _ = tutorial_formatted_data
    config = lambda: kpms.load_config(project_dir)

    sigmasq_loc = kpms.estimate_sigmasq_loc(
        data["Y"], data["mask"], filter_size=config()["fps"]
    )
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)
    return sigmasq_loc


@pytest.fixture(scope="session")
def tutorial_model(
    tutorial_outlier_removed, tutorial_formatted_data, tutorial_pca, tutorial_sigmasq_loc
):
    """Model from the modeling notebook's initialization cell."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    data, _ = tutorial_formatted_data
    pca, _ = tutorial_pca
    config = lambda: kpms.load_config(project_dir)
    return kpms.init_model(data, pca=pca, **config())


@pytest.fixture(scope="session")
def tutorial_arhmm(tutorial_outlier_removed, tutorial_formatted_data, tutorial_model):
    """Model, model name and iteration count from the modeling notebook's
    AR-HMM cell, run for 5 iterations instead of the notebook's 50."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    data, metadata = tutorial_formatted_data
    num_ar_iters = 5

    model, model_name = kpms.fit_model(
        tutorial_model, data, metadata, project_dir, ar_only=True, num_iters=num_ar_iters
    )
    return model, model_name, num_ar_iters


@pytest.fixture(scope="session")
def tutorial_full_model(tutorial_outlier_removed, tutorial_arhmm):
    """Model and final iteration from the modeling notebook's full-model cell:
    the AR-HMM checkpoint reloaded, kappa set to 1e4, and fit for 5 iterations
    instead of the notebook's 500."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    _, model_name, num_ar_iters = tutorial_arhmm
    num_full_iters = 5

    model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir, model_name, iteration=num_ar_iters
    )
    model = kpms.update_hypparams(model, kappa=1e4)
    model = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=False,
        start_iter=current_iter,
        num_iters=current_iter + num_full_iters,
    )[0]
    return model, current_iter + num_full_iters


@pytest.fixture(scope="session")
def tutorial_reindexed(tutorial_outlier_removed, tutorial_arhmm, tutorial_full_model):
    """Model name and permutation from the modeling notebook's reindexing cell,
    run on a copy of the fitted model directory. Reindexing rewrites the
    checkpoint in place, so the copy leaves the fitted checkpoint unchanged
    for the tests that compare it with the in-memory models."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    _, fitted_model_name, _ = tutorial_arhmm
    model_name = f"{fitted_model_name}-reindexed"
    shutil.copytree(
        os.path.join(project_dir, fitted_model_name),
        os.path.join(project_dir, model_name),
    )

    index = kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
    return model_name, index


@pytest.fixture(scope="session")
def tutorial_results(tutorial_outlier_removed, tutorial_reindexed):
    """Model, metadata and results from the modeling notebook's extraction
    cells, which write `results.h5` and per-recording csv files for the
    reindexed model."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    model_name, _ = tutorial_reindexed

    model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
    results = kpms.extract_results(model, metadata, project_dir, model_name)
    kpms.save_results_as_csv(results, project_dir, model_name)
    return model, metadata, results


@pytest.fixture(scope="session")
def tutorial_applied(
    tmp_path_factory,
    dlc_project_dir,
    snapshot_recording,
    tutorial_outlier_removed,
    tutorial_reindexed,
    tutorial_results,
):
    """Outputs of the modeling notebook's apply-to-new-data cell, run with 5
    iterations instead of the default 500 on a copy of the reindexed model
    directory, so the extracted results stay unchanged for their own tests.
    The new data is a copy of one tutorial recording saved under a new name.
    `return_model=True` is added to the notebook's call so the tests can
    compare the applied model's parameters with the trained ones."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    reindexed_model_name, _ = tutorial_reindexed
    model_name = f"{reindexed_model_name}-applied"
    shutil.copytree(
        os.path.join(project_dir, reindexed_model_name),
        os.path.join(project_dir, model_name),
    )
    new_data = str(tmp_path_factory.mktemp("new_data"))
    shutil.copy(
        os.path.join(dlc_project_dir, "videos", f"{snapshot_recording}.h5"),
        os.path.join(new_data, f"new_{snapshot_recording}.h5"),
    )
    config = lambda: kpms.load_config(project_dir)

    model = kpms.load_checkpoint(project_dir, model_name)[0]
    coordinates, confidences, bodyparts = kpms.load_keypoints(new_data, "deeplabcut")
    coordinates, confidences = kpms.outlier_removal(
        coordinates, confidences, project_dir, overwrite=False, **config()
    )
    data, metadata = kpms.format_data(coordinates, confidences, **config())
    results, applied_model = kpms.apply_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        num_iters=5,
        return_model=True,
        **config(),
    )
    return {
        "model_name": model_name,
        "model": model,
        "applied_model": applied_model,
        "coordinates": coordinates,
        "results": results,
    }


@pytest.fixture(scope="session")
def tutorial_trajectory_plots(tutorial_outlier_removed, tutorial_reindexed, tutorial_results):
    """Results and output directory of the modeling notebook's trajectory
    plot cell, run on the reindexed model."""
    import keypoint_moseq as kpms

    project_dir, coordinates, _ = tutorial_outlier_removed
    model_name, _ = tutorial_reindexed
    config = lambda: kpms.load_config(project_dir)

    results = kpms.load_results(project_dir, model_name)
    kpms.generate_trajectory_plots(
        coordinates, results, project_dir, model_name, **config()
    )
    plt.close("all")
    return results, os.path.join(project_dir, model_name, "trajectory_plots")


@pytest.fixture(scope="session")
def tutorial_grid_movies(
    dlc_project_dir, tutorial_outlier_removed, tutorial_reindexed, tutorial_results
):
    """Results, sampled instances and output directory of the modeling
    notebook's grid movie cell, run on the reindexed model with
    `min_frequency=0.06` added so that only the few most frequent syllables
    get movies. It runs from the directory containing `dlc_project` so the
    config's relative `video_dir` resolves the way it does in the notebook."""
    import keypoint_moseq as kpms

    project_dir, coordinates, _ = tutorial_outlier_removed
    model_name, _ = tutorial_reindexed
    config = lambda: kpms.load_config(project_dir)
    results = kpms.load_results(project_dir, model_name)

    working_dir = os.getcwd()
    os.chdir(os.path.dirname(dlc_project_dir))
    try:
        sampled_instances = kpms.generate_grid_movies(
            results,
            project_dir,
            model_name,
            coordinates=coordinates,
            min_frequency=0.06,
            **config(),
        )
    finally:
        os.chdir(working_dir)
    return results, sampled_instances, os.path.join(project_dir, model_name, "grid_movies")


@pytest.fixture(scope="session")
def tutorial_dendrogram(tutorial_outlier_removed, tutorial_reindexed, tutorial_results):
    """Results, leaf labels and save path (without extension) of the
    modeling notebook's similarity dendrogram cell, run on the reindexed
    model."""
    import keypoint_moseq as kpms

    project_dir, coordinates, _ = tutorial_outlier_removed
    model_name, _ = tutorial_reindexed
    config = lambda: kpms.load_config(project_dir)
    results = kpms.load_results(project_dir, model_name)

    kpms.plot_similarity_dendrogram(
        coordinates, results, project_dir, model_name, **config()
    )
    labels = [label.get_text() for label in plt.gcf().axes[0].get_xticklabels()]
    plt.close("all")
    return results, labels, os.path.join(project_dir, model_name, "similarity_dendrogram")
