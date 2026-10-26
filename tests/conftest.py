import contextlib
import io
import math
import os
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import pytest

from analysis_utils import (
    bigram_plot_summary,
    cohort_group,
    graph_plot_summary,
    plot_syll_stats,
    read_syll_info,
    syll_stats_plot_summary,
    syllable_description,
    syllable_label,
)

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
    data, metadata = tutorial_formatted_data
    pca, _ = tutorial_pca
    config = lambda: kpms.load_config(project_dir)
    return kpms.init_model(data, metadata=metadata, pca=pca, **config())


@pytest.fixture(scope="session")
def one_recording(tutorial_outlier_removed, tutorial_pca, tutorial_sigmasq_loc):
    """Config, data and metadata for one tutorial recording, and a
    keypoint-SLDS and a location-aware model initialized on it, keyed by
    `location_aware`."""
    import keypoint_moseq as kpms

    project_dir, coordinates, confidences = tutorial_outlier_removed
    pca, _ = tutorial_pca
    config = kpms.load_config(project_dir)
    name = sorted(coordinates)[0]
    data, metadata = kpms.format_data(
        {name: coordinates[name]}, {name: confidences[name]}, **config
    )
    models = {
        location_aware: kpms.init_model(
            data, metadata=metadata, pca=pca, location_aware=location_aware, **config
        )
        for location_aware in [False, True]
    }
    return config, data, metadata, models


@pytest.fixture(scope="session")
def tutorial_arhmm(tutorial_outlier_removed, tutorial_formatted_data, tutorial_model):
    """Model, model name and iteration count from the modeling notebook's
    AR-HMM cell, run for 5 iterations instead of the notebook's 50."""
    import keypoint_moseq as kpms

    project_dir, _, _ = tutorial_outlier_removed
    data, metadata = tutorial_formatted_data
    config = lambda: kpms.load_config(project_dir)
    num_ar_iters = 5

    model, model_name = kpms.fit_model(
        tutorial_model,
        data,
        metadata,
        project_dir,
        ar_only=True,
        num_iters=num_ar_iters,
        **config(),
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
    config = lambda: kpms.load_config(project_dir)
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
        **config(),
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


FROZEN_STATS_DF = os.path.join(os.path.dirname(__file__), "data", "frozen_stats_df.csv")


@pytest.fixture(scope="session")
def tutorial_analysis_project(
    tmp_path_factory,
    tutorial_outlier_removed,
    tutorial_reindexed,
    tutorial_results,
    tutorial_grid_movies,
):
    """Project directory and model name for the analysis notebook: a new
    project holding copies of the modeling project's config and the reindexed
    model's `results.h5` and grid movies, laid out as the notebook's setup
    cell describes. The analysis
    notebook writes `index.csv` to the project directory and `syll_info.csv`
    and figures to the model directory, so the copy keeps them out of the
    modeling project."""
    modeling_project_dir, _, _ = tutorial_outlier_removed
    model_name, _ = tutorial_reindexed
    source_dir = os.path.join(modeling_project_dir, model_name)
    project_dir = str(tmp_path_factory.mktemp("analysis") / "demo_project")
    model_dir = os.path.join(project_dir, model_name)

    os.makedirs(model_dir)
    shutil.copy(os.path.join(modeling_project_dir, "config.yml"), project_dir)
    shutil.copy(os.path.join(source_dir, "results.h5"), model_dir)
    shutil.copytree(
        os.path.join(source_dir, "grid_movies"), os.path.join(model_dir, "grid_movies")
    )
    return project_dir, model_name


@pytest.fixture
def analysis_project_without_groups(tmp_path, tutorial_analysis_project):
    """Project directory and model name holding only copies of the analysis
    project's config and `results.h5`, as before the group-setting cell has
    run."""
    source_project_dir, model_name = tutorial_analysis_project
    project_dir = str(tmp_path / "demo_project")
    model_dir = os.path.join(project_dir, model_name)

    os.makedirs(model_dir)
    shutil.copy(os.path.join(source_project_dir, "config.yml"), project_dir)
    shutil.copy(os.path.join(source_project_dir, model_name, "results.h5"), model_dir)
    return project_dir, model_name


@pytest.fixture(scope="session")
def tutorial_groups_assigned(tutorial_analysis_project):
    """Widget and `index.csv` contents (as generated and as saved) of the
    analysis notebook's group-setting cell. Each recording's group cell is
    edited to its cohort group, typed with surrounding spaces and a
    newline, and "Save group info" is clicked."""
    import keypoint_moseq as kpms

    project_dir, model_name = tutorial_analysis_project
    index_path = os.path.join(project_dir, "index.csv")

    widget = kpms.interactive_group_setting(project_dir, model_name)
    initial_index = pd.read_csv(index_path)
    table, button = widget[0], widget[1][0]
    table.patch(
        {
            "group": [
                (row, f"  {cohort_group(name)}\n")
                for row, name in enumerate(table.value["name"])
            ]
        }
    )
    button.clicks += 1
    return {
        "widget": widget,
        "initial_index": initial_index,
        "saved_index": pd.read_csv(index_path),
    }


@pytest.fixture(scope="session")
def tutorial_moseq_df(tutorial_analysis_project, tutorial_groups_assigned):
    """`moseq_df` from the analysis notebook's dataframe cell."""
    import keypoint_moseq as kpms

    project_dir, model_name = tutorial_analysis_project
    return kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)


@pytest.fixture(scope="session")
def tutorial_stats_df(tutorial_moseq_df):
    """`stats_df` from the analysis notebook's summary statistics cell."""
    import keypoint_moseq as kpms

    return kpms.compute_stats_df(
        tutorial_moseq_df,
        min_frequency=0.005,
        groupby=["group", "name"],
    )


@pytest.fixture(scope="session")
def tutorial_syllables_labeled(tutorial_analysis_project, tutorial_moseq_df):
    """Widget and `syll_info.csv` contents (as generated and as saved) of
    the analysis notebook's syllable labeling cell. Every syllable in the
    widget's table, which lists those with a grid movie, gets a label and a
    short description typed with surrounding whitespace, and "Save syllable
    info" is clicked."""
    import keypoint_moseq as kpms

    project_dir, model_name = tutorial_analysis_project
    syll_info_path = os.path.join(project_dir, model_name, "syll_info.csv")

    widget = kpms.label_syllables(project_dir, model_name, tutorial_moseq_df)
    initial_syll_info = read_syll_info(syll_info_path)
    table, button = widget[1][0], widget[1][1][0]
    syllables = list(table.value["syllable"])
    table.patch(
        {
            "label": [
                (row, f"  {syllable_label(s)}\n") for row, s in enumerate(syllables)
            ],
            "short_description": [
                (row, f" {syllable_description(s)}  ") for row, s in enumerate(syllables)
            ],
        }
    )
    button.clicks += 1
    return {
        "widget": widget,
        "initial_syll_info": initial_syll_info,
        "saved_syll_info": read_syll_info(syll_info_path),
    }


@pytest.fixture(scope="session")
def tutorial_syll_stats_plot(
    tutorial_analysis_project, tutorial_stats_df, tutorial_syllables_labeled
):
    """Figure summary and printed output of the analysis notebook's group
    comparison cell."""
    project_dir, model_name = tutorial_analysis_project
    return plot_syll_stats(tutorial_stats_df, project_dir, model_name)


@pytest.fixture(scope="session")
def frozen_stats_df():
    """A `stats_df` stored in the repository, so that statistics computed
    from it can be snapshotted by value."""
    # Output of `compute_stats_df`, called as in the analysis notebook with
    # the cohort groups, for a full-scale fit of the tutorial dataset (model
    # 2025_02_11-10_07_45 in the maintainers' test assets). The frequency of
    # one syllable differs significantly between the groups.
    return pd.read_csv(FROZEN_STATS_DF, float_precision="round_trip")


@pytest.fixture(scope="session")
def frozen_syll_stats_plot(tmp_path_factory, frozen_stats_df):
    """Figure summary and printed output of the analysis notebook's group
    comparison cell run on the frozen `stats_df`, in a project without
    syllable labels."""
    project_dir = str(tmp_path_factory.mktemp("frozen_stats") / "demo_project")
    return plot_syll_stats(frozen_stats_df, project_dir, "frozen_model")


@pytest.fixture(scope="session")
def tutorial_transition_matrices(
    tutorial_analysis_project, tutorial_groups_assigned, tutorial_syllables_labeled
):
    """Outputs, printed text and figure summary of the analysis notebook's
    transition matrix cell. `outputs` holds `trans_mats`, `usages`, `groups`
    and `syll_include`."""
    import keypoint_moseq as kpms

    project_dir, model_name = tutorial_analysis_project
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        outputs = kpms.generate_transition_matrices(
            project_dir, model_name, normalize="bigram", min_frequency=0.005
        )
        trans_mats, usages, groups, syll_include = outputs
        kpms.visualize_transition_bigram(
            project_dir,
            model_name,
            groups,
            trans_mats,
            syll_include,
            normalize="bigram",
            show_syllable_names=True,
        )
    figure = bigram_plot_summary(plt.gcf())
    plt.close("all")
    return {"outputs": outputs, "printed": printed.getvalue(), "figure": figure}


@pytest.fixture(scope="session")
def tutorial_transition_graphs(tutorial_analysis_project, tutorial_transition_matrices):
    """Printed text, axis titles and per-group graph summaries of the
    analysis notebook's transition graph cell."""
    import keypoint_moseq as kpms

    project_dir, model_name = tutorial_analysis_project
    trans_mats, usages, groups, syll_include = tutorial_transition_matrices["outputs"]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        kpms.plot_transition_graph_group(
            project_dir,
            model_name,
            groups,
            trans_mats,
            usages,
            syll_include,
            layout="circular",
            show_syllable_names=False,
        )
    axes = plt.gcf().axes
    summary = {
        "printed": printed.getvalue(),
        "titles": [ax.get_title() for ax in axes],
        "graphs": [graph_plot_summary(ax) for ax in axes[: len(groups)]],
    }
    plt.close("all")
    return summary


@pytest.fixture(scope="session")
def tutorial_transition_graph_difference(
    tutorial_analysis_project, tutorial_transition_matrices
):
    """Printed text, axis titles, legend entries and per-pair graph
    summaries of the analysis notebook's difference graph cell."""
    import keypoint_moseq as kpms

    project_dir, model_name = tutorial_analysis_project
    trans_mats, usages, groups, syll_include = tutorial_transition_matrices["outputs"]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        kpms.plot_transition_graph_difference(
            project_dir, model_name, groups, trans_mats, usages, syll_include, layout="circular"
        )
    axes = plt.gcf().axes
    summary = {
        "printed": printed.getvalue(),
        "titles": [ax.get_title() for ax in axes],
        "legend": [text.get_text() for text in axes[-1].get_legend().get_texts()],
        "graphs": [graph_plot_summary(ax) for ax in axes[: math.comb(len(groups), 2)]],
    }
    plt.close("all")
    return summary
