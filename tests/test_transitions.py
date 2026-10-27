import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from jax_moseq.utils import get_frequencies
from matplotlib.colors import to_rgb

import keypoint_moseq as kpms
from analysis_utils import (
    bigram_counts,
    check_figure_files,
    graph_edge_weights,
    syllable_names,
    synthetic_results,
)

MIN_FREQUENCY = 0.005  # `min_frequency` of the notebook's transition matrix cell


def figures_dir(project_dir, model_name):
    return os.path.join(project_dir, model_name, "figures")


def indexed_labels(project_dir, model_name):
    """Group and syllable sequence of each recording, in `index.csv` order."""
    index = pd.read_csv(os.path.join(project_dir, "index.csv"))
    results = kpms.load_results(project_dir, model_name)
    return [
        (group, results[name]["syllable"])
        for name, group in zip(index["name"], index["group"])
    ]


def check_edges(graph, weights, widths, colors):
    """The graph's edges are the keys of `weights`, drawn with the given
    width and color functions of each weight."""
    assert sorted(graph["edges"]) == sorted(weights)
    expected = {pair: weights[pair] for pair in graph["edges"]}
    np.testing.assert_allclose(
        graph["edge_widths"], [widths(w) for w in expected.values()], rtol=1e-12
    )
    np.testing.assert_array_equal(
        graph["edge_colors"], [to_rgb(colors(w)) for w in expected.values()]
    )


def test_transition_matrices_groups_and_syllables(
    tutorial_transition_matrices, tutorial_stats_df, tutorial_analysis_project
):
    project_dir, model_name = tutorial_analysis_project
    _, _, groups, syll_include = tutorial_transition_matrices["outputs"]
    labels = [sequence for _, sequence in indexed_labels(project_dir, model_name)]
    save_path = os.path.join(figures_dir(project_dir, model_name), "transition_matrices")

    assert groups == ["a", "b"]
    np.testing.assert_array_equal(
        syll_include, np.flatnonzero(get_frequencies(labels) > MIN_FREQUENCY)
    )
    np.testing.assert_array_equal(syll_include, sorted(tutorial_stats_df["syllable"].unique()))
    assert tutorial_transition_matrices["printed"] == (
        f"Group(s): a, b\nSaved figure to {save_path}.png\n"
    )


def test_transition_matrices_values(tutorial_transition_matrices, tutorial_analysis_project):
    project_dir, model_name = tutorial_analysis_project
    trans_mats, usages, groups, syll_include = tutorial_transition_matrices["outputs"]
    labels = indexed_labels(project_dir, model_name)
    included = np.ix_(syll_include, syll_include)

    assert len(trans_mats) == len(usages) == len(groups)
    for group, trans_mat, usage in zip(groups, trans_mats, usages):
        sequences = [sequence for g, sequence in labels if g == group]
        counts = sum(bigram_counts(sequence) for sequence in sequences)

        np.testing.assert_allclose(
            trans_mat, (counts / counts.sum())[included], rtol=1e-12, err_msg=group
        )
        np.testing.assert_array_equal(
            usage, get_frequencies(sequences)[syll_include], err_msg=group
        )


def test_transition_bigram_plot(
    tutorial_transition_matrices, tutorial_syllables_labeled, tutorial_analysis_project
):
    project_dir, model_name = tutorial_analysis_project
    trans_mats, _, groups, syll_include = tutorial_transition_matrices["outputs"]
    figure = tutorial_transition_matrices["figure"]
    names = syllable_names(tutorial_syllables_labeled["saved_syll_info"], syll_include)

    check_figure_files(
        os.path.join(figures_dir(project_dir, model_name), "transition_matrices"), (3600, 1800)
    )
    assert figure["titles"] == groups
    for image, clim, trans_mat in zip(figure["images"], figure["clims"], trans_mats):
        np.testing.assert_array_equal(image, trans_mat)
        assert clim[1] == max(m.max() for m in trans_mats)
    assert figure["xticklabels"] == [names] * len(groups)
    assert figure["yticklabels"] == names
    assert figure["colorbar_labels"] == ["Bigram transition probability"] * len(groups)


def test_transition_graph_plot(
    tutorial_transition_graphs, tutorial_transition_matrices, tutorial_analysis_project
):
    project_dir, model_name = tutorial_analysis_project
    trans_mats, usages, groups, syll_include = tutorial_transition_matrices["outputs"]
    save_path = os.path.join(figures_dir(project_dir, model_name), "transition_graphs")

    check_figure_files(save_path, (6000, 3000))
    assert tutorial_transition_graphs["printed"] == f"Saved figure to {save_path}.png\n"
    assert tutorial_transition_graphs["titles"] == groups
    for graph, trans_mat, usage in zip(tutorial_transition_graphs["graphs"], trans_mats, usages):
        assert graph["labels"] == [str(s) for s in syll_include]
        np.testing.assert_allclose(
            graph["node_sizes"], usage / usage.sum() * 2000 + 1000, rtol=1e-12
        )
        np.testing.assert_array_equal(
            graph["node_edgecolors"], [to_rgb("red")] * len(syll_include)
        )
        check_edges(
            graph,
            graph_edge_weights(trans_mat * 100),
            widths=lambda w: w,
            colors=lambda w: "black",
        )


def test_transition_graph_difference_plot(
    tutorial_transition_graph_difference,
    tutorial_transition_matrices,
    tutorial_analysis_project,
    data_regression,
):
    project_dir, model_name = tutorial_analysis_project
    trans_mats, usages, groups, syll_include = tutorial_transition_matrices["outputs"]
    plot = tutorial_transition_graph_difference
    save_path = os.path.join(figures_dir(project_dir, model_name), "transition_graphs_diff")
    pairs = list(combinations(range(len(groups)), 2))
    sign_color = lambda value: "blue" if value > 0 else "red"

    check_figure_files(save_path, (4800, 2400))
    assert plot["printed"] == f"Saved figure to {save_path}.png\n"
    assert plot["titles"][: len(pairs)] == [f"{groups[i]} - {groups[j]}" for i, j in pairs]
    for graph, (i, j) in zip(plot["graphs"], pairs):
        usage_diff = usages[i] - usages[j]
        assert graph["labels"] == [str(s) for s in syll_include]
        np.testing.assert_allclose(
            graph["node_sizes"],
            np.abs(usage_diff) / np.abs(usage_diff).sum() * 3000 + 500,
            rtol=1e-12,
        )
        np.testing.assert_array_equal(
            graph["node_edgecolors"], [to_rgb(sign_color(u)) for u in usage_diff]
        )
        check_edges(
            graph,
            graph_edge_weights((trans_mats[i] - trans_mats[j]) * 1000),
            widths=np.abs,
            colors=sign_color,
        )
    data_regression.check({"titles": plot["titles"], "legend": plot["legend"]})
    # blue where the first group of the panel title is higher, as drawn above
    for label, color in zip(plot["legend"], plot["legend_colors"]):
        assert color == to_rgb(sign_color(1 if "first group" in label else -1)), label


def test_analysis_figures(
    tutorial_syll_stats_plot,
    tutorial_transition_matrices,
    tutorial_transition_graphs,
    tutorial_transition_graph_difference,
    tutorial_analysis_project,
    data_regression,
):
    project_dir, model_name = tutorial_analysis_project

    data_regression.check({"files": sorted(os.listdir(figures_dir(project_dir, model_name)))})


def test_transition_graph_difference_with_one_group(analysis_project_without_groups, capsys):
    project_dir, model_name = analysis_project_without_groups
    # Saving the group-setting widget without edits puts every recording in
    # the group "default".
    widget = kpms.interactive_group_setting(project_dir, model_name)
    widget[1][0].clicks += 1

    trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
        project_dir, model_name, normalize="bigram", min_frequency=0.005
    )
    assert groups == ["default"]
    capsys.readouterr()
    open_figures = plt.get_fignums()
    kpms.plot_transition_graph_difference(
        project_dir, model_name, groups, trans_mats, usages, syll_include, layout="circular"
    )

    printed = " ".join(capsys.readouterr().out.split())
    assert "it needs at least two groups and got 1 (default)" in printed
    assert plt.get_fignums() == open_figures
    assert not os.path.exists(
        os.path.join(figures_dir(project_dir, model_name), "transition_graphs_diff.png")
    )


def test_transition_matrices_when_a_group_never_uses_an_included_syllable(tmp_path):
    group_of = {"rec_a1": "a", "rec_a2": "a", "rec_b1": "b", "rec_b2": "b"}
    results = synthetic_results(list(group_of))
    # group b never uses syllable 4, the highest included syllable
    for name in ["rec_b1", "rec_b2"]:
        syllables = results[name]["syllable"]
        results[name]["syllable"] = np.where(syllables == 4, 3, syllables)
    os.makedirs(tmp_path / "model")
    kpms.save_hdf5(str(tmp_path / "model" / "results.h5"), results)
    pd.DataFrame({"name": list(group_of), "group": list(group_of.values())}).to_csv(
        tmp_path / "index.csv", index=False
    )

    trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
        str(tmp_path), "model"
    )

    assert groups == ["a", "b"]
    np.testing.assert_array_equal(syll_include, np.arange(5))
    for group, usage in zip(groups, usages):
        sequences = [results[name]["syllable"] for name, g in group_of.items() if g == group]
        np.testing.assert_array_equal(
            usage, get_frequencies(sequences, num_states=5), err_msg=group
        )
    assert usages[1][4] == 0
    assert not trans_mats[1][4].any() and not trans_mats[1][:, 4].any()


@pytest.mark.parametrize("combine", [False, True], ids=["per_recording", "combined"])
def test_transition_matrix_counts_the_transition_out_of_the_first_run(combine):
    sequences = [np.array([0, 0, 1, 1, 2, 2]), np.array([2, 1, 1])]

    matrices = kpms.get_transition_matrix(
        sequences, max_syllable=3, normalize=None, combine=combine
    )

    expected = np.zeros((3, 3))
    expected[0, 1] = expected[1, 2] = expected[2, 1] = 1
    np.testing.assert_array_equal(matrices if combine else sum(matrices), expected)


def test_transition_matrices_with_syllables_numbered_100_or_higher(tmp_path):
    group_of = {"rec_a": "a", "rec_b": "b"}
    results = synthetic_results(list(group_of))
    for recording in results.values():
        recording["syllable"] = np.where(recording["syllable"] == 4, 119, recording["syllable"])
    os.makedirs(tmp_path / "model")
    kpms.save_hdf5(str(tmp_path / "model" / "results.h5"), results)
    pd.DataFrame({"name": list(group_of), "group": list(group_of.values())}).to_csv(
        tmp_path / "index.csv", index=False
    )

    trans_mats, _, groups, syll_include = kpms.generate_transition_matrices(
        str(tmp_path), "model"
    )

    np.testing.assert_array_equal(syll_include, [0, 1, 2, 3, 119])
    included = np.ix_(syll_include, syll_include)
    for group, trans_mat in zip(groups, trans_mats):
        [name] = [name for name, g in group_of.items() if g == group]
        counts = bigram_counts(results[name]["syllable"], num_labels=120)
        np.testing.assert_allclose(
            trans_mat, (counts / counts.sum())[included], rtol=1e-12, err_msg=group
        )


@pytest.mark.parametrize("combine", [False, True], ids=["per_recording", "combined"])
@pytest.mark.parametrize("as_type", [list, pd.Series], ids=["lists", "series"])
def test_transition_matrix_of_recordings_given_as_lists_or_series(as_type, combine):
    sequences = [as_type([0, 0, 1, 1, 2, 2]), as_type([2, 1, 1])]

    matrices = kpms.get_transition_matrix(
        sequences, max_syllable=3, normalize=None, combine=combine
    )

    expected = np.zeros((3, 3))
    expected[0, 1] = expected[1, 2] = expected[2, 1] = 1
    np.testing.assert_array_equal(matrices if combine else sum(matrices), expected)
