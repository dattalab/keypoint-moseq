import os

import numpy as np
import panel as pn

import keypoint_moseq as kpms
from analysis_utils import syllable_description, syllable_label


def movie_syllables(tutorial_grid_movies):
    """Syllables that have a grid movie."""
    _, sampled_instances, _ = tutorial_grid_movies
    return sorted(int(syllable) for syllable in sampled_instances)


def movie_path(project_dir, model_name, syllable):
    return os.path.join(project_dir, model_name, "grid_movies", f"syllable{syllable}.mp4")


def test_syll_info_generated(
    tutorial_syllables_labeled,
    tutorial_analysis_project,
    tutorial_grid_movies,
    data_regression,
):
    project_dir, model_name = tutorial_analysis_project
    results = kpms.load_results(project_dir, model_name)
    movies = movie_syllables(tutorial_grid_movies)
    syll_info = tutorial_syllables_labeled["initial_syll_info"]

    data_regression.check({"columns": list(syll_info.columns)})
    np.testing.assert_array_equal(
        syll_info["syllable"],
        np.unique(np.concatenate([r["syllable"] for r in results.values()])),
    )
    assert list(syll_info["movie_path"]) == [
        movie_path(project_dir, model_name, s) if s in movies else ""
        for s in syll_info["syllable"]
    ]
    assert (syll_info["label"] == "").all()
    assert (syll_info["short_description"] == "").all()


def test_label_syllables_widget(
    tutorial_syllables_labeled,
    tutorial_analysis_project,
    tutorial_moseq_df,
    tutorial_grid_movies,
):
    project_dir, model_name = tutorial_analysis_project
    movies = movie_syllables(tutorial_grid_movies)

    widget = kpms.label_syllables(project_dir, model_name, tutorial_moseq_df)
    (select, video), (table, column) = (part.objects for part in widget.objects)
    (button,) = column.objects

    assert isinstance(table, pn.widgets.Tabulator)
    assert button.name == "Save syllable info"
    assert select.options == movies
    assert list(table.value["syllable"]) == movies
    assert list(table.value["label"]) == [syllable_label(s) for s in movies]
    for syllable in movies:
        select.value = syllable
        pane = video.object()
        assert isinstance(pane, pn.pane.Video)
        assert pane.object == movie_path(project_dir, model_name, syllable)


def test_label_syllables_saves_stripped_labels(
    tutorial_syllables_labeled, tutorial_grid_movies
):
    table = tutorial_syllables_labeled["widget"][1][0]
    initial = tutorial_syllables_labeled["initial_syll_info"]
    saved = tutorial_syllables_labeled["saved_syll_info"]
    movies = movie_syllables(tutorial_grid_movies)
    without_movies = [s for s in initial["syllable"] if s not in movies]

    # The table keeps the text as typed; saving strips the whitespace.
    assert all(label != label.strip() for label in table.value["label"])
    assert list(saved.columns) == list(initial.columns)
    # Saved rows list the syllables with a movie first.
    assert list(saved["syllable"]) == movies + without_movies
    saved = saved.set_index("syllable")
    initial = initial.set_index("syllable")
    assert list(saved.loc[movies, "label"]) == [syllable_label(s) for s in movies]
    assert list(saved.loc[movies, "short_description"]) == [
        syllable_description(s) for s in movies
    ]
    assert (saved.loc[without_movies, ["label", "short_description"]] == "").all(axis=None)
    assert saved["movie_path"].sort_index().equals(initial["movie_path"])
