import os

import pandas as pd
import panel as pn

import keypoint_moseq as kpms
from analysis_utils import cohort_group


def index_path(project_dir):
    return os.path.join(project_dir, "index.csv")


def test_group_setting_initial_index(
    tutorial_groups_assigned, tutorial_analysis_project, data_regression
):
    project_dir, model_name = tutorial_analysis_project
    initial_index = tutorial_groups_assigned["initial_index"]

    assert list(initial_index["name"]) == list(kpms.load_results(project_dir, model_name))
    data_regression.check(initial_index.to_dict("list"))


def test_group_setting_widget(tutorial_groups_assigned):
    widget = tutorial_groups_assigned["widget"]
    table, column = widget.objects
    (button,) = column.objects

    assert isinstance(table, pn.widgets.Tabulator)
    assert isinstance(button, pn.widgets.Button)
    assert button.name == "Save group info"
    assert table.editors["name"] is None
    assert list(table.value["name"]) == list(tutorial_groups_assigned["initial_index"]["name"])


def test_group_setting_saves_stripped_groups(tutorial_groups_assigned, data_regression):
    table = tutorial_groups_assigned["widget"][0]
    initial_index = tutorial_groups_assigned["initial_index"]
    saved_index = tutorial_groups_assigned["saved_index"]

    # The table keeps the groups as typed; saving strips the whitespace.
    assert all(group != group.strip() for group in table.value["group"])
    assert list(saved_index["name"]) == list(initial_index["name"])
    assert list(saved_index["group"]) == [cohort_group(name) for name in saved_index["name"]]
    data_regression.check(saved_index.to_dict("list"))


def test_group_setting_reopens_saved_index(
    tutorial_groups_assigned, tutorial_analysis_project
):
    project_dir, model_name = tutorial_analysis_project
    with open(index_path(project_dir)) as f:
        saved = f.read()

    widget = kpms.interactive_group_setting(project_dir, model_name)

    pd.testing.assert_frame_equal(widget[0].value, tutorial_groups_assigned["saved_index"])
    with open(index_path(project_dir)) as f:
        assert f.read() == saved
