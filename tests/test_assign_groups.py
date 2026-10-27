import datetime
import html
import os
import re

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
    button, status = column.objects

    assert isinstance(table, pn.widgets.Tabulator)
    assert isinstance(button, pn.widgets.Button)
    assert isinstance(status, pn.pane.HTML)
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


def test_group_setting_confirms_the_saved_groups(
    tutorial_groups_assigned, tutorial_analysis_project
):
    project_dir, _ = tutorial_analysis_project
    status = tutorial_groups_assigned["widget"][1][1]
    counts = tutorial_groups_assigned["saved_index"]["group"].value_counts().sort_index()
    groups = ", ".join(
        f"{group} ({n} recording{'s' if n != 1 else ''})" for group, n in counts.items()
    )

    saved = re.fullmatch(r"Saved at \d\d:\d\d:\d\d to (.*)", status.object)
    assert saved is not None, status.object
    assert saved.group(1) == html.escape(f"{index_path(project_dir)}: {groups}")


def test_group_setting_sends_a_confirmation_for_every_save(
    analysis_project_without_groups, monkeypatch
):
    """The browser shows "Saving..." from the click until Python sends a
    confirmation, so saving unchanged groups again must send one again. The
    clock is fixed so both saves produce the same confirmation text."""
    class FixedClock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 12, 0, 0)

    monkeypatch.setattr(kpms.analysis, "datetime", FixedClock)
    project_dir, model_name = analysis_project_without_groups
    widget = kpms.interactive_group_setting(project_dir, model_name)
    button, status = widget[1].objects
    sent = []
    status.param.watch(lambda event: sent.append(event.new), "object")

    button.clicks += 1
    button.clicks += 1

    assert len([message for message in sent if message.startswith("Saved at")]) == 2
