import os

import pandas as pd
import pytest

import keypoint_moseq as kpms
from analysis_utils import synthetic_results

NAMES = ["rec_a", "rec_b", "rec_c"]
MODEL_NAME = "model"


@pytest.fixture
def results():
    return synthetic_results(NAMES)


def write_index(project_dir, text):
    with open(os.path.join(project_dir, "index.csv"), "w") as f:
        f.write(text)


def save_results(project_dir, results):
    model_dir = os.path.join(project_dir, MODEL_NAME)
    os.makedirs(model_dir)
    kpms.save_hdf5(os.path.join(model_dir, "results.h5"), results)


def moseq_df_groups(project_dir, results):
    """Group of each recording in the moseq_df built from `results`."""
    moseq_df = kpms.compute_moseq_df(str(project_dir), MODEL_NAME, results_dict=results, fps=30)
    return moseq_df.groupby("name")["group"].unique().map(list).to_dict()


def index_error(project_dir, results):
    with pytest.raises(ValueError) as error:
        kpms.compute_moseq_df(str(project_dir), MODEL_NAME, results_dict=results, fps=30)
    return str(error.value)


def test_index_groups_reach_moseq_df(tmp_path, results):
    write_index(tmp_path, "name,group\nrec_a,WT\nrec_b,KO\nrec_c,KO\n")

    assert moseq_df_groups(tmp_path, results) == {
        "rec_a": ["WT"],
        "rec_b": ["KO"],
        "rec_c": ["KO"],
    }


def test_moseq_df_without_index_puts_every_recording_in_default(tmp_path, results):
    assert moseq_df_groups(tmp_path, results) == {name: ["default"] for name in NAMES}


def test_index_rows_for_other_recordings_are_ignored(tmp_path, results):
    write_index(tmp_path, "name,group\nrec_a,WT\nrec_b,KO\nrec_c,KO\nother_model_rec,WT\n")

    assert set(moseq_df_groups(tmp_path, results)) == set(NAMES)


def test_blank_index_rows_are_ignored(tmp_path, results):
    write_index(tmp_path, "name,group\nrec_a,WT\nrec_b,KO\nrec_c,KO\n,\n,\n")

    assert moseq_df_groups(tmp_path, results)["rec_c"] == ["KO"]


def test_recordings_missing_from_index_raise(tmp_path, results):
    write_index(tmp_path, "name,group\nrec_a,WT\n")

    message = index_error(tmp_path, results)
    assert "rec_b" in message
    assert "rec_c" in message
    assert os.path.join(str(tmp_path), "index.csv") in message
    assert "interactive_group_setting" in message


def test_empty_group_raises_naming_the_line(tmp_path, results):
    write_index(tmp_path, "name,group\nrec_a,WT\nrec_b,\nrec_c,KO\n")

    message = index_error(tmp_path, results)
    assert "rec_b" in message
    assert "line 3" in message


def test_row_with_group_but_no_name_is_reported_with_missing_recordings(tmp_path, results):
    write_index(tmp_path, "name,group\nrec_a,WT\n,KO\nrec_c,KO\n")

    message = index_error(tmp_path, results)
    assert "rec_b" in message
    assert "line 3" in message


def test_duplicate_rows_for_a_recording_raise(tmp_path, results):
    write_index(tmp_path, "name,group\nrec_a,WT\nrec_b,KO\nrec_b,WT\nrec_c,KO\n")

    message = index_error(tmp_path, results)
    assert "rec_b" in message
    assert "lines 3, 4" in message


def test_group_names_that_pandas_reads_as_missing_are_kept(tmp_path, results):
    write_index(tmp_path, "name,group\nrec_a,NA\nrec_b,None\nrec_c,null\n")

    assert moseq_df_groups(tmp_path, results) == {
        "rec_a": ["NA"],
        "rec_b": ["None"],
        "rec_c": ["null"],
    }


def test_numeric_looking_recording_names_match(tmp_path):
    results = synthetic_results(["0012", "0013"])
    write_index(tmp_path, "name,group\n0012,WT\n0013,KO\n")

    assert moseq_df_groups(tmp_path, results) == {"0012": ["WT"], "0013": ["KO"]}


def test_whitespace_around_index_values_is_ignored(tmp_path, results):
    write_index(tmp_path, "name, group\nrec_a, WT\n rec_b ,KO\nrec_c,KO \n")

    assert moseq_df_groups(tmp_path, results) == {
        "rec_a": ["WT"],
        "rec_b": ["KO"],
        "rec_c": ["KO"],
    }


def test_transition_matrices_ignore_blank_index_rows(tmp_path, results):
    save_results(tmp_path, results)
    write_index(tmp_path, "name,group\nrec_a,WT\nrec_b,KO\nrec_c,KO\n,\n")

    _, _, groups, _ = kpms.generate_transition_matrices(
        str(tmp_path), MODEL_NAME, min_frequency=0.0
    )

    assert groups == ["KO", "WT"]


def test_transition_matrices_raise_for_recordings_missing_from_index(tmp_path, results):
    save_results(tmp_path, results)
    write_index(tmp_path, "name,group\nrec_a,WT\nrec_b,KO\n")

    with pytest.raises(ValueError) as error:
        kpms.generate_transition_matrices(str(tmp_path), MODEL_NAME)

    assert "rec_c" in str(error.value)


def test_group_setting_adds_recordings_missing_from_index(tmp_path, results):
    save_results(tmp_path, results)
    write_index(tmp_path, "name,group\nrec_a,WT\n")

    widget = kpms.interactive_group_setting(str(tmp_path), MODEL_NAME)

    index = pd.read_csv(os.path.join(tmp_path, "index.csv"), dtype=str)
    assert list(index["name"]) == NAMES
    assert list(index["group"]) == ["WT", "default", "default"]
    assert list(widget[0].value["name"]) == NAMES
