import os

import numpy as np
import pandas as pd
import pytest

import keypoint_moseq as kpms
from model_utils import assert_tree_equal, structure

STATES_IN_RESULTS = {"latent_state": "x", "centroid": "v", "heading": "h"}


def frames_from_each_segment(keys, bounds):
    """(recording, start, stop) for each segment, where [start, stop) are the
    frames the segment supplies after unbatching. Segments overlap, and the
    later segment supplies the overlapping frames."""
    ranges = []
    for key, (start, end) in zip(keys, bounds):
        later_starts = [s for k, (s, _) in zip(keys, bounds) if k == key and s > start]
        ranges.append((key, start, min(later_starts, default=end)))
    return ranges


def results_csv_dir(project_dir, model_name):
    return os.path.join(project_dir, model_name, "results")


def test_results_structure(tutorial_results, data_regression):
    _, _, results = tutorial_results

    data_regression.check(structure(results))


def test_results_lengths_match_keypoints(tutorial_results, tutorial_keypoints):
    _, _, results = tutorial_results
    coordinates, _, _ = tutorial_keypoints

    assert sorted(results) == sorted(coordinates)
    for name, recording in results.items():
        for key, values in recording.items():
            assert len(values) == len(coordinates[name]), (name, key)


def test_results_match_model_states(tutorial_results):
    model, (keys, bounds), results = tutorial_results
    states = model["states"]
    nlags = states["x"].shape[1] - states["z"].shape[1]

    for segment, (name, start, stop) in enumerate(frames_from_each_segment(keys, bounds)):
        length = stop - start
        for result_key, state_key in STATES_IN_RESULTS.items():
            np.testing.assert_array_equal(
                results[name][result_key][start:stop],
                states[state_key][segment, :length],
                err_msg=f"{name} {result_key} segment {segment}",
            )
        # The first syllable of each segment is repeated for the first
        # `nlags` frames, which have no syllable of their own.
        padded_z = np.pad(states["z"][segment], (nlags, 0), mode="edge")
        np.testing.assert_array_equal(
            results[name]["syllable"][start:stop],
            padded_z[:length],
            err_msg=f"{name} syllable segment {segment}",
        )


def test_results_syllables_in_range(tutorial_results):
    model, _, results = tutorial_results
    num_states = model["hypparams"]["trans_hypparams"]["num_states"]

    for recording in results.values():
        assert recording["syllable"].min() >= 0
        assert recording["syllable"].max() < num_states


def test_results_h5_round_trip(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed
):
    _, _, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed

    assert_tree_equal(kpms.load_results(project_dir, model_name), results, "results")


def test_rerunning_extract_results_requires_overwrite(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed, tmp_path
):
    model, metadata, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed

    with pytest.raises(RuntimeError) as error:
        kpms.extract_results(model, metadata, project_dir, model_name)
    assert "already contains results" in " ".join(str(error.value).split())

    path = str(tmp_path / "results.h5")
    kpms.extract_results(model, metadata, path=path)
    kpms.extract_results(model, metadata, path=path, overwrite=True)
    assert_tree_equal(kpms.load_results(path=path), results, "results")


def test_results_csv_files(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed
):
    _, _, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed

    assert sorted(os.listdir(results_csv_dir(project_dir, model_name))) == sorted(
        f"{name}.csv" for name in results
    )


def test_results_csv_columns(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed, data_regression
):
    _, _, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed
    csv_dir = results_csv_dir(project_dir, model_name)

    headers = [
        list(pd.read_csv(os.path.join(csv_dir, f"{name}.csv"), nrows=0).columns)
        for name in results
    ]

    assert all(header == headers[0] for header in headers)
    data_regression.check({"columns": headers[0]})


def test_results_csv_values_match_results(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed
):
    _, _, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed
    csv_dir = results_csv_dir(project_dir, model_name)

    for name, recording in results.items():
        table = pd.read_csv(
            os.path.join(csv_dir, f"{name}.csv"), float_precision="round_trip"
        )
        latent_columns = [f"latent_state {i}" for i in range(recording["latent_state"].shape[1])]

        np.testing.assert_array_equal(table["syllable"], recording["syllable"])
        np.testing.assert_array_equal(
            table[["centroid x", "centroid y"]], np.round(recording["centroid"], 4)
        )
        np.testing.assert_array_equal(table["heading"], np.round(recording["heading"], 4))
        np.testing.assert_array_equal(
            table[latent_columns], np.round(recording["latent_state"], 4)
        )
