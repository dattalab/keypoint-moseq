import os

import h5py
import numpy as np
from jax_moseq.utils import get_frequencies

import keypoint_moseq as kpms
from model_utils import assert_tree_equal

PER_STATE_PARAMS = ["betas", "Ab", "Q"]


def saved_iterations(project_dir, model_name):
    path = os.path.join(project_dir, model_name, "checkpoint.h5")
    with h5py.File(path, "r") as f:
        return sorted(int(iteration) for iteration in f["model_snapshots"])


def test_reindex_index_is_permutation(tutorial_reindexed, tutorial_full_model):
    _, index = tutorial_reindexed
    model, _ = tutorial_full_model
    num_states = model["hypparams"]["trans_hypparams"]["num_states"]

    np.testing.assert_array_equal(np.sort(index), np.arange(num_states))


def test_reindexed_syllables_ordered_by_frequency(
    tutorial_reindexed, tutorial_outlier_removed
):
    model_name, index = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed

    model, data, _, _ = kpms.load_checkpoint(project_dir, model_name)
    frequencies = get_frequencies(
        np.asarray(model["states"]["z"]),
        np.asarray(data["mask"]),
        num_states=len(index),
        runlength=True,
    )

    assert np.all(np.diff(frequencies) <= 0)


def test_reindex_permutes_every_snapshot(
    tutorial_reindexed, tutorial_arhmm, tutorial_outlier_removed
):
    model_name, index = tutorial_reindexed
    _, fitted_model_name, _ = tutorial_arhmm
    project_dir, _, _ = tutorial_outlier_removed
    iterations = saved_iterations(project_dir, fitted_model_name)

    assert saved_iterations(project_dir, model_name) == iterations
    for iteration in iterations:
        before, data_before, metadata_before, _ = kpms.load_checkpoint(
            project_dir, fitted_model_name, iteration=iteration
        )
        after, data_after, metadata_after, _ = kpms.load_checkpoint(
            project_dir, model_name, iteration=iteration
        )

        np.testing.assert_array_equal(
            after["states"]["z"], np.argsort(index)[before["states"]["z"]]
        )
        for key in PER_STATE_PARAMS:
            np.testing.assert_array_equal(
                after["params"][key], before["params"][key][index]
            )
        np.testing.assert_array_equal(
            after["params"]["pi"], before["params"]["pi"][index][:, index]
        )

        for model in (before, after):
            del model["states"]["z"]
            for key in PER_STATE_PARAMS + ["pi"]:
                del model["params"][key]
        assert_tree_equal(after, before, f"snapshot {iteration}")
        assert_tree_equal(data_after, data_before, "data")
        np.testing.assert_array_equal(metadata_after[0], metadata_before[0])
        np.testing.assert_array_equal(metadata_after[1], metadata_before[1])
