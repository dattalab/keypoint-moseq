import os

import h5py
import numpy as np
from jax_moseq.utils import get_frequencies

import keypoint_moseq as kpms
from keypoint_moseq.io import load_hdf5, save_hdf5
from model_utils import assert_tree_equal

PER_STATE_PARAMS = ["betas", "Ab", "Q"]
LOCATION_PARAMS = ["delta_h", "sigmasq_h", "delta_v", "sigmasq_v"]


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


def synthetic_checkpoint(path, num_states=20):
    """A checkpoint of a location-aware model, saved as fit_model saves one,
    with three snapshots of random parameters and states for three 400-frame
    segments. Labels are drawn from the first four syllables only, so the
    other 16 are never used and tie at frequency 0."""
    rng = np.random.default_rng(0)
    segments, frames, keypoints, latent_dim = 3, 400, 5, 4

    def snapshot():
        return {
            "params": {
                "betas": rng.random(num_states),
                "pi": rng.random((num_states, num_states)),
                "Ab": rng.random((num_states, latent_dim, latent_dim + 1)),
                "Q": rng.random((num_states, latent_dim, latent_dim)),
                "Cd": rng.random((keypoints * 2, latent_dim + 1)),
                "sigmasq": rng.random(keypoints),
                "delta_h": rng.random(num_states),
                "sigmasq_h": rng.random(num_states),
                "delta_v": rng.random((num_states, 2)),
                "sigmasq_v": rng.random(num_states),
            },
            "states": {
                "z": rng.integers(0, 4, (segments, frames - 1)),
                "x": rng.random((segments, frames, latent_dim)),
                "v": rng.random((segments, frames, 2)),
                "h": rng.random((segments, frames)),
                "s": rng.random((segments, frames, keypoints)),
            },
            "noise_prior": rng.random((segments, frames, keypoints)),
            "seed": 3,
            "hypparams": {"trans_hypparams": {"num_states": num_states}},
        }

    save_hdf5(
        str(path),
        {
            "model_snapshots": {str(i): snapshot() for i in (0, 25, 50)},
            "metadata": (np.array(["a", "b", "c"]), np.array([[0, frames]] * segments)),
            "data": {"mask": np.ones((segments, frames))},
        },
    )
    return str(path)


def test_reindex_permutes_the_location_aware_parameters(tmp_path):
    path = synthetic_checkpoint(tmp_path / "checkpoint.h5")
    before = load_hdf5(path)
    index = np.random.default_rng(1).permutation(20)

    kpms.reindex_syllables_in_checkpoint(path=path, index=index)

    after = load_hdf5(path)
    for iteration, snapshot in before["model_snapshots"].items():
        reindexed = after["model_snapshots"][iteration]
        np.testing.assert_array_equal(
            reindexed["states"]["z"], np.argsort(index)[snapshot["states"]["z"]]
        )
        for key in PER_STATE_PARAMS + LOCATION_PARAMS:
            np.testing.assert_array_equal(
                reindexed["params"][key], snapshot["params"][key][index]
            )
        np.testing.assert_array_equal(
            reindexed["params"]["pi"], snapshot["params"]["pi"][index][:, index]
        )
        for model in (snapshot, reindexed):
            del model["states"]["z"]
            for key in PER_STATE_PARAMS + LOCATION_PARAMS + ["pi"]:
                del model["params"][key]
        assert_tree_equal(reindexed, snapshot, f"snapshot {iteration}")
    assert_tree_equal(after["data"], before["data"], "data")
    for actual, expected in zip(after["metadata"], before["metadata"]):
        np.testing.assert_array_equal(actual, expected)


def test_reindex_leaves_the_checkpoint_size_unchanged(tmp_path):
    path = synthetic_checkpoint(tmp_path / "checkpoint.h5")
    size = os.path.getsize(path)

    kpms.reindex_syllables_in_checkpoint(path=path)
    kpms.reindex_syllables_in_checkpoint(path=path, index=np.random.default_rng(1).permutation(20))

    assert os.path.getsize(path) == size


def test_reindexing_again_changes_nothing(tmp_path):
    """Syllables seen equally often, here the 16 never used, keep their
    order, so a reindexed checkpoint is already in frequency order."""
    path = synthetic_checkpoint(tmp_path / "checkpoint.h5")
    kpms.reindex_syllables_in_checkpoint(path=path)
    reindexed = load_hdf5(path)

    index = kpms.reindex_syllables_in_checkpoint(path=path)

    np.testing.assert_array_equal(index, np.arange(20))
    assert_tree_equal(load_hdf5(path, "model_snapshots"), reindexed["model_snapshots"], "snapshots")
