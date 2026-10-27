import os
import re

import h5py
import numpy as np
import pytest

from jax_moseq.utils import batch

import keypoint_moseq as kpms
from model_utils import assert_tree_equal, model_structure, plain


def test_tutorial_model_hypparams(
    tutorial_model, tutorial_outlier_removed, data_regression
):
    project_dir, _, _ = tutorial_outlier_removed
    config = lambda: kpms.load_config(project_dir)
    hypparams = plain(tutorial_model["hypparams"])

    # sigmasq_loc is estimated from the data; its value is snapshotted with a
    # tolerance in test_hyperparameters.py, so here only check that it reached
    # the model unchanged.
    sigmasq_loc = hypparams["cen_hypparams"].pop("sigmasq_loc")
    assert sigmasq_loc == config()["cen_hypparams"]["sigmasq_loc"]
    data_regression.check(hypparams)


def test_tutorial_model_structure(tutorial_model, data_regression):
    data_regression.check(model_structure(tutorial_model))


def test_tutorial_arhmm_structure(tutorial_arhmm, data_regression):
    model, _, _ = tutorial_arhmm

    data_regression.check(model_structure(model))


def test_tutorial_arhmm_syllables_in_range(tutorial_arhmm):
    model, _, _ = tutorial_arhmm
    z = np.asarray(model["states"]["z"])
    num_states = model["hypparams"]["trans_hypparams"]["num_states"]

    assert z.min() >= 0
    assert z.max() < num_states


def test_tutorial_arhmm_output_files(
    tutorial_arhmm, tutorial_outlier_removed, data_regression
):
    _, model_name, _ = tutorial_arhmm
    project_dir, _, _ = tutorial_outlier_removed
    model_dir = os.path.join(project_dir, model_name)

    assert re.fullmatch(r"\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}", model_name)
    files = sorted(
        os.path.relpath(os.path.join(root, filename), model_dir)
        for root, _, filenames in os.walk(model_dir)
        for filename in filenames
    )
    data_regression.check({"files": files})


def test_tutorial_arhmm_checkpoint_round_trip(
    tutorial_model, tutorial_arhmm, tutorial_formatted_data, tutorial_outlier_removed
):
    model, model_name, num_ar_iters = tutorial_arhmm
    data, (keys, bounds) = tutorial_formatted_data
    project_dir, _, _ = tutorial_outlier_removed

    saved_model, saved_data, saved_metadata, iteration = kpms.load_checkpoint(
        project_dir, model_name, iteration=num_ar_iters
    )
    initial_model, _, _, initial_iteration = kpms.load_checkpoint(
        project_dir, model_name, iteration=0
    )

    assert iteration == num_ar_iters
    assert_tree_equal(saved_model, model)
    assert_tree_equal(saved_data, data, "data")
    np.testing.assert_array_equal(saved_metadata[0], keys)
    np.testing.assert_array_equal(saved_metadata[1], bounds)
    assert initial_iteration == 0
    assert_tree_equal(initial_model, tutorial_model)


def test_tutorial_full_model_hypparams(
    tutorial_full_model, tutorial_arhmm, tutorial_outlier_removed, data_regression
):
    model, _ = tutorial_full_model
    ar_model, _, _ = tutorial_arhmm
    project_dir, _, _ = tutorial_outlier_removed
    config = lambda: kpms.load_config(project_dir)
    hypparams = plain(model["hypparams"])

    expected = plain(ar_model["hypparams"])
    expected["trans_hypparams"]["kappa"] = 1e4
    assert hypparams == expected

    sigmasq_loc = hypparams["cen_hypparams"].pop("sigmasq_loc")
    assert sigmasq_loc == config()["cen_hypparams"]["sigmasq_loc"]
    data_regression.check(hypparams)


def test_tutorial_full_model_structure(tutorial_full_model, data_regression):
    model, _ = tutorial_full_model

    data_regression.check(model_structure(model))


def test_tutorial_full_model_syllables_in_range(tutorial_full_model):
    model, _ = tutorial_full_model
    z = np.asarray(model["states"]["z"])
    num_states = model["hypparams"]["trans_hypparams"]["num_states"]

    assert z.min() >= 0
    assert z.max() < num_states


def test_tutorial_full_model_checkpoint(
    tutorial_full_model, tutorial_arhmm, tutorial_outlier_removed, data_regression
):
    model, num_iters = tutorial_full_model
    ar_model, model_name, num_ar_iters = tutorial_arhmm
    project_dir, _, _ = tutorial_outlier_removed
    model_dir = os.path.join(project_dir, model_name)

    saved_model, _, _, iteration = kpms.load_checkpoint(project_dir, model_name)
    saved_ar_model, _, _, _ = kpms.load_checkpoint(
        project_dir, model_name, iteration=num_ar_iters
    )

    assert iteration == num_iters
    assert_tree_equal(saved_model, model)
    assert_tree_equal(saved_ar_model, ar_model)
    with h5py.File(os.path.join(model_dir, "checkpoint.h5"), "r") as f:
        snapshots = sorted(f["model_snapshots"], key=int)
    files = sorted(
        os.path.relpath(os.path.join(root, filename), model_dir)
        for root, _, filenames in os.walk(model_dir)
        for filename in filenames
    )
    data_regression.check({"snapshots": snapshots, "files": files})


def snapshot_iterations(checkpoint_path):
    with h5py.File(checkpoint_path, "r") as f:
        return sorted(int(i) for i in f["model_snapshots"])


def test_fit_model_continues_a_checkpoint_on_the_same_data(one_recording, tmp_path):
    config, data, metadata, models = one_recording
    project_dir = str(tmp_path)

    model, model_name = kpms.fit_model(
        models[False], data, metadata, project_dir, ar_only=True, num_iters=1, **config
    )
    kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=True,
        start_iter=1,
        num_iters=2,
        **config,
    )

    checkpoint_path = os.path.join(project_dir, model_name, "checkpoint.h5")
    assert snapshot_iterations(checkpoint_path) == [0, 1, 2]


def test_fit_model_rejects_a_checkpoint_saved_from_other_data(
    one_recording, tutorial_formatted_data, tutorial_model, tmp_path
):
    """Fitting the whole tutorial dataset under the name of a model fit to
    one of its recordings raises before the checkpoint is changed (#174)."""
    config, data, metadata, models = one_recording
    all_data, all_metadata = tutorial_formatted_data
    project_dir = str(tmp_path)
    _, model_name = kpms.fit_model(
        models[False], data, metadata, project_dir, ar_only=True, num_iters=1, **config
    )
    model_dir = os.path.join(project_dir, model_name)
    checkpoint_path = os.path.join(model_dir, "checkpoint.h5")

    with pytest.raises(ValueError) as error:
        kpms.fit_model(
            tutorial_model,
            all_data,
            all_metadata,
            project_dir,
            model_name,
            ar_only=True,
            num_iters=1,
            **config,
        )

    message = " ".join(str(error.value).split())
    only_in_data = sorted(set(all_metadata[0]) - set(metadata[0]))
    num_names = len(set(all_metadata[0]))
    assert f"model_name '{model_name}'" in message
    assert checkpoint_path in message
    assert f"Recording names (checkpoint: 1, data: {num_names})" in message
    assert (
        f"only in data: {', '.join(only_in_data[:5])}, and {len(only_in_data) - 5} more"
        in message
    )
    assert (
        f"Segment bounds (checkpoint: {len(metadata[0])} segments, "
        f"data: {len(all_metadata[0])} segments)" in message
    )
    for key in ["Y", "conf", "mask"]:
        assert (
            f"{key}: checkpoint {tuple(data[key].shape)}, "
            f"data {tuple(all_data[key].shape)}" in message
        )
    assert f"delete {model_dir} and run again" in message
    assert "use apply_model instead" in message
    assert snapshot_iterations(checkpoint_path) == [0, 1]


def test_syllable_marginal_samples_are_timepoints_by_samples(one_recording):
    config, data, metadata, models = one_recording

    marginals, samples = kpms.estimate_syllable_marginals(
        models[False],
        data,
        metadata,
        burn_in_iters=1,
        num_samples=2,
        steps_per_sample=1,
        return_samples=True,
        **config,
    )

    for name, recording_marginals in marginals.items():
        assert samples[name].shape == (len(recording_marginals), 2), name


@pytest.mark.parametrize(
    "nlags, num_frames",
    [(3, 100), (5, 84)],
    ids=["default_nlags", "last_segment_shorter_than_nlags"],
)
def test_syllable_marginals_where_segments_overlap(nlags, num_frames, monkeypatch):
    """Each segment's marginals put all weight on the frame's own index, so
    the most likely syllable of each frame shows whether a segment inferred it
    or it was copied. Segments start every 40 frames, so 84 frames leave a
    last segment of 4. Model setup and resampling are replaced, so only the
    reassembly of segments runs."""
    _, mask, metadata = batch({"rec": np.zeros((num_frames, 1))}, seg_length=40)
    _, bounds = metadata
    frame_indexes = bounds[:, :1] + np.arange(mask.shape[1])
    num_states = frame_indexes.max() + 1
    model = {
        "seed": 0,
        "params": {"Ab": np.zeros((num_states, 2, 2 * nlags + 1))},
        "hypparams": {"trans_hypparams": {"num_states": num_states}},
        "states": {"z": frame_indexes[:, nlags:], "x": np.zeros((*mask.shape, 2))},
    }
    monkeypatch.setattr(kpms.fitting, "init_model", lambda **kwargs: model)
    monkeypatch.setattr(kpms.fitting, "_wrapped_resample", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        kpms.fitting,
        "stateseq_marginals",
        lambda *args, **kwargs: np.eye(num_states)[model["states"]["z"]],
    )

    marginals = kpms.estimate_syllable_marginals(
        model, {"mask": mask}, metadata, burn_in_iters=0, num_samples=1, steps_per_sample=1
    )

    expected = np.concatenate([np.repeat(nlags, nlags), np.arange(nlags, num_frames)])
    np.testing.assert_array_equal(marginals["rec"].argmax(axis=1), expected)
