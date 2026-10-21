import os
import re

import h5py
import numpy as np

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
