import os
import shutil

import pytest

import keypoint_moseq as kpms
from model_utils import assert_tree_equal


def test_apply_model_results_for_new_recording(tutorial_applied):
    results = tutorial_applied["results"]
    coordinates = tutorial_applied["coordinates"]
    num_states = tutorial_applied["model"]["hypparams"]["trans_hypparams"]["num_states"]

    assert sorted(results) == sorted(coordinates)
    for name, recording in results.items():
        assert sorted(recording) == ["centroid", "heading", "latent_state", "syllable"]
        for key, values in recording.items():
            assert len(values) == len(coordinates[name]), (name, key)
        assert recording["syllable"].min() >= 0
        assert recording["syllable"].max() < num_states


def test_apply_model_appends_to_results_h5(
    tutorial_applied, tutorial_results, tutorial_outlier_removed
):
    _, _, training_results = tutorial_results
    project_dir, _, _ = tutorial_outlier_removed
    new_results = tutorial_applied["results"]

    saved = kpms.load_results(project_dir, tutorial_applied["model_name"])

    assert sorted(saved) == sorted([*training_results, *new_results])
    for name in training_results:
        assert_tree_equal(saved[name], training_results[name], name)
    for name in new_results:
        assert_tree_equal(saved[name], new_results[name], name)


def test_apply_model_keeps_trained_parameters(tutorial_applied):
    model = tutorial_applied["model"]
    applied_model = tutorial_applied["applied_model"]

    assert_tree_equal(applied_model["params"], model["params"], "params")
    assert_tree_equal(applied_model["hypparams"], model["hypparams"], "hypparams")


def test_apply_model_to_training_recording_requires_overwrite(
    tutorial_applied,
    tutorial_results,
    tutorial_reindexed,
    tutorial_outlier_removed,
    dlc_project_dir,
    snapshot_recording,
    tmp_path,
):
    _, _, training_results = tutorial_results
    reindexed_model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed
    model = tutorial_applied["model"]
    config = lambda: kpms.load_config(project_dir)
    results_path = str(tmp_path / "results.h5")
    shutil.copy(
        os.path.join(project_dir, reindexed_model_name, "results.h5"), results_path
    )

    coordinates, confidences, _ = kpms.load_keypoints(
        os.path.join(dlc_project_dir, "videos", f"{snapshot_recording}.h5"),
        "deeplabcut",
    )
    coordinates, confidences = kpms.outlier_removal(
        coordinates, confidences, project_dir, overwrite=False, **config()
    )
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    with pytest.raises(RuntimeError) as error:
        kpms.apply_model(
            model, data, metadata, num_iters=5, results_path=results_path, **config()
        )
    assert "already contains results" in " ".join(str(error.value).split())

    results = kpms.apply_model(
        model,
        data,
        metadata,
        num_iters=5,
        results_path=results_path,
        overwrite=True,
        **config(),
    )
    saved = kpms.load_results(path=results_path)

    assert sorted(saved) == sorted(training_results)
    assert_tree_equal(saved[snapshot_recording], results[snapshot_recording])
    for name in training_results:
        if name != snapshot_recording:
            assert_tree_equal(saved[name], training_results[name], name)


@pytest.fixture
def sampler_calls(monkeypatch):
    """Keyword arguments of each call to jax-moseq's keypoint-SLDS
    resampling function, which still resamples."""
    calls = []
    resample_model = kpms.fitting.keypoint_slds.resample_model

    def record(*args, **kwargs):
        calls.append(kwargs)
        return resample_model(*args, **kwargs)

    monkeypatch.setattr(kpms.fitting.keypoint_slds, "resample_model", record)
    return calls


@pytest.mark.parametrize(
    "ar_only, num_iters, iterations_run, warns",
    [(True, 5, 1, True), (True, 1, 1, False), (True, None, 1, False), (False, 2, 2, False)],
    ids=["ar_only", "ar_only_one_iteration", "ar_only_default", "full_model"],
)
def test_ar_only_apply_model_runs_one_iteration(
    ar_only, num_iters, iterations_run, warns, one_recording, sampler_calls, recwarn
):
    """With `ar_only=True` each iteration draws the syllable labels
    independently given the same pose trajectories, so one is enough.
    `num_iters=None` leaves `num_iters` at its default."""
    config, data, metadata, models = one_recording
    iterations = {} if num_iters is None else {"num_iters": num_iters}

    kpms.apply_model(
        models[False],
        data,
        metadata,
        ar_only=ar_only,
        save_results=False,
        **iterations,
        **config,
    )

    messages = [" ".join(str(w.message).split()) for w in recwarn]
    stopped = [m for m in messages if "Stopping after 1 iteration" in m]
    assert len(sampler_calls) == iterations_run
    if warns:
        assert len(stopped) == 1
        assert f"instead of running {num_iters}" in stopped[0]
        assert "Set `num_iters=1` to silence this warning" in stopped[0]
    else:
        assert stopped == []


def test_full_model_apply_model_runs_500_iterations_by_default(
    one_recording, monkeypatch, recwarn
):
    config, data, metadata, models = one_recording
    calls = []

    def record(data, **kwargs):
        """Count the iteration and return the model unchanged."""
        calls.append(kwargs)
        return {key: kwargs[key] for key in models[False]}

    monkeypatch.setattr(kpms.fitting.keypoint_slds, "resample_model", record)
    kpms.apply_model(models[False], data, metadata, save_results=False, **config)

    assert len(calls) == 500
    assert not any(call["ar_only"] for call in calls)
    assert not any("Stopping after 1 iteration" in str(w.message) for w in recwarn)
