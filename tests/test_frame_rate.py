"""Frame rates come from the project config, not an assumed 30 fps.

Velocities and syllable durations are converted from frames with the frame
rate. The analysis dataframes used to default to 30 fps whatever the
recording rate, so the tests here use 60 and 100 fps, where a 30 fps
assumption gives visibly wrong numbers.
"""

import os

import h5py
import numpy as np
import pytest
from jax_moseq.utils import get_durations

import keypoint_moseq as kpms
from analysis_utils import mean_complete_run_lengths, synthetic_results

NAMES = ["rec_a", "rec_b"]


def project_with_fps(tmp_path, fps):
    project_dir = str(tmp_path / "project")
    kpms.setup_project(project_dir, bodyparts=["nose", "tail"], skeleton=[], video_dir=".")
    kpms.update_config(project_dir, fps=fps)
    return project_dir


def expected_velocity(centroid, fps):
    """Centroid speed in pixels per second, 0 on the first frame."""
    return np.concatenate([[0], np.linalg.norm(np.diff(centroid, axis=0), axis=1) * fps])


def test_moseq_df_reads_the_frame_rate_from_the_project_config(tmp_path):
    project_dir = project_with_fps(tmp_path, 60)
    results = synthetic_results(NAMES)

    moseq_df = kpms.compute_moseq_df(project_dir, "model", results_dict=results)

    assert (moseq_df["fps"] == 60).all()
    np.testing.assert_allclose(
        moseq_df.loc[moseq_df["name"] == "rec_a", "velocity_px_s"],
        expected_velocity(results["rec_a"]["centroid"], 60),
    )


def test_moseq_df_reads_the_frame_rate_without_checking_the_bodyparts(tmp_path, capsys):
    project_dir = project_with_fps(tmp_path, 60)
    kpms.update_config(project_dir, anterior_bodyparts=["not_a_bodypart"])

    moseq_df = kpms.compute_moseq_df(project_dir, "model", results_dict=synthetic_results(NAMES))

    assert (moseq_df["fps"] == 60).all()
    assert "ACTION REQUIRED" not in capsys.readouterr().out


def test_explicit_frame_rate_is_used_over_the_config(tmp_path):
    project_dir = project_with_fps(tmp_path, 60)

    moseq_df = kpms.compute_moseq_df(
        project_dir, "model", results_dict=synthetic_results(NAMES), fps=100
    )

    assert (moseq_df["fps"] == 100).all()


def test_moseq_df_without_a_frame_rate_raises(tmp_path):
    with pytest.raises(ValueError) as error:
        kpms.compute_moseq_df(str(tmp_path), "model", results_dict=synthetic_results(NAMES))

    assert "fps" in str(error.value)
    assert "config" in str(error.value)


def test_stats_df_durations_use_the_moseq_df_frame_rate(tmp_path):
    results = synthetic_results(NAMES)
    moseq_df = kpms.compute_moseq_df(str(tmp_path), "model", results_dict=results, fps=60)

    stats_df = kpms.compute_stats_df(moseq_df, min_frequency=0.0)

    rows = stats_df[stats_df["name"] == "rec_a"]
    expected = mean_complete_run_lengths(results["rec_a"]["syllable"], rows["syllable"]) / 60
    np.testing.assert_allclose(rows["duration"], expected)


def test_stats_df_without_a_frame_rate_column_raises(tmp_path):
    moseq_df = kpms.compute_moseq_df(
        str(tmp_path), "model", results_dict=synthetic_results(NAMES), fps=60
    )

    with pytest.raises(ValueError) as error:
        kpms.compute_stats_df(moseq_df.drop(columns="fps"))

    assert "compute_moseq_df" in str(error.value)


def test_stats_df_with_several_frame_rates_raises(tmp_path):
    moseq_df = kpms.compute_moseq_df(
        str(tmp_path), "model", results_dict=synthetic_results(NAMES), fps=60
    )
    moseq_df.loc[moseq_df["name"] == "rec_b", "fps"] = 30

    with pytest.raises(ValueError) as error:
        kpms.compute_stats_df(moseq_df)

    assert "60" in str(error.value)
    assert "30" in str(error.value)


def median_durations_in_checkpoint(checkpoint_path, mask):
    """Median syllable duration, in frames, of each saved snapshot."""
    with h5py.File(checkpoint_path, "r") as f:
        iterations = sorted(int(i) for i in f["model_snapshots"])
        return [
            np.median(get_durations(np.array(f[f"model_snapshots/{i}/states/z"]), mask))
            for i in iterations
        ]


@pytest.fixture
def progress_inputs(tutorial_outlier_removed, tutorial_formatted_data, tutorial_arhmm):
    project_dir, _, _ = tutorial_outlier_removed
    data, _ = tutorial_formatted_data
    model, model_name, num_iters = tutorial_arhmm
    checkpoint_path = os.path.join(project_dir, model_name, "checkpoint.h5")
    medians = median_durations_in_checkpoint(checkpoint_path, np.array(data["mask"]))
    assert len(medians) > 1, "the median duration panel needs several snapshots"
    return model, data, checkpoint_path, num_iters, medians


def plotted_medians(axs):
    return axs[2].collections[0].get_offsets()[:, 1]


def test_progress_plot_shows_durations_in_seconds_with_a_frame_rate(progress_inputs):
    model, data, checkpoint_path, num_iters, medians = progress_inputs

    fig, axs = kpms.plot_progress(model, data, checkpoint_path, num_iters, savefig=False, fps=30)

    np.testing.assert_allclose(plotted_medians(axs), np.array(medians) / 30)
    assert "(s)" in axs[2].get_ylabel()
    assert "(s)" in axs[1].get_xlabel()


def test_progress_plot_shows_durations_in_frames_without_a_frame_rate(progress_inputs):
    model, data, checkpoint_path, num_iters, medians = progress_inputs

    fig, axs = kpms.plot_progress(model, data, checkpoint_path, num_iters, savefig=False)

    np.testing.assert_allclose(plotted_medians(axs), medians)
    assert "(frames)" in axs[2].get_ylabel()
    assert "(frames)" in axs[1].get_xlabel()


def test_fit_model_passes_the_frame_rate_to_the_progress_plot(
    tutorial_formatted_data, tutorial_model, tmp_path, monkeypatch
):
    data, metadata = tutorial_formatted_data
    real_plot_progress = kpms.fitting.plot_progress
    frame_rates = []

    def recording_plot_progress(*args, **kwargs):
        frame_rates.append(kwargs.get("fps"))
        return real_plot_progress(*args, **kwargs)

    monkeypatch.setattr(kpms.fitting, "plot_progress", recording_plot_progress)

    kpms.fit_model(
        tutorial_model, data, metadata, str(tmp_path), ar_only=True, num_iters=1, fps=60
    )

    assert frame_rates == [60]
