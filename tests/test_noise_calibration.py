import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent

import keypoint_moseq as kpms

NUM_ANNOTATIONS = 25


@pytest.fixture
def calibration_inputs(
    tutorial_project_dir, tutorial_outlier_removed, dlc_project_dir, monkeypatch
):
    """Project directory and outlier-removed keypoints for the calibration
    cell, run from the directory containing `dlc_project` so the config's
    relative `video_dir` resolves the way it does in the notebook."""
    monkeypatch.chdir(os.path.dirname(dlc_project_dir))
    _, coordinates, confidences = tutorial_outlier_removed
    yield tutorial_project_dir, coordinates, confidences
    plt.close("all")


def click(fig, x, y):
    """Left-click the figure's axes at data coordinates (x, y)."""
    x_display, y_display = fig.axes[0].transData.transform((x, y))
    event = MouseEvent("button_press_event", fig.canvas, x_display, y_display, button=MouseButton.LEFT)
    fig.canvas.callbacks.process("button_press_event", event)


def annotate(widget, num_annotations):
    """Click a fixed offset from the detected keypoint on successive frames,
    pressing Next after each click."""
    fig = plt.gcf()
    next_button = widget.children[0].children[1]
    for i in range(num_annotations):
        detected_x, detected_y = fig.axes[0].collections[0].get_offsets()[0]
        click(fig, detected_x + 1 + 3 * (i % 5), detected_y)
        next_button.click()


def annotations_path(project_dir):
    return os.path.join(project_dir, "error_annotations.csv")


def test_noise_calibration(calibration_inputs, dataframe_regression, num_regression):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)

    np.random.seed(0)
    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config())
    annotate(widget, NUM_ANNOTATIONS)

    info_label, usr_msg = widget.children[1].children
    assert info_label.value.endswith(f"Completed annotations = {NUM_ANNOTATIONS}")
    assert usr_msg.value == f"Annotations saved to {annotations_path(project_dir)}"

    annotations = pd.read_csv(annotations_path(project_dir), comment="#")
    assert len(annotations) == NUM_ANNOTATIONS
    dataframe_regression.check(annotations)
    num_regression.check(
        {
            "slope": [config()["error_estimator"]["slope"]],
            "intercept": [config()["error_estimator"]["intercept"]],
            "conf_threshold": [config()["conf_threshold"]],
        },
        basename="test_noise_calibration_config",
    )


def test_noise_calibration_does_not_save_before_twenty_annotations(
    calibration_inputs,
):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    original_config = config()

    np.random.seed(0)
    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config())
    annotate(widget, 19)

    _, usr_msg = widget.children[1].children
    assert usr_msg.value.startswith("Annotations not saved: complete 1 more")
    assert not os.path.exists(annotations_path(project_dir))
    assert config()["error_estimator"] == original_config["error_estimator"]


def test_noise_calibration_existing_annotations_declined(
    calibration_inputs, monkeypatch
):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    with open(annotations_path(project_dir), "w") as f:
        f.write("existing annotations")
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config())

    assert widget is None
    with open(annotations_path(project_dir)) as f:
        assert f.read() == "existing annotations"


def test_noise_calibration_existing_annotations_overwritten(
    calibration_inputs, monkeypatch
):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    with open(annotations_path(project_dir), "w") as f:
        f.write("existing annotations")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config())

    assert widget is not None
    assert not os.path.exists(annotations_path(project_dir))


def test_noise_calibration_rejects_keypoints_longer_than_video(calibration_inputs):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    name = sorted(coordinates)[0]
    coordinates, confidences = dict(coordinates), dict(confidences)
    coordinates[name] = np.concatenate([coordinates[name], coordinates[name][-1:]])
    confidences[name] = np.concatenate([confidences[name], confidences[name][-1:]])

    with pytest.raises(ValueError) as error:
        kpms.noise_calibration(project_dir, coordinates, confidences, **config())

    assert name in str(error.value)
    assert "up to" in str(error.value)


def test_noise_calibration_rejects_frame_indexes_missing_a_recording(calibration_inputs):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    first, *rest = sorted(coordinates)
    video_frame_indexes = {k: np.arange(len(coordinates[k])) for k in rest}

    with pytest.raises(ValueError) as error:
        kpms.noise_calibration(
            project_dir,
            coordinates,
            confidences,
            video_frame_indexes=video_frame_indexes,
            **config(),
        )

    assert first in str(error.value)
    assert "video_frame_indexes" in str(error.value)


class Checked(Exception):
    """Raised by the stand-in check_videos, which noise_calibration calls
    right after checking the matplotlib backend."""


def start_calibration(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise Checked

    monkeypatch.setattr(kpms.calibration, "check_videos", stop)
    coordinates = {"mouseA": np.zeros((10, 4, 2))}
    confidences = {"mouseA": np.ones((10, 4))}
    with pytest.raises(Checked):
        kpms.noise_calibration(
            str(tmp_path),
            coordinates,
            confidences,
            bodyparts=list("abcd"),
            use_bodyparts=list("abcd"),
            video_dir=str(tmp_path),
        )


def test_calibration_warns_without_the_ipympl_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")

    with pytest.warns(UserWarning) as record:
        start_calibration(tmp_path, monkeypatch)

    messages = [" ".join(str(w.message).split()) for w in record]
    message = next(m for m in messages if "ipympl" in m)
    assert "'agg'" in message
    assert "%matplotlib widget" in message


@pytest.mark.parametrize("backend", ["widget", "ipympl", "module://ipympl.backend_nbagg"])
def test_calibration_is_silent_with_the_ipympl_backend(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        start_calibration(tmp_path, monkeypatch)
