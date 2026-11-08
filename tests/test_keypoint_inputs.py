"""Malformed keypoint dictionaries are rejected by the first function that
reads them, with a message naming the recording and the problem.

Custom loaders build `coordinates` and `confidences` by hand, and the
modeling notebook passes them to `check_videos`, `outlier_removal` and
`format_data` in that order, so each of the three checks them.
"""

import os

import numpy as np
import pytest

import keypoint_moseq as kpms

BODYPARTS = ["nose", "head", "spine", "tail"]
N_FRAMES = 50


def keypoints(n_frames=N_FRAMES, n_keypoints=len(BODYPARTS), seed=0):
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(10, 50, (n_frames, n_keypoints, 2))
    confidences = rng.uniform(0.5, 1, (n_frames, n_keypoints))
    return coordinates, confidences


def recordings(*names):
    coordinates, confidences = {}, {}
    for seed, name in enumerate(names):
        coordinates[name], confidences[name] = keypoints(seed=seed)
    return coordinates, confidences


def format_data_error(coordinates, confidences):
    with pytest.raises(ValueError) as error:
        kpms.format_data(coordinates, confidences, bodyparts=BODYPARTS)
    return str(error.value)


def test_well_formed_keypoints_pass():
    coordinates, confidences = recordings("rec_a", "rec_b")

    data, metadata = kpms.format_data(coordinates, confidences, bodyparts=BODYPARTS)

    assert sorted(metadata[0]) == ["rec_a", "rec_b"]


def test_non_string_recording_names_are_rejected():
    coordinates, confidences = keypoints()

    message = format_data_error({os.path.basename: coordinates}, {os.path.basename: confidences})

    assert "basename" in message
    assert "string" in message


def test_confidences_for_different_recordings_are_rejected():
    coordinates, confidences = recordings("rec_a", "rec_b")
    del confidences["rec_b"]

    message = format_data_error(coordinates, confidences)

    assert "rec_b" in message
    assert "confidences" in message


def test_confidences_with_the_wrong_shape_are_rejected():
    coordinates, confidences = recordings("rec_a")
    confidences["rec_a"] = confidences["rec_a"][:, :3]

    message = format_data_error(coordinates, confidences)

    assert "rec_a" in message
    assert f"({N_FRAMES}, 4, 2)" in message
    assert f"({N_FRAMES}, 3)" in message


def test_coordinates_with_the_wrong_number_of_dimensions_are_rejected():
    coordinates, confidences = recordings("rec_a")
    coordinates["rec_a"] = coordinates["rec_a"].reshape(N_FRAMES, -1)

    message = format_data_error(coordinates, confidences)

    assert "rec_a" in message
    assert f"({N_FRAMES}, 8)" in message


def test_non_numeric_coordinates_are_rejected():
    # A header row parsed as data turns a DeepLabCut array into text.
    coordinates, confidences = recordings("rec_a")
    coordinates["rec_a"] = coordinates["rec_a"].astype(str).astype(object)

    message = format_data_error(coordinates, confidences)

    assert "rec_a" in message
    assert "numeric" in message


def test_outlier_removal_rejects_non_string_recording_names(tmp_path):
    coordinates, confidences = keypoints()

    with pytest.raises(ValueError, match="string"):
        kpms.outlier_removal(
            {os.path.basename: coordinates},
            {os.path.basename: confidences},
            str(tmp_path),
            bodyparts=BODYPARTS,
        )


def test_check_videos_rejects_non_string_recording_names(tmp_path):
    coordinates, _ = keypoints()

    with pytest.raises(ValueError, match="string"):
        kpms.check_videos({os.path.basename: coordinates}, video_dir=str(tmp_path))
