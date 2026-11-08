"""save_keypoints writes one csv per recording, with x and y (and z) columns
for each bodypart, plus a conf column when confidences are given."""

import numpy as np
import pandas as pd
import pytest

import keypoint_moseq as kpms


def random_keypoints(num_frames, num_keypoints, num_dims, seed=0):
    return np.random.default_rng(seed).uniform(0, 100, (num_frames, num_keypoints, num_dims))


@pytest.mark.parametrize("with_confidences", [True, False], ids=["conf", "no_conf"])
@pytest.mark.parametrize("num_dims", [2, 3], ids=["2D", "3D"])
@pytest.mark.parametrize("num_keypoints", [1, 2, 3, 9])
def test_saved_columns_and_values(tmp_path, num_keypoints, num_dims, with_confidences):
    coordinates = random_keypoints(10, num_keypoints, num_dims)
    confidences = np.random.default_rng(1).uniform(0, 1, (10, num_keypoints))
    bodyparts = [f"part{i}" for i in range(num_keypoints)]

    kpms.save_keypoints(
        str(tmp_path),
        {"mouse1": coordinates},
        {"mouse1": confidences} if with_confidences else None,
        bodyparts=bodyparts,
    )

    saved = pd.read_csv(tmp_path / "mouse1.csv")
    suffixes = ["x", "y", "z"][:num_dims] + (["conf"] if with_confidences else [])
    assert list(saved.columns) == [f"{bp}_{s}" for bp in bodyparts for s in suffixes]
    for i, bp in enumerate(bodyparts):
        for d, axis in enumerate("xyz"[:num_dims]):
            np.testing.assert_allclose(saved[f"{bp}_{axis}"], coordinates[:, i, d], rtol=1e-12)
        if with_confidences:
            np.testing.assert_allclose(saved[f"{bp}_conf"], confidences[:, i], rtol=1e-12)


def test_default_bodypart_names_and_path_separator(tmp_path):
    kpms.save_keypoints(str(tmp_path), {"session1/mouse1": random_keypoints(5, 3, 2)})

    saved = pd.read_csv(tmp_path / "session1-mouse1.csv")
    assert list(saved.columns) == [f"bodypart{i}_{axis}" for i in range(3) for axis in "xy"]
