import numpy as np
import pytest

import keypoint_moseq as kpms

SNAPSHOT_FRAMES = 1000


def test_tutorial_format_data_shapes_and_metadata(
    tutorial_formatted_data, data_regression
):
    data, (keys, bounds) = tutorial_formatted_data

    data_regression.check(
        {
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in data.items()
            },
            "keys": [str(key) for key in keys],
            "bounds": np.asarray(bounds).tolist(),
        }
    )


def test_tutorial_format_data_values(
    tutorial_formatted_data, snapshot_recording, ndarrays_regression
):
    data, (keys, bounds) = tutorial_formatted_data
    segment = list(keys).index(snapshot_recording)

    ndarrays_regression.check(
        {
            "Y": np.asarray(data["Y"][segment, :SNAPSHOT_FRAMES]),
            "conf": np.asarray(data["conf"][segment, :SNAPSHOT_FRAMES]),
            "mask": np.asarray(data["mask"]),
        }
    )


def test_recording_lengths_from_metadata():
    rng = np.random.default_rng(0)
    lengths = {"rec_a": 2500, "rec_b": 1200}
    coordinates = {name: rng.uniform(0, 50, (n, 4, 2)) for name, n in lengths.items()}
    confidences = {name: np.ones((n, 4)) for name, n in lengths.items()}

    data, metadata = kpms.format_data(coordinates, confidences, seg_length=500)

    # Consecutive segments overlap, so the mask counts some frames twice.
    assert data["mask"].sum() > sum(lengths.values())
    assert kpms.get_recording_lengths(metadata) == lengths


def test_float32_keypoints_formatted_as_float64():
    # DeepLabCut 3 and SLEAP .slp files load as float32, and jax-moseq's
    # init_model rejects 32-bit arrays because keypoint-MoSeq runs JAX in
    # 64-bit mode.
    rng = np.random.default_rng(0)
    coordinates = {"rec": rng.uniform(0, 50, (1000, 4, 2)).astype(np.float32)}
    confidences = {"rec": rng.uniform(0, 1, (1000, 4)).astype(np.float32)}

    data, _ = kpms.format_data(coordinates, confidences, seg_length=500)

    assert {name: str(array.dtype) for name, array in data.items()} == {
        "mask": "float64",
        "Y": "float64",
        "conf": "float64",
    }


def test_format_data_rejects_recordings_no_longer_than_nlags():
    """A recording of `nlags` frames or fewer has no frame the model can
    assign a syllable to."""
    rng = np.random.default_rng(0)
    lengths = {"long": 100, "short_a": 5, "short_b": 6, "just_long_enough": 7}
    coordinates = {name: rng.uniform(0, 50, (n, 4, 2)) for name, n in lengths.items()}

    with pytest.raises(ValueError) as error:
        kpms.format_data(coordinates, ar_hypparams={"nlags": 6})

    message = " ".join(str(error.value).split())
    assert "Recording 'short_a' has 5 frames." in message
    assert "Recording 'short_b' has 6 frames." in message
    assert "'long'" not in message and "'just_long_enough'" not in message
    assert "every recording needs at least 7 frames" in message

    del coordinates["short_a"], coordinates["short_b"]
    kpms.format_data(coordinates, ar_hypparams={"nlags": 6})
