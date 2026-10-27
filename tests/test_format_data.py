import numpy as np

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
