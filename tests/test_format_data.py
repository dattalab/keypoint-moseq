import numpy as np

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
