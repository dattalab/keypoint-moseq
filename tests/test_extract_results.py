import os

import numpy as np
import pandas as pd
import pytest

from jax_moseq.utils import batch

import keypoint_moseq as kpms
from model_utils import assert_tree_equal, structure

STATES_IN_RESULTS = {"latent_state": "x", "centroid": "v", "heading": "h"}


def frames_from_each_segment(keys, bounds):
    """(recording, start, stop) for each segment, where [start, stop) are the
    frames the segment supplies after unbatching. Segments overlap, and the
    later segment supplies the overlapping frames."""
    ranges = []
    for key, (start, end) in zip(keys, bounds):
        later_starts = [s for k, (s, _) in zip(keys, bounds) if k == key and s > start]
        ranges.append((key, start, min(later_starts, default=end)))
    return ranges


def results_csv_dir(project_dir, model_name):
    return os.path.join(project_dir, model_name, "results")


def test_results_structure(tutorial_results, data_regression):
    _, _, results = tutorial_results

    data_regression.check(structure(results))


def test_results_lengths_match_keypoints(tutorial_results, tutorial_keypoints):
    _, _, results = tutorial_results
    coordinates, _, _ = tutorial_keypoints

    assert sorted(results) == sorted(coordinates)
    for name, recording in results.items():
        for key, values in recording.items():
            assert len(values) == len(coordinates[name]), (name, key)


def test_results_match_model_states(tutorial_results):
    model, (keys, bounds), results = tutorial_results
    states = model["states"]
    nlags = states["x"].shape[1] - states["z"].shape[1]

    for segment, (name, start, stop) in enumerate(frames_from_each_segment(keys, bounds)):
        length = stop - start
        for result_key, state_key in STATES_IN_RESULTS.items():
            np.testing.assert_array_equal(
                results[name][result_key][start:stop],
                states[state_key][segment, :length],
                err_msg=f"{name} {result_key} segment {segment}",
            )
        # A segment has no syllables for its first `nlags` frames. The
        # previous segment supplies them, and at the start of a recording the
        # first syllable is repeated.
        np.testing.assert_array_equal(
            results[name]["syllable"][start + nlags : stop],
            states["z"][segment, : length - nlags],
            err_msg=f"{name} syllable segment {segment}",
        )
        if start == 0:
            first_syllables = np.repeat(states["z"][segment, 0], nlags)
        else:
            offset = start - bounds[segment - 1, 0] - nlags
            first_syllables = states["z"][segment - 1, offset : offset + nlags]
        np.testing.assert_array_equal(
            results[name]["syllable"][start : start + nlags],
            first_syllables,
            err_msg=f"{name} first syllables of segment {segment}",
        )


@pytest.mark.parametrize(
    "nlags, num_frames",
    [(3, 100), (5, 84)],
    ids=["default_nlags", "last_segment_shorter_than_nlags"],
)
def test_syllables_where_segments_overlap(nlags, num_frames):
    """The syllable inferred for each frame is the frame's index, so each
    frame's result shows whether a segment inferred it or it was copied.
    Segments start every 40 frames, so 84 frames leave a last segment of 4."""
    _, mask, metadata = batch({"rec": np.zeros((num_frames, 1))}, seg_length=40)
    _, bounds = metadata
    frame_indexes = bounds[:, :1] + np.arange(mask.shape[1])
    states = {
        "z": frame_indexes[:, nlags:],
        "x": np.zeros((*mask.shape, 2)),
        "v": np.zeros((*mask.shape, 2)),
        "h": np.zeros(mask.shape),
    }

    results = kpms.extract_results({"states": states}, metadata, save_results=False)

    expected = np.concatenate([np.repeat(nlags, nlags), np.arange(nlags, num_frames)])
    np.testing.assert_array_equal(results["rec"]["syllable"], expected)


def test_results_syllables_in_range(tutorial_results):
    model, _, results = tutorial_results
    num_states = model["hypparams"]["trans_hypparams"]["num_states"]

    for recording in results.values():
        assert recording["syllable"].min() >= 0
        assert recording["syllable"].max() < num_states


def test_results_h5_round_trip(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed
):
    _, _, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed

    assert_tree_equal(kpms.load_results(project_dir, model_name), results, "results")


def test_rerunning_extract_results_requires_overwrite(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed, tmp_path
):
    model, metadata, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed

    with pytest.raises(RuntimeError) as error:
        kpms.extract_results(model, metadata, project_dir, model_name)
    assert "already contains results" in " ".join(str(error.value).split())

    path = str(tmp_path / "results.h5")
    kpms.extract_results(model, metadata, path=path)
    kpms.extract_results(model, metadata, path=path, overwrite=True)
    assert_tree_equal(kpms.load_results(path=path), results, "results")


def test_results_csv_files(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed
):
    _, _, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed

    assert sorted(os.listdir(results_csv_dir(project_dir, model_name))) == sorted(
        f"{name}.csv" for name in results
    )


def test_results_csv_columns(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed, data_regression
):
    _, _, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed
    csv_dir = results_csv_dir(project_dir, model_name)

    headers = [
        list(pd.read_csv(os.path.join(csv_dir, f"{name}.csv"), nrows=0).columns)
        for name in results
    ]

    assert all(header == headers[0] for header in headers)
    data_regression.check({"columns": headers[0]})


def test_results_csv_values_match_results(
    tutorial_results, tutorial_reindexed, tutorial_outlier_removed
):
    _, _, results = tutorial_results
    model_name, _ = tutorial_reindexed
    project_dir, _, _ = tutorial_outlier_removed
    csv_dir = results_csv_dir(project_dir, model_name)

    for name, recording in results.items():
        table = pd.read_csv(
            os.path.join(csv_dir, f"{name}.csv"), float_precision="round_trip"
        )
        latent_columns = [f"latent_state {i}" for i in range(recording["latent_state"].shape[1])]

        np.testing.assert_array_equal(table["syllable"], recording["syllable"])
        # the CSV rounds each value to 4 decimals in float64, whatever the
        # results' precision
        rounded = lambda key: np.round(np.asarray(recording[key], np.float64), 4)  # noqa: E731
        np.testing.assert_array_equal(table[["centroid x", "centroid y"]], rounded("centroid"))
        np.testing.assert_array_equal(table["heading"], rounded("heading"))
        np.testing.assert_array_equal(table[latent_columns], rounded("latent_state"))


def reference_save_results_as_csv(results, save_dir, path_sep="-"):
    """save_results_as_csv as written before it formatted floats with orjson:
    one pandas table per recording, floats rounded to 4 decimals."""
    os.makedirs(save_dir, exist_ok=True)
    for key, recording in results.items():
        column_names, data = [], []
        if "syllable" in recording:
            column_names.append(["syllable"])
            data.append(recording["syllable"].reshape(-1, 1))
        if "centroid" in recording:
            d = recording["centroid"].shape[1]
            column_names.append(["centroid x", "centroid y", "centroid z"][:d])
            data.append(recording["centroid"])
        if "heading" in recording:
            column_names.append(["heading"])
            data.append(recording["heading"].reshape(-1, 1))
        if "latent_state" in recording:
            latent_dim = recording["latent_state"].shape[1]
            column_names.append([f"latent_state {i}" for i in range(latent_dim)])
            data.append(recording["latent_state"])
        df = pd.concat([pd.DataFrame(arr, columns=cols) for arr, cols in zip(data, column_names)], axis=1)
        for col in df.select_dtypes(include=[np.floating]).columns:
            df[col] = df[col].astype(float).round(4)
        df.to_csv(os.path.join(save_dir, key.replace(os.path.sep, path_sep)) + ".csv", index=False)


def synthetic_values(rng, shape, dtype=np.float64):
    """Values of both signs from 1e-6 to 1e15, so that some round to 0 or
    -0 at 4 decimals, mixed with whole numbers and other edge cases."""
    values = rng.choice([-1, 1], shape) * 10 ** rng.uniform(-6, 15, shape)
    edges = [0.0, -0.0, 1.0, -2.0, 0.0001, -0.00005, 0.1 + 0.2, 2.5, 1e15, 9.99999e14]
    flat = values.reshape(-1)
    flat[rng.choice(flat.size, len(edges), replace=False)] = edges
    return values.astype(dtype)


def test_results_csv_text_is_unchanged(tmp_path):
    """save_results_as_csv writes the same bytes as the pandas implementation
    for recordings with every combination it meets: a 2-D and a 3-D
    centroid, float32 values, some results missing, NaN and infinite values,
    values of 1e16 and more, no frames, and a name with a path separator."""
    rng = np.random.default_rng(0)
    frames = 3000

    def recording(centroid_dim=2, latent_dim=4, dtype=np.float64):
        return {
            "syllable": rng.integers(0, 100, frames),
            "centroid": synthetic_values(rng, (frames, centroid_dim), dtype),
            "heading": synthetic_values(rng, frames, dtype),
            "latent_state": synthetic_values(rng, (frames, latent_dim), dtype),
        }

    results = {
        "plain": recording(),
        "three_d": recording(centroid_dim=3, latent_dim=6),
        "float32": recording(dtype=np.float32),
        "partial": {k: v for k, v in recording().items() if k in ("syllable", "centroid")},
        "missing_values": recording(),
        "large_values": recording(),
        "no_frames": {k: v[:0] for k, v in recording().items()},
        os.path.join("session", "mouse"): recording(),
    }
    results["missing_values"]["heading"][[3, 50]] = [np.nan, np.inf]
    results["missing_values"]["centroid"][7, 1] = -np.inf
    results["large_values"]["latent_state"][11, 2] = 3.5e16

    kpms.save_results_as_csv(results, save_dir=str(tmp_path / "written"))
    reference_save_results_as_csv(results, str(tmp_path / "reference"))

    names = sorted(os.listdir(tmp_path / "reference"))
    assert sorted(os.listdir(tmp_path / "written")) == names
    for name in names:
        written = (tmp_path / "written" / name).read_bytes()
        assert written == (tmp_path / "reference" / name).read_bytes(), name
