import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

import keypoint_moseq.io as io
import keypoint_moseq.util as util


def make_coordinates_with_missing_and_outlier():
    """Create simple poses with one missing point and one spatial outlier."""
    base_x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    coordinates = np.zeros((7, len(base_x), 2))
    coordinates[:, :, 0] = base_x
    coordinates[1, 2] = np.nan
    coordinates[4, 4] = [20.0, 0.0]
    return coordinates


def test_get_distance_to_medoid_ignores_missing_keypoints():
    coordinates = np.array(
        [[[0.0, 0.0], [2.0, 0.0], [np.nan, np.nan]]]
    )

    distances = util.get_distance_to_medoid(coordinates)

    np.testing.assert_allclose(distances[0, :2], [1.0, 1.0])
    assert np.isnan(distances[0, 2])


def test_find_medoid_distance_outliers_handles_intermittent_nans():
    coordinates = make_coordinates_with_missing_and_outlier()

    result = util.find_medoid_distance_outliers(coordinates)

    assert np.isfinite(result["thresholds"]).all()
    expected_mask = np.zeros(coordinates.shape[:2], dtype=bool)
    expected_mask[4, 4] = True
    np.testing.assert_array_equal(result["mask"], expected_mask)


def test_nan_aware_calculations_preserve_finite_data_behavior():
    rng = np.random.default_rng(0)
    coordinates = rng.normal(size=(50, 5, 2))
    distances = util.get_distance_to_medoid(coordinates)
    medians = np.median(distances, axis=0)
    mads = np.median(np.abs(distances - medians[None, :]), axis=0)
    expected_thresholds = medians + 6.0 * mads
    expected_mask = distances > expected_thresholds[None, :]

    result = util.find_medoid_distance_outliers(coordinates)

    np.testing.assert_allclose(result["thresholds"], expected_thresholds)
    np.testing.assert_array_equal(result["mask"], expected_mask)


def test_outlier_plot_labels_upper_distance_threshold():
    traces = [np.ones((3, 1))]

    fig = util.plot_keypoint_traces(
        traces,
        bodyparts=["tail"],
        thresholds=np.array([2.0]),
    )

    ax = fig.axes[0]
    assert ax.get_ylabel() == "Distance to frame-wise median pose"
    assert "Upper distance threshold (2.00)" in [
        text.get_text() for text in ax.get_legend().get_texts()
    ]
    plt.close(fig)


def test_outlier_removal_interpolates_missing_and_outlier_points(
    monkeypatch, tmp_path
):
    raw_coordinates = make_coordinates_with_missing_and_outlier()
    coordinates = {"recording": raw_coordinates.copy()}
    confidences = {"recording": np.full(raw_coordinates.shape[:2], 0.9)}
    confidences["recording"][1, 2] = np.nan
    captured_plot_args = {}

    def capture_plot(
        project_dir,
        recording_name,
        original_coordinates,
        interpolated_coordinates,
        outlier_mask,
        outlier_thresholds,
        **kwargs,
    ):
        captured_plot_args.update(
            original_coordinates=original_coordinates,
            interpolated_coordinates=interpolated_coordinates,
            outlier_mask=outlier_mask,
            outlier_thresholds=outlier_thresholds,
        )

    monkeypatch.setattr(util, "plot_medoid_distance_outliers", capture_plot)

    cleaned_coordinates, cleaned_confidences = util.outlier_removal(
        coordinates,
        confidences,
        tmp_path,
        bodyparts=[f"keypoint{i}" for i in range(raw_coordinates.shape[1])],
    )
    cleaned_coordinates = cleaned_coordinates["recording"]
    cleaned_confidences = cleaned_confidences["recording"]

    assert np.isfinite(cleaned_coordinates).all()
    np.testing.assert_allclose(cleaned_coordinates[1, 2], [0.0, 0.0])
    np.testing.assert_allclose(cleaned_coordinates[4, 4], [2.0, 0.0])
    assert cleaned_confidences[1, 2] == 0
    assert cleaned_confidences[4, 4] == 0

    expected_interpolation_mask = np.zeros(raw_coordinates.shape[:2], dtype=bool)
    expected_interpolation_mask[1, 2] = True
    expected_interpolation_mask[4, 4] = True
    np.testing.assert_array_equal(
        captured_plot_args["outlier_mask"], expected_interpolation_mask
    )
    assert np.isfinite(captured_plot_args["outlier_thresholds"]).all()
    assert np.isnan(captured_plot_args["original_coordinates"][1, 2]).all()
    np.testing.assert_array_equal(
        captured_plot_args["interpolated_coordinates"], cleaned_coordinates
    )

    data, _ = util.format_data(
        {"recording": cleaned_coordinates.copy()},
        {"recording": cleaned_confidences.copy()},
        seg_length=len(cleaned_coordinates),
        added_noise_level=0,
    )
    formatted_confidences = np.asarray(data["conf"])[0]
    assert formatted_confidences[1, 2] == pytest.approx(1e-3)
    assert formatted_confidences[4, 4] == pytest.approx(1e-3)
    assert formatted_confidences[0, 0] == pytest.approx(0.901)


def test_all_missing_frame_is_ignored_for_thresholds_and_interpolated(
    monkeypatch, tmp_path
):
    raw_coordinates = make_coordinates_with_missing_and_outlier()
    raw_coordinates[1] = np.nan
    coordinates = {"recording": raw_coordinates.copy()}
    confidences = {"recording": np.full(raw_coordinates.shape[:2], 0.9)}
    confidences["recording"][1] = np.nan
    monkeypatch.setattr(util, "plot_medoid_distance_outliers", lambda *a, **k: None)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = util.find_medoid_distance_outliers(raw_coordinates)
        cleaned_coordinates, cleaned_confidences = util.outlier_removal(
            coordinates,
            confidences,
            tmp_path,
            bodyparts=[f"keypoint{i}" for i in range(raw_coordinates.shape[1])],
        )

    assert not [
        warning
        for warning in caught_warnings
        if "All-NaN slice encountered" in str(warning.message)
    ]
    assert np.isfinite(result["thresholds"]).all()
    assert result["mask"][4, 4]
    assert np.isfinite(cleaned_coordinates["recording"]).all()
    np.testing.assert_allclose(
        cleaned_coordinates["recording"][1, :, 0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
    )
    np.testing.assert_array_equal(cleaned_confidences["recording"][1], 0)


def test_sleap_h5_nans_are_interpolated_during_outlier_removal(
    monkeypatch, tmp_path
):
    raw_coordinates = make_coordinates_with_missing_and_outlier()
    raw_confidences = np.full(raw_coordinates.shape[:2], 0.9)
    raw_confidences[1, 2] = np.nan
    sleap_path = tmp_path / "predictions.h5"

    with h5py.File(sleap_path, "w") as file:
        file.create_dataset(
            "tracks", data=raw_coordinates[None].transpose((0, 3, 2, 1))
        )
        file.create_dataset(
            "point_scores", data=raw_confidences[None].transpose((0, 2, 1))
        )
        file.create_dataset(
            "node_names",
            data=np.asarray(
                [f"keypoint{i}".encode() for i in range(raw_coordinates.shape[1])]
            ),
        )

    coordinates, confidences, bodyparts = io.load_keypoints(str(sleap_path), "sleap")
    assert np.isnan(coordinates["predictions"][1, 2]).all()
    monkeypatch.setattr(util, "plot_medoid_distance_outliers", lambda *a, **k: None)

    cleaned_coordinates, cleaned_confidences = util.outlier_removal(
        coordinates,
        confidences,
        tmp_path,
        bodyparts=bodyparts,
    )

    assert np.isfinite(cleaned_coordinates["predictions"]).all()
    np.testing.assert_allclose(cleaned_coordinates["predictions"][1, 2], [0.0, 0.0])
    np.testing.assert_allclose(cleaned_coordinates["predictions"][4, 4], [2.0, 0.0])
    assert cleaned_confidences["predictions"][1, 2] == 0
    assert cleaned_confidences["predictions"][4, 4] == 0


def test_entirely_missing_keypoint_does_not_poison_other_thresholds(
    monkeypatch, tmp_path
):
    raw_coordinates = make_coordinates_with_missing_and_outlier()
    raw_coordinates[:, 2] = np.nan
    coordinates = {"recording": raw_coordinates.copy()}
    confidences = {"recording": np.full(raw_coordinates.shape[:2], 0.9)}
    confidences["recording"][:, 2] = np.nan
    monkeypatch.setattr(util, "plot_medoid_distance_outliers", lambda *a, **k: None)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = util.find_medoid_distance_outliers(raw_coordinates)
        cleaned_coordinates, cleaned_confidences = util.outlier_removal(
            coordinates,
            confidences,
            tmp_path,
            bodyparts=[f"keypoint{i}" for i in range(raw_coordinates.shape[1])],
        )

    assert not [
        warning
        for warning in caught_warnings
        if "All-NaN slice encountered" in str(warning.message)
    ]
    assert np.isnan(result["thresholds"][2])
    assert np.isfinite(np.delete(result["thresholds"], 2)).all()
    assert result["mask"][4, 4]
    np.testing.assert_array_equal(cleaned_coordinates["recording"][:, 2], 0)
    np.testing.assert_array_equal(cleaned_confidences["recording"][:, 2], 0)
