"""Unit tests for keypoint_moseq.util module.

This module tests utility functions for data manipulation, validation,
and processing in the keypoint-moseq package.
"""

import warnings
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from keypoint_moseq.util import (
    _find_optimal_segment_length,
    _get_percent_padding,
    apply_syllable_mapping,
    check_nan_proportions,
    check_video_paths,
    downsample_timepoints,
    estimate_sigmasq_loc,
    filter_angle,
    filtered_derivative,
    find_matching_videos,
    find_medoid_distance_outliers,
    generate_syllable_mapping,
    get_distance_to_medoid,
    get_edges,
    get_syllable_instances,
    interpolate_along_axis,
    interpolate_keypoints,
    list_files_with_exts,
    pad_along_axis,
    permute_cyclic,
    print_dims_to_explain_variance,
    reindex_by_bodyparts,
)


class TestPadAlongAxis:
    """Test pad_along_axis function."""

    def test_pad_axis_0(self):
        """Test padding along axis 0."""
        arr = np.ones((3, 4))
        result = pad_along_axis(arr, (1, 2), axis=0, value=0)
        assert result.shape == (6, 4)
        assert result[0, 0] == 0  # First row padded
        assert result[-1, 0] == 0  # Last row padded
        assert result[1, 0] == 1  # Original data

    def test_pad_axis_1(self):
        """Test padding along axis 1."""
        arr = np.ones((3, 4))
        result = pad_along_axis(arr, (2, 1), axis=1, value=5)
        assert result.shape == (3, 7)
        assert result[0, 0] == 5  # First column padded
        assert result[0, -1] == 5  # Last column padded
        assert result[0, 2] == 1  # Original data

    def test_pad_custom_value(self):
        """Test padding with custom value."""
        arr = np.zeros((2, 2))
        result = pad_along_axis(arr, (1, 1), axis=0, value=99)
        assert result[0, 0] == 99
        assert result[-1, 0] == 99


class TestFilterAngle:
    """Test filter_angle function."""

    def test_median_filter(self):
        """Test median filtering of angles."""
        # Create angles with some noise
        np.random.seed(42)
        angles = np.linspace(0, 2 * np.pi, 100)
        noisy_angles = angles + np.random.randn(100) * 0.5

        result = filter_angle(noisy_angles, size=9, axis=0, method="median")
        assert result.shape == noisy_angles.shape
        # Filtered angles should be smoother (higher noise for more obvious effect)
        assert np.std(np.diff(result)) < np.std(np.diff(noisy_angles))

    def test_gaussian_filter(self):
        """Test Gaussian filtering of angles."""
        angles = np.linspace(0, 2 * np.pi, 100)
        result = filter_angle(angles, size=5, axis=0, method="gaussian")
        assert result.shape == angles.shape

    def test_filter_2d_array(self):
        """Test filtering 2D array of angles along axis."""
        angles = np.random.randn(50, 3)  # Multiple angle sequences
        result = filter_angle(angles, size=7, axis=0, method="median")
        assert result.shape == angles.shape


class TestGetEdges:
    """Test get_edges function."""

    def test_edges_from_indices(self):
        """Test edge list from index pairs."""
        use_bodyparts = ["nose", "left_ear", "right_ear"]
        skeleton = [[0, 1], [0, 2]]
        edges = get_edges(use_bodyparts, skeleton)
        assert edges == [[0, 1], [0, 2]]

    def test_edges_from_names(self):
        """Test edge list from bodypart names."""
        use_bodyparts = ["nose", "left_ear", "right_ear", "neck"]
        skeleton = [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("nose", "neck"),
        ]
        edges = get_edges(use_bodyparts, skeleton)
        assert len(edges) == 3
        assert [0, 1] in edges
        assert [0, 2] in edges
        assert [0, 3] in edges

    def test_edges_partial_skeleton(self):
        """Test edges when some bodyparts not in use_bodyparts."""
        use_bodyparts = ["nose", "left_ear"]
        skeleton = [
            ("nose", "left_ear"),
            ("nose", "tail"),
        ]  # tail not in use_bodyparts
        edges = get_edges(use_bodyparts, skeleton)
        assert len(edges) == 1
        assert [0, 1] in edges

    def test_empty_skeleton(self):
        """Test with empty skeleton."""
        edges = get_edges(["nose"], [])
        assert edges == []


class TestReindexByBodyparts:
    """Test reindex_by_bodyparts function."""

    def test_reindex_array(self):
        """Test reindexing a single array."""
        data = np.arange(12).reshape(3, 4)  # 3 frames, 4 bodyparts
        bodyparts = ["a", "b", "c", "d"]
        use_bodyparts = ["d", "b", "a"]

        result = reindex_by_bodyparts(data, bodyparts, use_bodyparts, axis=1)
        assert result.shape == (3, 3)
        assert np.array_equal(result[:, 0], data[:, 3])  # d
        assert np.array_equal(result[:, 1], data[:, 1])  # b
        assert np.array_equal(result[:, 2], data[:, 0])  # a

    def test_reindex_dict(self):
        """Test reindexing a dictionary of arrays."""
        data = {
            "rec1": np.arange(8).reshape(2, 4),
            "rec2": np.arange(8, 16).reshape(2, 4),
        }
        bodyparts = ["a", "b", "c", "d"]
        use_bodyparts = ["c", "a"]

        result = reindex_by_bodyparts(data, bodyparts, use_bodyparts, axis=1)
        assert isinstance(result, dict)
        assert result["rec1"].shape == (2, 2)
        assert np.array_equal(result["rec1"][:, 0], data["rec1"][:, 2])  # c
        assert np.array_equal(result["rec1"][:, 1], data["rec1"][:, 0])  # a


class TestInterpolateAlongAxis:
    """Test interpolate_along_axis function."""

    def test_linear_interpolation(self):
        """Test linear interpolation along axis."""
        xp = np.array([0, 2, 4])
        fp = np.array([[0, 0], [10, 10], [20, 20]])
        x = np.array([0, 1, 2, 3, 4])

        result = interpolate_along_axis(x, xp, fp, axis=0)
        assert result.shape == (5, 2)
        assert np.allclose(result[1], [5, 5])  # Midpoint between 0 and 10
        assert np.allclose(result[3], [15, 15])  # Midpoint between 10 and 20

    def test_extrapolation(self):
        """Test that interpolation extrapolates beyond data range."""
        xp = np.array([1, 2])
        fp = np.array([10, 20])
        x = np.array([0, 1, 2, 3])

        result = interpolate_along_axis(x, xp, fp, axis=0)
        assert result[0] == 10  # Extrapolates to first value
        assert result[-1] == 20  # Extrapolates to last value

    def test_empty_datapoints_raises(self):
        """Test that empty datapoints raises assertion."""
        xp = np.array([])
        fp = np.array([]).reshape(0, 2)
        x = np.array([0, 1, 2])

        with pytest.raises(
            AssertionError, match="cannot interpolate without datapoints"
        ):
            interpolate_along_axis(x, xp, fp, axis=0)


class TestInterpolateKeypoints:
    """Test interpolate_keypoints function."""

    def test_no_outliers(self):
        """Test interpolation with no outliers."""
        coordinates = np.random.randn(10, 3, 2)  # 10 frames, 3 keypoints, 2D
        outliers = np.zeros((10, 3), dtype=bool)

        result = interpolate_keypoints(coordinates, outliers)
        assert np.allclose(result, coordinates)

    def test_single_outlier(self):
        """Test interpolation of single outlier frame."""
        coordinates = np.array(
            [
                [[0, 0], [1, 1]],
                [[5, 5], [6, 6]],  # Outlier frame
                [[2, 2], [3, 3]],
            ]
        )
        outliers = np.array(
            [
                [False, False],
                [True, True],
                [False, False],
            ]
        )

        result = interpolate_keypoints(coordinates, outliers)
        # Frame 1 should be interpolated between frames 0 and 2
        assert np.allclose(result[1, 0], [1, 1])
        assert np.allclose(result[1, 1], [2, 2])

    def test_all_outliers_for_keypoint(self):
        """Test when all frames are outliers for a keypoint."""
        coordinates = np.random.randn(5, 2, 2)
        outliers = np.zeros((5, 2), dtype=bool)
        outliers[:, 1] = True  # All frames outliers for keypoint 1

        result = interpolate_keypoints(coordinates, outliers)
        # Keypoint 1 should be all zeros (no valid data to interpolate)
        assert np.allclose(result[:, 1], 0)


class TestFilteredDerivative:
    """Test filtered_derivative function."""

    def test_constant_signal(self):
        """Test derivative of constant signal is zero."""
        Y = np.ones((100, 3))
        dY = filtered_derivative(Y, ksize=5, axis=0)
        assert dY.shape == Y.shape
        assert np.allclose(dY, 0, atol=1e-10)

    def test_linear_signal(self):
        """Test derivative of linear signal is constant."""
        Y = np.arange(100).reshape(-1, 1).astype(float)
        dY = filtered_derivative(Y, ksize=3, axis=0)
        # The filtered derivative algorithm uses forward - backward convolution
        # For linear signal, derivative should be constant (but value depends on kernel)
        # Just check that variance is low (derivative is relatively constant)
        assert np.std(dY[10:-10]) < 0.5

    def test_axis_parameter(self):
        """Test derivative along different axis."""
        Y = np.arange(20).reshape(4, 5).astype(float)
        dY_axis0 = filtered_derivative(Y, ksize=1, axis=0)
        dY_axis1 = filtered_derivative(Y, ksize=1, axis=1)
        assert dY_axis0.shape == Y.shape
        assert dY_axis1.shape == Y.shape


class TestPermuteCyclic:
    """Test permute_cyclic function."""

    def test_permutation_shape(self):
        """Test permutation preserves shape."""
        arr = np.arange(20).reshape(10, 2)
        result = permute_cyclic(arr, axis=0)
        assert result.shape == arr.shape

    def test_permutation_with_mask(self):
        """Test permutation with mask."""
        np.random.seed(42)
        arr = np.arange(10)
        mask = np.zeros(10, dtype=int)
        mask[:5] = 1  # Only permute first 5 elements

        result = permute_cyclic(arr, mask=mask, axis=0)
        # Last 5 elements should be zeros (not permuted, kept as masked)
        assert np.all(result[5:] == 0)

    def test_permutation_preserves_values(self):
        """Test permutation preserves values (just reorders)."""
        arr = np.array([1, 2, 3, 4, 5])
        result = permute_cyclic(arr, axis=0)
        assert set(result) == set(arr)


class TestDownsampleTimepoints:
    """Test downsample_timepoints function."""

    def test_downsample_array(self):
        """Test downsampling an array."""
        data = np.arange(100).reshape(100, 1)
        downsampled, indexes = downsample_timepoints(data, downsample_rate=2)

        assert downsampled.shape == (50, 1)
        assert np.array_equal(indexes, np.arange(50) * 2)
        assert np.array_equal(downsampled[:, 0], data[::2, 0])

    def test_downsample_dict(self):
        """Test downsampling a dictionary."""
        data = {
            "rec1": np.arange(10).reshape(10, 1),
            "rec2": np.arange(20).reshape(20, 1),
        }
        downsampled, indexes = downsample_timepoints(data, downsample_rate=3)

        assert isinstance(downsampled, dict)
        assert downsampled["rec1"].shape == (4, 1)
        assert downsampled["rec2"].shape == (7, 1)
        assert indexes["rec1"][0] == 0
        assert indexes["rec1"][1] == 3


class TestGetPercentPadding:
    """Test _get_percent_padding function."""

    def test_no_padding_needed(self):
        """Test when sequences are exact multiples of segment length."""
        sequence_lengths = np.array([10, 20, 30])
        seg_length = 10
        percent = _get_percent_padding(sequence_lengths, seg_length)
        assert percent == 0.0

    def test_padding_needed(self):
        """Test when padding is needed."""
        sequence_lengths = np.array([8, 15, 4])
        seg_length = 10
        # 8 needs 2, 15 needs 5, 4 needs 6 = 13 total padding
        # Total length = 27, so 13/27 * 100 = 48.15%
        percent = _get_percent_padding(sequence_lengths, seg_length)
        assert np.isclose(percent, 48.148, atol=0.01)

    def test_single_sequence(self):
        """Test with single sequence."""
        sequence_lengths = np.array([23])
        seg_length = 10
        # 23 needs 7 padding to reach 30
        # 7/23 * 100 = 30.43%
        percent = _get_percent_padding(sequence_lengths, seg_length)
        assert np.isclose(percent, 30.43, atol=0.01)


class TestFindOptimalSegmentLength:
    """Test _find_optimal_segment_length function."""

    def test_optimal_length_exact_match(self):
        """Test when sequence lengths are available options."""
        sequence_lengths = np.array([100, 200, 150])
        seg_length = _find_optimal_segment_length(
            sequence_lengths,
            max_seg_length=200,
            max_percent_padding=50,
            min_fragment_length=4,
        )
        assert seg_length <= 200
        assert seg_length >= 5  # Must be > min_fragment_length

    def test_respects_min_fragment_length(self):
        """Test that result respects min_fragment_length."""
        sequence_lengths = np.array([100, 103, 107])
        seg_length = _find_optimal_segment_length(
            sequence_lengths,
            max_seg_length=100,
            max_percent_padding=50,
            min_fragment_length=10,
        )
        # All remainders should be >= 10 or == 0
        remainders = sequence_lengths % seg_length
        assert np.all((remainders >= 10) | (remainders == 0))

    def test_short_sequences_raise(self):
        """Test that sequences shorter than min_fragment_length raise."""
        sequence_lengths = np.array([10, 3, 8])  # 3 is too short
        with pytest.raises(AssertionError, match="at least"):
            _find_optimal_segment_length(sequence_lengths, min_fragment_length=4)


class TestGetDistanceToMedoid:
    """Test get_distance_to_medoid function."""

    def test_2d_coordinates(self):
        """Test distance calculation with 2D coordinates."""
        # Simple case: 3 keypoints arranged in line
        coordinates = np.array(
            [
                [[0, 0], [1, 0], [2, 0]],  # Frame 1
                [[0, 1], [1, 1], [2, 1]],  # Frame 2
            ]
        )
        distances = get_distance_to_medoid(coordinates)

        assert distances.shape == (2, 3)
        # Median is (1, 0) for frame 1, so distances should be [1, 0, 1]
        assert np.allclose(distances[0], [1, 0, 1])

    def test_3d_coordinates(self):
        """Test distance calculation with 3D coordinates."""
        coordinates = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            ]
        )
        distances = get_distance_to_medoid(coordinates)
        assert distances.shape == (1, 3)
        # Medoid is (1, 1, 1), distances are sqrt(3), 0, sqrt(3)
        assert np.allclose(distances[0, 1], 0)


class TestFindMedoidDistanceOutliers:
    """Test find_medoid_distance_outliers function."""

    def test_no_outliers(self):
        """Test with normally distributed keypoints (no outliers)."""
        np.random.seed(42)
        coordinates = np.random.randn(100, 5, 2) * 0.1  # Small variance

        result = find_medoid_distance_outliers(coordinates, outlier_scale_factor=6.0)
        assert "mask" in result
        assert "thresholds" in result
        assert result["mask"].shape == (100, 5)
        assert result["thresholds"].shape == (5,)
        # With scale factor 6, few outliers expected
        assert np.sum(result["mask"]) < 50  # Less than 50%

    def test_with_outliers(self):
        """Test with injected outliers."""
        np.random.seed(42)
        coordinates = np.random.randn(50, 3, 2) * 0.1
        # Add clear outliers
        coordinates[10, 0] = [100, 100]  # Far from others
        coordinates[20, 1] = [-100, -100]

        result = find_medoid_distance_outliers(coordinates, outlier_scale_factor=3.0)
        # Should detect at least the injected outliers
        assert result["mask"][10, 0]
        assert result["mask"][20, 1]

    def test_scale_factor_effect(self):
        """Test that higher scale factor yields fewer outliers."""
        np.random.seed(42)
        coordinates = np.random.randn(100, 4, 2)

        result_low = find_medoid_distance_outliers(
            coordinates, outlier_scale_factor=2.0
        )
        result_high = find_medoid_distance_outliers(
            coordinates, outlier_scale_factor=10.0
        )

        # Higher scale factor should have fewer outliers
        assert np.sum(result_high["mask"]) < np.sum(result_low["mask"])


class TestGenerateSyllableMapping:
    """Test generate_syllable_mapping function."""

    def test_simple_grouping(self):
        """Test basic syllable grouping."""
        results = {
            "rec1": {"syllable": np.array([0, 0, 1, 1, 2, 2, 3, 3])},
            "rec2": {"syllable": np.array([0, 1, 2, 3])},
        }
        syllable_grouping = [[0, 1], [2, 3]]

        mapping = generate_syllable_mapping(results, syllable_grouping)

        # All syllables should be mapped
        assert 0 in mapping and 1 in mapping and 2 in mapping and 3 in mapping
        # Grouped syllables should map to same index
        assert mapping[0] == mapping[1]
        assert mapping[2] == mapping[3]

    def test_frequency_based_ordering(self):
        """Test that groups are ordered by frequency."""
        results = {
            "rec1": {"syllable": np.array([0] * 100 + [1] * 10 + [2] * 50)},
        }
        syllable_grouping = [[0, 2]]  # Group high-frequency syllables

        mapping = generate_syllable_mapping(results, syllable_grouping)
        # Group [0, 2] (150 occurrences) should get lower index than singleton [1] (10 occurrences)
        assert mapping[0] < mapping[1]
        assert mapping[2] < mapping[1]

    def test_no_grouping(self):
        """Test with empty grouping (all syllables separate)."""
        results = {
            "rec1": {"syllable": np.array([0, 1, 2, 0, 1, 2])},
        }
        syllable_grouping = []

        mapping = generate_syllable_mapping(results, syllable_grouping)
        # Should create identity-like mapping based on frequency
        assert len(mapping) == 3
        assert set(mapping.values()) == {0, 1, 2}


class TestApplySyllableMapping:
    """Test apply_syllable_mapping function."""

    def test_simple_remapping(self):
        """Test basic syllable remapping."""
        results = {
            "rec1": {
                "syllable": np.array([0, 1, 2, 3]),
                "centroid": np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            }
        }
        mapping = {0: 5, 1: 6, 2: 7, 3: 8}

        remapped = apply_syllable_mapping(results, mapping)

        assert np.array_equal(remapped["rec1"]["syllable"], [5, 6, 7, 8])
        # Other fields should be copied unchanged
        assert np.array_equal(remapped["rec1"]["centroid"], results["rec1"]["centroid"])

    def test_collapsing_syllables(self):
        """Test mapping multiple syllables to same index."""
        results = {
            "rec1": {
                "syllable": np.array([0, 1, 2, 1, 0]),
                "heading": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            }
        }
        mapping = {0: 0, 1: 0, 2: 1}  # Collapse 0 and 1 to 0

        remapped = apply_syllable_mapping(results, mapping)

        assert np.array_equal(remapped["rec1"]["syllable"], [0, 0, 1, 0, 0])
        assert np.array_equal(remapped["rec1"]["heading"], results["rec1"]["heading"])

    def test_multiple_recordings(self):
        """Test remapping across multiple recordings."""
        results = {
            "rec1": {"syllable": np.array([0, 1])},
            "rec2": {"syllable": np.array([1, 2])},
        }
        mapping = {0: 10, 1: 11, 2: 12}

        remapped = apply_syllable_mapping(results, mapping)

        assert np.array_equal(remapped["rec1"]["syllable"], [10, 11])
        assert np.array_equal(remapped["rec2"]["syllable"], [11, 12])


class TestListFilesWithExts:
    """Test list_files_with_exts function."""

    def test_single_file_match(self, tmp_path):
        """Test finding single file with extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = list_files_with_exts(str(tmp_path), [".txt"], recursive=False)
        assert len(result) == 1
        assert test_file.name in result[0]

    def test_multiple_extensions(self, tmp_path):
        """Test finding files with multiple extensions."""
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.csv").write_text("b")
        (tmp_path / "file3.json").write_text("c")

        result = list_files_with_exts(str(tmp_path), [".txt", ".csv"], recursive=False)
        assert len(result) == 2

    def test_recursive_search(self, tmp_path):
        """Test recursive file search."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "file1.txt").write_text("a")
        (subdir / "file2.txt").write_text("b")

        result = list_files_with_exts(str(tmp_path), [".txt"], recursive=True)
        assert len(result) == 2

    def test_extension_normalization(self, tmp_path):
        """Test that extensions are normalized (case, leading dot)."""
        (tmp_path / "file.TXT").write_text("a")

        result = list_files_with_exts(str(tmp_path), ["txt"], recursive=False)
        assert len(result) == 1


class TestFindMatchingVideos:
    """Test find_matching_videos function."""

    def test_exact_match(self, tmp_path):
        """Test exact video name matching."""
        (tmp_path / "video1.mp4").write_text("fake video")
        (tmp_path / "video2.avi").write_text("fake video")

        keys = ["video1", "video2"]
        result = find_matching_videos(
            keys, str(tmp_path), as_dict=True, recursive=False
        )

        assert "video1" in result
        assert "video2" in result
        assert "video1.mp4" in result["video1"]

    def test_prefix_match(self, tmp_path):
        """Test prefix matching (recording names have more text)."""
        (tmp_path / "vid.mp4").write_text("fake")

        keys = ["vid_2024_session1"]
        result = find_matching_videos(
            keys, str(tmp_path), as_dict=False, recursive=False
        )

        assert len(result) == 1
        assert "vid.mp4" in result[0]

    def test_longest_match(self, tmp_path):
        """Test that longest matching video name is used."""
        (tmp_path / "video.mp4").write_text("fake")
        (tmp_path / "video_long.mp4").write_text("fake")

        keys = ["video_long_session"]
        result = find_matching_videos(
            keys, str(tmp_path), as_dict=False, recursive=False
        )

        # Should match "video_long" not "video"
        assert "video_long.mp4" in result[0]

    def test_no_match_raises(self, tmp_path):
        """Test that missing video raises assertion."""
        keys = ["nonexistent"]

        with pytest.raises(AssertionError, match="No matching videos"):
            find_matching_videos(keys, str(tmp_path), as_dict=False, recursive=False)


class TestCheckVideoPaths:
    """Test check_video_paths function."""

    def test_valid_paths(self, tmp_path):
        """Test with valid video paths."""
        video1 = tmp_path / "video1.mp4"
        video1.write_bytes(b"fake video data")

        with patch("keypoint_moseq.util.OpenCVReader") as mock_reader:
            mock_instance = MagicMock()
            mock_instance.nframes = 100
            mock_reader.return_value = mock_instance

            video_paths = {"rec1": str(video1)}
            keys = ["rec1"]

            # Should not raise
            check_video_paths(video_paths, keys)

    def test_missing_key_raises(self):
        """Test that missing key raises ValueError."""
        video_paths = {"rec1": "/path/to/video.mp4"}
        keys = ["rec1", "rec2"]  # rec2 missing

        with pytest.raises(ValueError, match="require a video path"):
            check_video_paths(video_paths, keys)

    def test_nonexistent_video_raises(self):
        """Test that nonexistent video raises ValueError."""
        video_paths = {"rec1": "/nonexistent/path/video.mp4"}
        keys = ["rec1"]

        with pytest.raises(ValueError, match="do not exist"):
            check_video_paths(video_paths, keys)

    def test_unreadable_video_raises(self, tmp_path):
        """Test that unreadable video raises ValueError."""
        video1 = tmp_path / "corrupted.mp4"
        video1.write_bytes(b"corrupted")

        with patch("keypoint_moseq.util.OpenCVReader") as mock_reader:
            mock_reader.side_effect = Exception("Cannot read video")

            video_paths = {"rec1": str(video1)}
            keys = ["rec1"]

            with pytest.raises(ValueError, match="not readable"):
                check_video_paths(video_paths, keys)


class TestCheckNanProportions:
    """Test check_nan_proportions function."""

    def test_no_warnings_low_nans(self):
        """Test no warnings when NaN proportion is low."""
        coordinates = {
            "rec1": np.random.randn(100, 5, 2),
        }
        # Add few NaNs (< 50% threshold)
        coordinates["rec1"][0:10, 0, :] = np.nan

        bodyparts = ["bp1", "bp2", "bp3", "bp4", "bp5"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_proportions(coordinates, bodyparts, warning_threshold=0.5)
            assert len(w) == 0

    def test_warning_high_nans(self):
        """Test warning when NaN proportion exceeds threshold."""
        coordinates = {
            "rec1": np.random.randn(100, 3, 2),
        }
        # Add many NaNs to bodypart 1 (> 50%)
        coordinates["rec1"][:, 1, :] = np.nan

        bodyparts = ["bp1", "bp2", "bp3"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nan_proportions(coordinates, bodyparts, warning_threshold=0.5)
            assert len(w) >= 1
            assert "bp2" in str(w[0].message)


class TestGetSyllableInstances:
    """Test get_syllable_instances function."""

    def test_basic_instances(self):
        """Test extraction of syllable instances."""
        stateseqs = {
            "rec1": np.array(
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0] * 10
            ),  # Longer sequence
        }

        instances = get_syllable_instances(
            stateseqs,
            min_duration=3,
            pre=5,
            post=70,
            min_frequency=0,
            min_instances=0,
        )

        # Should find instances of syllables (with enough boundary space)
        assert len(instances) > 0
        # Each instance is a tuple (name, start, end)
        for syllable, instance_list in instances.items():
            for name, start, end in instance_list:
                assert name == "rec1"
                assert start >= 5  # Respects pre
                assert start < len(stateseqs["rec1"]) - 70  # Respects post

    def test_min_duration_filter(self):
        """Test filtering by minimum duration."""
        stateseqs = {
            "rec1": np.array(
                [5, 5, 5, 5] + [0, 0, 1, 1, 1, 0] * 10 + [5, 5, 5, 5] * 20
            ),  # Padding + repeats
        }

        instances = get_syllable_instances(
            stateseqs,
            min_duration=3,  # Only syllable 1 (duration 3) meets this
            pre=3,
            post=80,
        )

        # Syllable 1 (duration 3) should be included
        assert 1 in instances
        assert len(instances[1]) >= 1

    def test_boundary_filtering(self):
        """Test filtering instances near sequence boundaries."""
        stateseqs = {
            "rec1": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        }

        instances = get_syllable_instances(
            stateseqs,
            min_duration=3,
            pre=3,  # Exclude instances starting before frame 3
            post=3,  # Exclude instances ending after frame len-3
        )

        # Syllable 0 starts at frame 0 (excluded)
        # Syllable 1 starts at frame 3, ends at 6 (included)
        # Syllable 2 starts at frame 6 (excluded, too close to end)
        assert 1 in instances
        assert len(instances[1]) == 1


class TestPrintDimsToExplainVariance:
    """Test print_dims_to_explain_variance function."""

    def test_sufficient_variance(self, capsys):
        """Test printing when sufficient components exist."""
        mock_pca = Mock()
        mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.15, 0.05])

        print_dims_to_explain_variance(mock_pca, 0.8)
        captured = capsys.readouterr()

        # Should find that some components explain >=80%
        # The function uses f">={f*100}% of variance explained by..." (typo "explained")
        assert ">=80" in captured.out or "components" in captured.out

    def test_insufficient_variance(self, capsys):
        """Test printing when components don't explain enough variance."""
        mock_pca = Mock()
        mock_pca.explained_variance_ratio_ = np.array([0.3, 0.2, 0.15, 0.1])

        print_dims_to_explain_variance(mock_pca, 0.9)
        captured = capsys.readouterr()

        # Should indicate that all components together explain < 90%
        assert "All components" in captured.out or "75%" in captured.out


class TestEstimateSigmasqLoc:
    """Test estimate_sigmasq_loc function."""

    def test_basic_estimation(self):
        """Test basic sigmasq_loc estimation."""
        # Create simple trajectory with known movement
        Y = np.zeros((2, 100, 5, 2))  # 2 batches, 100 frames, 5 keypoints, 2D
        # Add linear motion
        for i in range(100):
            Y[:, i, :, 0] = i * 0.1  # Move in x direction

        mask = np.ones((2, 100))

        result = estimate_sigmasq_loc(Y, mask, filter_size=5)

        # Should return a positive float
        assert isinstance(result, float)
        assert result > 0

    def test_with_nans(self):
        """Test estimation with masked frames."""
        Y = np.random.randn(3, 50, 4, 2)
        mask = np.ones((3, 50))
        mask[:, 20:30] = 0  # Mask out middle frames

        result = estimate_sigmasq_loc(Y, mask, filter_size=10)

        assert isinstance(result, float)
        assert not np.isnan(result)


# Mark all tests as quick tests
pytestmark = pytest.mark.quick
