"""
Unit tests for keypoint_moseq.analysis module

Tests core analysis functions without requiring full model fitting.
Focuses on statistical analysis, transition matrices, and data processing functions.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_syllable_data():
    """Mock syllable data for testing"""
    return {
        "rec1": np.array([0, 0, 1, 1, 1, 2, 2, 0, 0, 3, 3, 3]),
        "rec2": np.array([1, 1, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3]),
        "rec3": np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 0, 0]),
    }


@pytest.fixture
def mock_results_dict():
    """Mock results dictionary matching keypoint-MoSeq output format"""
    return {
        "rec1": {
            "syllable": np.array([0, 0, 1, 1, 1, 2, 2, 0]),
            "centroid": np.array(
                [
                    [0.0, 0.0],
                    [0.1, 0.1],
                    [0.2, 0.2],
                    [0.3, 0.3],
                    [0.4, 0.4],
                    [0.5, 0.5],
                    [0.6, 0.6],
                    [0.7, 0.7],
                ]
            ),
            "heading": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        },
        "rec2": {
            "syllable": np.array([1, 1, 0, 0, 2, 2, 2, 1]),
            "centroid": np.array(
                [
                    [1.0, 1.0],
                    [1.1, 1.1],
                    [1.2, 1.2],
                    [1.3, 1.3],
                    [1.4, 1.4],
                    [1.5, 1.5],
                    [1.6, 1.6],
                    [1.7, 1.7],
                ]
            ),
            "heading": np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]),
        },
    }


@pytest.fixture
def temp_project():
    """Create temporary project directory"""
    tmpdir = tempfile.mkdtemp(prefix="kpms_analysis_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# Test transition matrix functions


@pytest.mark.quick
def test_get_transitions():
    """Test syllable transition detection"""
    from keypoint_moseq.analysis import get_transitions

    # Simple case: clear transitions
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    transitions, locs = get_transitions(labels)

    assert len(transitions) == 3, "Should detect 3 transitions"
    assert len(locs) == 3, "Should have 3 transition locations"
    assert np.array_equal(transitions, [1, 2, 3]), "Transitions should be [1, 2, 3]"
    assert np.array_equal(locs, [2, 4, 7]), "Locations should be [2, 4, 7]"


@pytest.mark.quick
def test_get_transitions_no_transitions():
    """Test get_transitions with no transitions"""
    from keypoint_moseq.analysis import get_transitions

    # All same syllable
    labels = np.array([5, 5, 5, 5, 5])
    transitions, locs = get_transitions(labels)

    assert len(transitions) == 0, "Should detect no transitions"
    assert len(locs) == 0, "Should have no transition locations"


@pytest.mark.quick
def test_n_gram_transition_matrix():
    """Test n-gram transition matrix computation"""
    from keypoint_moseq.analysis import n_gram_transition_matrix

    # Simple bigram case
    labels = [0, 1, 2, 1, 0]
    trans_mat = n_gram_transition_matrix(labels, n=2, max_label=5)

    assert trans_mat.shape == (5, 5), "Transition matrix should be 5x5"
    assert trans_mat[0, 1] == 1.0, "0->1 transition should occur once"
    assert trans_mat[1, 2] == 1.0, "1->2 transition should occur once"
    assert trans_mat[2, 1] == 1.0, "2->1 transition should occur once"
    assert trans_mat[1, 0] == 1.0, "1->0 transition should occur once"


@pytest.mark.quick
def test_normalize_transition_matrix():
    """Test transition matrix normalization"""
    from keypoint_moseq.analysis import normalize_transition_matrix

    # Create simple 3x3 matrix
    matrix = np.array(
        [
            [1.0, 2.0, 1.0],
            [3.0, 0.0, 1.0],
            [0.0, 2.0, 2.0],
        ]
    )

    # Test bigram normalization
    norm_bigram = normalize_transition_matrix(matrix.copy(), "bigram")
    assert np.isclose(norm_bigram.sum(), 1.0), "Bigram normalization should sum to 1"

    # Test row normalization
    norm_rows = normalize_transition_matrix(matrix.copy(), "rows")
    assert np.allclose(norm_rows.sum(axis=1), 1.0), "Row sums should be 1"

    # Test column normalization
    norm_cols = normalize_transition_matrix(matrix.copy(), "columns")
    assert np.allclose(norm_cols.sum(axis=0), 1.0), "Column sums should be 1"

    # Test None normalization (no change)
    norm_none = normalize_transition_matrix(matrix.copy(), None)
    assert np.array_equal(
        norm_none, matrix
    ), "None normalization should not change matrix"


@pytest.mark.quick
def test_get_transition_matrix_single_recording(mock_syllable_data):
    """Test transition matrix for single recording"""
    from keypoint_moseq.analysis import get_transition_matrix

    syllables = mock_syllable_data["rec1"]
    trans_mats = get_transition_matrix(syllables, max_syllable=10, normalize="bigram")

    assert len(trans_mats) == 1, "Should return 1 transition matrix"
    assert trans_mats[0].shape == (10, 10), "Matrix should be 10x10"
    assert np.isclose(trans_mats[0].sum(), 1.0), "Normalized matrix should sum to 1"


@pytest.mark.quick
def test_get_transition_matrix_combined(mock_syllable_data):
    """Test combined transition matrix across recordings"""
    from keypoint_moseq.analysis import get_transition_matrix

    syllables = list(mock_syllable_data.values())
    trans_mat = get_transition_matrix(
        syllables, max_syllable=10, normalize="bigram", combine=True
    )

    assert isinstance(trans_mat, np.ndarray), "Combined matrix should be ndarray"
    assert trans_mat.shape == (10, 10), "Matrix should be 10x10"
    assert np.isclose(trans_mat.sum(), 1.0), "Normalized matrix should sum to 1"


# Test syllable name functions


@pytest.mark.quick
def test_get_syllable_names_no_file(temp_project):
    """Test get_syllable_names when syll_info.csv doesn't exist"""
    from keypoint_moseq.analysis import get_syllable_names

    model_name = "test_model"
    Path(temp_project, model_name).mkdir(parents=True)
    syllable_ixs = [0, 1, 2]

    names = get_syllable_names(temp_project, model_name, syllable_ixs)

    assert len(names) == 3, "Should return 3 names"
    assert names == ["0", "1", "2"], "Should return index strings when no file"


@pytest.mark.quick
def test_get_syllable_names_with_labels(temp_project):
    """Test get_syllable_names with custom labels"""
    from keypoint_moseq.analysis import get_syllable_names

    model_name = "test_model"
    model_dir = Path(temp_project, model_name)
    model_dir.mkdir(parents=True)

    # Create syll_info.csv with custom labels
    syll_info = pd.DataFrame(
        {
            "syllable": [0, 1, 2],
            "label": ["walk", "run", ""],
            "short_description": ["walking behavior", "running", ""],
        }
    )
    syll_info.to_csv(model_dir / "syll_info.csv", index=False)

    syllable_ixs = [0, 1, 2]
    names = get_syllable_names(temp_project, model_name, syllable_ixs)

    assert len(names) == 3, "Should return 3 names"
    assert names[0] == "0 (walk)", "Syllable 0 should have custom label"
    assert names[1] == "1 (run)", "Syllable 1 should have custom label"
    assert names[2] == "2", "Syllable 2 should only have index (empty label)"


# Test index generation


@pytest.mark.quick
def test_generate_index_new_file(temp_project, mock_results_dict):
    """Test index generation when file doesn't exist"""
    from unittest.mock import patch

    from keypoint_moseq.analysis import generate_index

    model_name = "test_model"
    model_dir = Path(temp_project, model_name)
    model_dir.mkdir(parents=True)
    index_filepath = Path(temp_project, "index.csv")

    # Mock load_results to return our mock data
    with patch("keypoint_moseq.analysis.load_results", return_value=mock_results_dict):
        generate_index(temp_project, model_name, str(index_filepath))

    assert index_filepath.exists(), "Index file should be created"

    # Verify contents
    index_df = pd.read_csv(index_filepath)
    assert len(index_df) == 2, "Should have 2 recordings"
    assert "name" in index_df.columns, "Should have 'name' column"
    assert "group" in index_df.columns, "Should have 'group' column"
    assert set(index_df["name"]) == {
        "rec1",
        "rec2",
    }, "Should have correct recording names"
    assert all(index_df["group"] == "default"), "All groups should be 'default'"


@pytest.mark.quick
def test_generate_index_append_missing(temp_project, mock_results_dict):
    """Test index generation appends missing recordings"""
    from unittest.mock import patch

    from keypoint_moseq.analysis import generate_index

    model_name = "test_model"
    model_dir = Path(temp_project, model_name)
    model_dir.mkdir(parents=True)
    index_filepath = Path(temp_project, "index.csv")

    # Create existing index with only rec1
    existing_index = pd.DataFrame({"name": ["rec1"], "group": ["experimental"]})
    existing_index.to_csv(index_filepath, index=False)

    # Mock load_results to return data with rec1 and rec2
    with patch("keypoint_moseq.analysis.load_results", return_value=mock_results_dict):
        generate_index(temp_project, model_name, str(index_filepath))

    # Verify rec2 was added
    index_df = pd.read_csv(index_filepath)
    assert len(index_df) == 2, "Should have 2 recordings now"
    assert "rec2" in index_df["name"].values, "rec2 should be added"
    assert (
        index_df[index_df["name"] == "rec1"]["group"].values[0] == "experimental"
    ), "rec1 group should be preserved"
    assert (
        index_df[index_df["name"] == "rec2"]["group"].values[0] == "default"
    ), "rec2 should have default group"


# Test syllable sorting functions


@pytest.mark.quick
def test_sort_syllables_by_stat_frequency():
    """Test sorting syllables by frequency"""
    from keypoint_moseq.analysis import sort_syllables_by_stat

    # Create mock stats dataframe
    stats_df = pd.DataFrame(
        {
            "syllable": [2, 0, 1, 3],
            "frequency": [0.3, 0.1, 0.4, 0.2],
            "duration": [1.0, 2.0, 1.5, 1.2],
        }
    )

    ordering, relabel_mapping = sort_syllables_by_stat(stats_df, stat="frequency")

    # For frequency, should sort by syllable index
    assert ordering == [
        0,
        1,
        2,
        3,
    ], "Frequency sorting should use syllable index order"
    assert relabel_mapping == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }, "Mapping should be identity for sorted indices"


@pytest.mark.quick
def test_sort_syllables_by_stat_duration():
    """Test sorting syllables by duration"""
    from keypoint_moseq.analysis import sort_syllables_by_stat

    # Create mock stats dataframe
    stats_df = pd.DataFrame(
        {
            "syllable": [0, 1, 2, 3],
            "frequency": [0.1, 0.4, 0.3, 0.2],
            "duration": [2.0, 1.5, 1.0, 1.2],
            "group": ["A", "A", "A", "A"],
        }
    )

    ordering, relabel_mapping = sort_syllables_by_stat(stats_df, stat="duration")

    # Should sort by duration descending
    assert ordering[0] == 0, "Syllable 0 has highest duration (2.0)"
    assert ordering[-1] == 2, "Syllable 2 has lowest duration (1.0)"


@pytest.mark.quick
def test_sort_syllables_by_stat_difference():
    """Test sorting syllables by difference between groups"""
    from keypoint_moseq.analysis import sort_syllables_by_stat_difference

    # Create mock stats dataframe with two groups
    stats_df = pd.DataFrame(
        {
            "syllable": [0, 0, 1, 1, 2, 2],
            "group": [
                "control",
                "experimental",
                "control",
                "experimental",
                "control",
                "experimental",
            ],
            "frequency": [0.2, 0.5, 0.3, 0.1, 0.5, 0.4],
            "name": ["rec1", "rec2", "rec1", "rec2", "rec1", "rec2"],
        }
    )

    ordering = sort_syllables_by_stat_difference(
        stats_df, "control", "experimental", stat="frequency"
    )

    # Syllable 0: exp(0.5) - ctrl(0.2) = +0.3 (highest increase)
    # Syllable 2: exp(0.4) - ctrl(0.5) = -0.1 (small decrease)
    # Syllable 1: exp(0.1) - ctrl(0.3) = -0.2 (largest decrease)
    assert ordering[0] == 0, "Syllable 0 should be first (largest increase)"
    assert ordering[-1] == 1, "Syllable 1 should be last (largest decrease)"


# Test Kruskal-Wallis helper functions


@pytest.mark.quick
def test_get_tie_correction():
    """Test tie correction computation for Kruskal-Wallis"""
    from keypoint_moseq.analysis import get_tie_correction

    # Case 1: No ties
    x = pd.Series([1, 2, 3, 4, 5])
    N_m = 5
    correction = get_tie_correction(x, N_m)
    assert correction == 0.0, "No ties should give 0 correction"

    # Case 2: Some ties
    x = pd.Series([1, 1, 2, 2, 2])
    N_m = 5
    correction = get_tie_correction(x, N_m)
    assert correction > 0.0, "Ties should give positive correction"


# Test moseq dataframe computation


@pytest.mark.quick
def test_compute_moseq_df_basic_structure(temp_project, mock_results_dict):
    """Test compute_moseq_df creates proper dataframe structure"""
    from unittest.mock import patch

    from keypoint_moseq.analysis import compute_moseq_df

    model_name = "test_model"
    model_dir = Path(temp_project, model_name)
    model_dir.mkdir(parents=True)

    # Create index file to avoid UnboundLocalError in compute_moseq_df
    index_df = pd.DataFrame({"name": ["rec1", "rec2"], "group": ["default", "default"]})
    index_df.to_csv(Path(temp_project, "index.csv"), index=False)

    with patch("keypoint_moseq.analysis.load_results", return_value=mock_results_dict):
        moseq_df = compute_moseq_df(
            temp_project, model_name, fps=30, smooth_heading=False
        )

    # Verify structure
    assert isinstance(moseq_df, pd.DataFrame), "Should return DataFrame"
    assert len(moseq_df) == 16, "Should have 16 rows (8 frames × 2 recordings)"

    # Check required columns
    required_cols = [
        "name",
        "centroid_x",
        "centroid_y",
        "heading",
        "angular_velocity",
        "velocity_px_s",
        "syllable",
        "frame_index",
        "group",
        "onset",
    ]
    for col in required_cols:
        assert col in moseq_df.columns, f"Missing column: {col}"

    # Check data types
    assert moseq_df["syllable"].dtype in [
        np.int32,
        np.int64,
    ], "Syllables should be integers"
    assert moseq_df["onset"].dtype == bool, "Onset should be boolean"


@pytest.mark.quick
def test_compute_moseq_df_onset_detection(temp_project, mock_results_dict):
    """Test syllable onset detection in compute_moseq_df"""
    from unittest.mock import patch

    from keypoint_moseq.analysis import compute_moseq_df

    model_name = "test_model"
    model_dir = Path(temp_project, model_name)
    model_dir.mkdir(parents=True)

    # Create index file to avoid UnboundLocalError in compute_moseq_df
    index_df = pd.DataFrame({"name": ["rec1", "rec2"], "group": ["default", "default"]})
    index_df.to_csv(Path(temp_project, "index.csv"), index=False)

    with patch("keypoint_moseq.analysis.load_results", return_value=mock_results_dict):
        moseq_df = compute_moseq_df(
            temp_project, model_name, fps=30, smooth_heading=False
        )

    # Check onset detection
    # First frame of each recording should have onset=True
    rec1_data = moseq_df[moseq_df["name"] == "rec1"]
    assert rec1_data.iloc[0]["onset"], "First frame should have onset"

    # Frames where syllable changes should have onset=True
    syllables = rec1_data["syllable"].values
    onsets = rec1_data["onset"].values

    # Check transitions
    for i in range(1, len(syllables)):
        if syllables[i] != syllables[i - 1]:
            assert onsets[i], f"Frame {i} should have onset (transition)"


# Test validation function


@pytest.mark.quick
def test_validate_and_order_syll_stats_params():
    """Test parameter validation and ordering"""
    from keypoint_moseq.analysis import _validate_and_order_syll_stats_params

    # Create mock dataframe
    complete_df = pd.DataFrame(
        {
            "syllable": [0, 1, 2, 0, 1, 2],
            "group": ["control", "control", "control", "exp", "exp", "exp"],
            "frequency": [0.3, 0.2, 0.5, 0.4, 0.3, 0.3],
            "duration": [1.0, 2.0, 1.5, 1.2, 1.8, 1.3],
        }
    )

    # Test basic validation
    ordering, groups, colors, figsize = _validate_and_order_syll_stats_params(
        complete_df, stat="frequency", order="stat"
    )

    assert len(ordering) > 0, "Should return ordering"
    assert len(groups) > 0, "Should return groups"
    assert len(colors) == len(groups), "Should have color for each group"
    assert figsize == (10, 5), "Should return figsize"


@pytest.mark.quick
def test_validate_and_order_invalid_stat():
    """Test validation with invalid statistic"""
    from keypoint_moseq.analysis import _validate_and_order_syll_stats_params

    complete_df = pd.DataFrame(
        {
            "syllable": [0, 1, 2],
            "group": ["A", "A", "A"],
            "frequency": [0.3, 0.2, 0.5],
        }
    )

    with pytest.raises(ValueError, match="Invalid stat entered"):
        _validate_and_order_syll_stats_params(
            complete_df, stat="nonexistent_column", order="stat"
        )


@pytest.mark.quick
def test_validate_and_order_diff_without_groups():
    """Test validation for diff ordering without proper groups"""
    from keypoint_moseq.analysis import _validate_and_order_syll_stats_params

    complete_df = pd.DataFrame(
        {
            "syllable": [0, 1, 2],
            "group": ["A", "A", "A"],
            "frequency": [0.3, 0.2, 0.5],
        }
    )

    with pytest.raises(ValueError, match="Attempting to sort by"):
        _validate_and_order_syll_stats_params(
            complete_df,
            stat="frequency",
            order="diff",
            ctrl_group="B",  # Group B doesn't exist
            exp_group="C",
        )


# Test summary statistics computation


@pytest.mark.quick
def test_compute_stats_df_basic(temp_project, mock_results_dict):
    """Test basic stats dataframe computation"""
    from unittest.mock import patch

    from keypoint_moseq.analysis import compute_moseq_df, compute_stats_df

    model_name = "test_model"
    model_dir = Path(temp_project, model_name)
    model_dir.mkdir(parents=True)

    # Create index file
    index_df = pd.DataFrame(
        {"name": ["rec1", "rec2"], "group": ["control", "experimental"]}
    )
    index_df.to_csv(Path(temp_project, "index.csv"), index=False)

    with patch("keypoint_moseq.analysis.load_results", return_value=mock_results_dict):
        moseq_df = compute_moseq_df(
            temp_project, model_name, fps=30, smooth_heading=False
        )
        stats_df = compute_stats_df(
            temp_project, model_name, moseq_df, min_frequency=0.0, fps=30
        )

    # Verify structure
    assert isinstance(stats_df, pd.DataFrame), "Should return DataFrame"
    assert "syllable" in stats_df.columns, "Should have syllable column"
    assert "frequency" in stats_df.columns, "Should have frequency column"
    assert "duration" in stats_df.columns, "Should have duration column"
    assert "group" in stats_df.columns, "Should have group column"
