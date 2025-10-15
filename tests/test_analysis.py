"""
Test suite for keypoint-MoSeq analysis functionality

Tests result extraction, visualization, and analysis tools.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.medium
@pytest.mark.notebook
def test_result_extraction(fitted_model, kpms):
    """Test extracting results from fitted model

    Expected duration: ~1 minute (uses fitted_model fixture)
    """
    from tests.conftest import assert_result_keys, load_path_from_model

    # Use fitted model from fixture
    project_dir = fitted_model["project_dir"]
    model = fitted_model["model"]
    model_name = fitted_model["model_name"]
    metadata = fitted_model["metadata"]
    config = fitted_model["config"]

    # Verify checkpoint exists
    checkpoint_path = load_path_from_model(
        project_dir, model_name, "checkpoint.h5"
    )
    assert checkpoint_path.exists(), "Checkpoint file not created"

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Delete results.h5 if it exists (from previous test using same fixture)
    results_h5_path = load_path_from_model(
        project_dir, model_name, "results.h5", delete_existing=True
    )

    # Extract results
    results = kpms.extract_results(
        model, metadata, project_dir, model_name, config
    )

    # Verify results structure - results is dict[recording_name -> dict[key -> data]]
    assert len(results) > 0, "No recordings in results"

    # Check that each recording has expected keys
    expected_keys = ["syllable", "centroid", "heading", "latent_state"]
    for recording_name, recording_results in results.items():
        assert_result_keys(recording_results, expected_keys)

        # Check data types
        assert isinstance(
            recording_results["syllable"], np.ndarray
        ), f"Syllables not array for {recording_name}"
        assert recording_results["syllable"].dtype in [
            np.int32,
            np.int64,
        ], f"Syllables wrong dtype for {recording_name}"


@pytest.mark.medium
@pytest.mark.notebook
def test_csv_export(fitted_model, kpms):
    """Test CSV export of results

    Expected duration: ~1 minute (uses fitted_model fixture)
    """
    from tests.conftest import load_path_from_model

    # Use fitted model from fixture
    project_dir = fitted_model["project_dir"]
    model = fitted_model["model"]
    model_name = fitted_model["model_name"]
    metadata = fitted_model["metadata"]
    config = fitted_model["config"]

    # Verify checkpoint exists
    checkpoint_path = load_path_from_model(
        project_dir, model_name, "checkpoint.h5"
    )
    assert checkpoint_path.exists(), "Checkpoint file not created"

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Delete results.h5 if it exists (from previous test using same fixture)
    results_h5_path = load_path_from_model(
        project_dir, model_name, "results.h5", delete_existing=True
    )

    results = kpms.extract_results(
        model, metadata, project_dir, model_name, config
    )

    # Export to CSV
    kpms.save_results_as_csv(results, project_dir, model_name)

    # Verify CSV files
    results_dir = Path(project_dir) / model_name / "results"
    assert results_dir.exists(), "Results directory not created"

    csv_files = list(results_dir.glob("*.csv"))
    assert len(csv_files) > 0, "No CSV files created"

    # Verify CSV structure
    first_csv = csv_files[0]
    df = pd.read_csv(first_csv)

    expected_columns = ["syllable", "centroid_x", "centroid_y", "heading"]
    for col in expected_columns:
        assert col in df.columns, f"CSV missing column: {col}"

    # Check data validity
    assert len(df) > 0, "CSV is empty"
    assert df["syllable"].dtype in [
        np.int32,
        np.int64,
    ], "Syllable column wrong dtype"


@pytest.mark.medium
@pytest.mark.notebook
def test_trajectory_plots(fitted_model, kpms):
    """Test trajectory plot generation

    Expected duration: ~1 minute (uses fitted_model fixture)
    """
    from tests.conftest import load_path_from_model

    # Use fitted model from fixture
    project_dir = fitted_model["project_dir"]
    model = fitted_model["model"]
    model_name = fitted_model["model_name"]
    metadata = fitted_model["metadata"]
    config = fitted_model["config"]
    coordinates = fitted_model["coordinates"]

    # Verify checkpoint exists
    checkpoint_path = load_path_from_model(
        project_dir, model_name, "checkpoint.h5"
    )
    assert checkpoint_path.exists(), "Checkpoint file not created"

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Delete results.h5 if it exists (from previous test using same fixture)
    results_h5_path = load_path_from_model(
        project_dir, model_name, "results.h5", delete_existing=True
    )

    results = kpms.extract_results(
        model, metadata, project_dir, model_name, config
    )

    # Generate trajectory plots
    kpms.generate_trajectory_plots(
        coordinates, results, project_dir, model_name, config
    )

    # Verify outputs
    trajectory_dir = Path(project_dir) / model_name / "trajectory_plots"
    assert trajectory_dir.exists(), "Trajectory plots directory not created"

    pdf_files = list(trajectory_dir.glob("*.pdf"))
    assert len(pdf_files) > 0, "No trajectory PDFs created"

    # Should have one PDF per syllable
    # Collect all syllables from all recordings
    all_syllables = []
    for recording_results in results.values():
        syllables = recording_results["syllable"]
        all_syllables.extend(syllables[syllables >= 0])
    num_syllables = len(np.unique(all_syllables))
    assert len(pdf_files) >= num_syllables * 0.8, "Too few trajectory plots"


@pytest.mark.slow
@pytest.mark.notebook
def test_grid_movies(fitted_model, kpms):
    """Test grid movie generation

    Expected duration: ~2 minutes (uses fitted_model fixture + video rendering)
    """
    from tests.conftest import load_path_from_model

    # Use fitted model from fixture
    project_dir = fitted_model["project_dir"]
    model = fitted_model["model"]
    model_name = fitted_model["model_name"]
    metadata = fitted_model["metadata"]
    config = fitted_model["config"]
    coordinates = fitted_model["coordinates"]

    # Verify checkpoint exists
    checkpoint_path = load_path_from_model(
        project_dir, model_name, "checkpoint.h5"
    )
    assert checkpoint_path.exists(), "Checkpoint file not created"

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Delete results.h5 if it exists (from previous test using same fixture)
    results_h5_path = load_path_from_model(
        project_dir, model_name, "results.h5", delete_existing=True
    )

    results = kpms.extract_results(
        model, metadata, project_dir, model_name, config
    )

    # Generate grid movies
    kpms.generate_grid_movies(
        coordinates,
        results,
        project_dir,
        model_name,
        config=config,
        fps=30,
        frame_path=None,
    )

    # Verify outputs
    grid_movies_dir = Path(project_dir) / model_name / "grid_movies"
    assert grid_movies_dir.exists(), "Grid movies directory not created"

    mp4_files = list(grid_movies_dir.glob("*.mp4"))
    assert len(mp4_files) > 0, "No grid movies created"

    # Verify file sizes (should not be empty)
    for mp4 in mp4_files:
        assert mp4.stat().st_size > 1000, f"Grid movie too small: {mp4}"


@pytest.mark.medium
@pytest.mark.notebook
def test_similarity_dendrogram(fitted_model, kpms):
    """Test similarity dendrogram generation

    Expected duration: ~1 minute (uses fitted_model fixture)
    """
    from tests.conftest import load_path_from_model

    # Use fitted model from fixture
    project_dir = fitted_model["project_dir"]
    model_name = fitted_model["model_name"]
    config = fitted_model["config"]

    # Verify checkpoint exists
    checkpoint_path = load_path_from_model(
        project_dir, model_name, "checkpoint.h5"
    )
    assert checkpoint_path.exists(), "Checkpoint file not created"

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Generate dendrogram
    kpms.generate_similarity_dendrogram(project_dir, model_name, config)

    # Verify output
    dendrogram_pdf = load_path_from_model(
        project_dir, model_name, "similarity_dendrogram.pdf"
    )
    assert dendrogram_pdf.exists(), "Dendrogram PDF not created"

    dendrogram_png = load_path_from_model(
        project_dir, model_name, "similarity_dendrogram.png"
    )
    assert dendrogram_png.exists(), "Dendrogram PNG not created"

    # Verify file sizes
    assert dendrogram_pdf.stat().st_size > 1000, "Dendrogram PDF too small"
    assert dendrogram_png.stat().st_size > 1000, "Dendrogram PNG too small"


@pytest.mark.quick
@pytest.mark.notebook
def test_syllable_statistics():
    """Test syllable statistics computation

    Expected duration: < 1 second
    """
    # Mock syllable data
    syllables = {
        "rec1": np.array([0, 0, 1, 1, 1, 2, 2, 0, 0]),
        "rec2": np.array([1, 1, 0, 0, 2, 2, 2, 1, 1, 1]),
    }

    # Count syllable occurrences
    all_syllables = np.concatenate([s for s in syllables.values()])
    unique, counts = np.unique(all_syllables, return_counts=True)

    # Verify
    assert len(unique) == 3, "Should have 3 unique syllables"
    assert sum(counts) == 19, "Total syllable count should be 19"

    # Check frequencies
    frequencies = counts / sum(counts)
    assert np.isclose(sum(frequencies), 1.0), "Frequencies should sum to 1"
