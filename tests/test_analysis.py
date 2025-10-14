"""
Test suite for keypoint-MoSeq analysis functionality

Tests result extraction, visualization, and analysis tools.
"""
import pytest
import numpy as np
from pathlib import Path
import pandas as pd


@pytest.mark.medium
@pytest.mark.notebook
def test_result_extraction(temp_project_dir, dlc_config, reduced_iterations):
    """Test extracting results from fitted model

    Expected duration: ~15 minutes (includes model fitting)
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup and fit model (abbreviated workflow)
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    config = lambda: kpms.load_config(project_dir)

    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ]
    })

    coordinates, confidences, _ = kpms.load_keypoints(project_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    pca = kpms.fit_pca(**data, **config())
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    config.update({'latent_dim': int(latent_dim)})

    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    config.update(hypparams)

    model = kpms.init_model(pca=pca, **data, **config())
    model = kpms.fit_model(
        model, pca=pca, **data, **config(),
        ar_only=True,
        num_iters=reduced_iterations['ar_hmm_iters']
    )
    model = kpms.fit_model(
        model, pca=pca, **data, **config(),
        num_iters=reduced_iterations['full_model_iters']
    )

    model_name = kpms.save_model(
        model, project_dir, metadata=metadata,
        pca=pca, config=config()
    )

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Extract results
    results = kpms.extract_results(model, metadata, project_dir, model_name, config())

    # Verify results structure
    assert 'syllable' in results, "Results missing syllable"
    assert 'centroid' in results, "Results missing centroid"
    assert 'heading' in results, "Results missing heading"
    assert 'latent_state' in results, "Results missing latent_state"

    # Verify all recordings present
    assert len(results['syllable']) > 0, "No syllables in results"

    # Check data types
    for recording_name, syllables in results['syllable'].items():
        assert isinstance(syllables, np.ndarray), f"Syllables not array for {recording_name}"
        assert syllables.dtype in [np.int32, np.int64], f"Syllables wrong dtype for {recording_name}"


@pytest.mark.medium
@pytest.mark.notebook
def test_csv_export(temp_project_dir, dlc_config, reduced_iterations):
    """Test CSV export of results

    Expected duration: ~15 minutes (includes model fitting)
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Run abbreviated workflow
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    config = lambda: kpms.load_config(project_dir)

    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ]
    })

    coordinates, confidences, _ = kpms.load_keypoints(project_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    pca = kpms.fit_pca(**data, **config())
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    config.update({'latent_dim': int(latent_dim)})

    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    config.update(hypparams)

    model = kpms.init_model(pca=pca, **data, **config())
    model = kpms.fit_model(
        model, pca=pca, **data, **config(),
        ar_only=True, num_iters=5
    )
    model = kpms.fit_model(
        model, pca=pca, **data, **config(),
        num_iters=10
    )

    model_name = kpms.save_model(
        model, project_dir, metadata=metadata,
        pca=pca, config=config()
    )

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
    results = kpms.extract_results(model, metadata, project_dir, model_name, config())

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

    expected_columns = ['syllable', 'centroid_x', 'centroid_y', 'heading']
    for col in expected_columns:
        assert col in df.columns, f"CSV missing column: {col}"

    # Check data validity
    assert len(df) > 0, "CSV is empty"
    assert df['syllable'].dtype in [np.int32, np.int64], "Syllable column wrong dtype"


@pytest.mark.medium
@pytest.mark.notebook
def test_trajectory_plots(temp_project_dir, dlc_config, reduced_iterations):
    """Test trajectory plot generation

    Expected duration: ~15 minutes (includes model fitting)
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Abbreviated workflow
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    config = lambda: kpms.load_config(project_dir)

    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ]
    })

    coordinates, confidences, _ = kpms.load_keypoints(project_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    pca = kpms.fit_pca(**data, **config())
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    config.update({'latent_dim': int(latent_dim)})

    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    config.update(hypparams)

    model = kpms.init_model(pca=pca, **data, **config())
    model = kpms.fit_model(model, pca=pca, **data, **config(), ar_only=True, num_iters=5)
    model = kpms.fit_model(model, pca=pca, **data, **config(), num_iters=10)

    model_name = kpms.save_model(model, project_dir, metadata=metadata, pca=pca, config=config())
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
    results = kpms.extract_results(model, metadata, project_dir, model_name, config())

    # Generate trajectory plots
    kpms.generate_trajectory_plots(
        coordinates, results, project_dir, model_name, config()
    )

    # Verify outputs
    trajectory_dir = Path(project_dir) / model_name / "trajectory_plots"
    assert trajectory_dir.exists(), "Trajectory plots directory not created"

    pdf_files = list(trajectory_dir.glob("*.pdf"))
    assert len(pdf_files) > 0, "No trajectory PDFs created"

    # Should have one PDF per syllable
    num_syllables = len(np.unique([v for v in results['syllable'].values() if v >= 0]))
    assert len(pdf_files) >= num_syllables * 0.8, "Too few trajectory plots"


@pytest.mark.slow
@pytest.mark.notebook
def test_grid_movies(temp_project_dir, dlc_config, reduced_iterations):
    """Test grid movie generation

    Expected duration: ~20 minutes (includes model fitting + video rendering)
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Abbreviated workflow
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    config = lambda: kpms.load_config(project_dir)

    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ]
    })

    coordinates, confidences, _ = kpms.load_keypoints(project_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    pca = kpms.fit_pca(**data, **config())
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    config.update({'latent_dim': int(latent_dim)})

    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    config.update(hypparams)

    model = kpms.init_model(pca=pca, **data, **config())
    model = kpms.fit_model(model, pca=pca, **data, **config(), ar_only=True, num_iters=5)
    model = kpms.fit_model(model, pca=pca, **data, **config(), num_iters=10)

    model_name = kpms.save_model(model, project_dir, metadata=metadata, pca=pca, config=config())
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
    results = kpms.extract_results(model, metadata, project_dir, model_name, config())

    # Generate grid movies
    kpms.generate_grid_movies(
        coordinates, results, project_dir, model_name,
        config=config(), fps=30, frame_path=None
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
def test_similarity_dendrogram(temp_project_dir, dlc_config, reduced_iterations):
    """Test similarity dendrogram generation

    Expected duration: ~15 minutes (includes model fitting)
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Abbreviated workflow
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    config = lambda: kpms.load_config(project_dir)

    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ]
    })

    coordinates, confidences, _ = kpms.load_keypoints(project_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    pca = kpms.fit_pca(**data, **config())
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    config.update({'latent_dim': int(latent_dim)})

    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    config.update(hypparams)

    model = kpms.init_model(pca=pca, **data, **config())
    model = kpms.fit_model(model, pca=pca, **data, **config(), ar_only=True, num_iters=5)
    model = kpms.fit_model(model, pca=pca, **data, **config(), num_iters=10)

    model_name = kpms.save_model(model, project_dir, metadata=metadata, pca=pca, config=config())
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Generate dendrogram
    kpms.generate_similarity_dendrogram(project_dir, model_name, config())

    # Verify output
    dendrogram_pdf = Path(project_dir) / model_name / "similarity_dendrogram.pdf"
    assert dendrogram_pdf.exists(), "Dendrogram PDF not created"

    dendrogram_png = Path(project_dir) / model_name / "similarity_dendrogram.png"
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
        'rec1': np.array([0, 0, 1, 1, 1, 2, 2, 0, 0]),
        'rec2': np.array([1, 1, 0, 0, 2, 2, 2, 1, 1, 1])
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
