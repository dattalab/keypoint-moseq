"""
Test suite for the keypoint-MoSeq colab workflow

This test suite validates the complete workflow from the colab notebook,
adapted for pytest with appropriate fixtures and assertions.
"""
import pytest
import os
from pathlib import Path
import numpy as np
import h5py


@pytest.mark.integration
@pytest.mark.notebook
def test_complete_workflow(temp_project_dir, dlc_config, reduced_iterations):
    """Test the complete keypoint-MoSeq workflow end-to-end

    This test runs the full pipeline with reduced iterations suitable for CI/CD.
    Expected duration: ~15 minutes
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Step 1: Setup project
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    assert Path(project_dir, "config.yml").exists(), "Config file not created"

    # Step 2: Update config
    config = lambda: kpms.load_config(project_dir)
    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ],
        'anterior_bodyparts': ['head', 'nose', 'right ear', 'left ear'],
        'posterior_bodyparts': ['spine4', 'spine3', 'spine2', 'spine1'],
        'seg_length': 5
    })

    # Step 3: Load keypoints
    coordinates, confidences, bodyparts = kpms.load_keypoints(project_dir, 'deeplabcut')
    assert len(coordinates) > 0, "No keypoints loaded"
    assert len(bodyparts) == 9, f"Expected 9 bodyparts, got {len(bodyparts)}"

    # Step 4: Format data
    data, metadata = kpms.format_data(coordinates, confidences, **config())
    assert 'coordinates' in data, "Formatted data missing coordinates"
    assert 'heading' in data, "Formatted data missing heading"

    # Step 5: Outlier removal
    outlier_detection_params = {
        'num_points': 30, 'cutoff': 1,
        'use_bodyparts': config()['use_bodyparts']
    }
    data = kpms.keypoint_distance_outliers(
        data, metadata, project_dir,
        generate_plots=True,
        **outlier_detection_params
    )
    qa_dir = Path(project_dir) / "QA" / "plots" / "keypoint_distance_outliers"
    assert qa_dir.exists(), "QA plots directory not created"

    # Step 6: Reformat data
    data, metadata = kpms.format_data(data['coordinates'], **config())
    assert len(data) == len(metadata), "Data/metadata length mismatch"

    # Step 7: Skip calibration (not needed for minimal dataset)
    # Manual calibration widget would go here in interactive mode

    # Step 8: Fit PCA
    pca = kpms.fit_pca(**data, **config())
    pca_path = Path(project_dir) / "pca.p"
    assert pca_path.exists(), "PCA model not saved"

    # Step 9: Update latent dimensions
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    assert latent_dim >= 3, f"Expected at least 3 PCs, got {latent_dim}"
    config.update({'latent_dim': int(latent_dim)})

    # Step 10: Estimate hyperparameters
    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    config.update(hypparams)

    # Step 11: Initialize model
    model = kpms.init_model(pca=pca, **data, **config())
    assert model is not None, "Model initialization failed"

    # Step 12: Fit AR-HMM with reduced iterations
    model = kpms.fit_model(
        model, pca=pca, **data, **config(),
        ar_only=True,
        num_iters=reduced_iterations['ar_hmm_iters']
    )

    # Step 13: Fit full model with reduced iterations
    model = kpms.fit_model(
        model, pca=pca, **data, **config(),
        num_iters=reduced_iterations['full_model_iters']
    )

    # Step 14: Save results
    model_name = kpms.save_model(
        model, project_dir, metadata=metadata,
        pca=pca, config=config()
    )
    assert model_name is not None, "Model saving failed"

    checkpoint_path = Path(project_dir) / model_name / "checkpoint.h5"
    assert checkpoint_path.exists(), "Checkpoint file not created"

    # Step 15: Reindex syllables
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Step 16: Extract results
    results = kpms.extract_results(model, metadata, project_dir, model_name, config())
    assert 'syllable' in results, "Results missing syllable labels"

    results_h5_path = Path(project_dir) / model_name / "results.h5"
    assert results_h5_path.exists(), "Results HDF5 not created"

    # Validate results structure
    with h5py.File(results_h5_path, 'r') as f:
        recording_keys = list(f.keys())
        assert len(recording_keys) > 0, "No recordings in results"

        first_recording = f[recording_keys[0]]
        assert 'syllable' in first_recording, "Results missing syllable dataset"
        assert 'centroid' in first_recording, "Results missing centroid dataset"
        assert 'heading' in first_recording, "Results missing heading dataset"
        assert 'latent_state' in first_recording, "Results missing latent_state dataset"

    # Step 17: Save as CSV
    kpms.save_results_as_csv(results, project_dir, model_name)
    results_dir = Path(project_dir) / model_name / "results"
    assert results_dir.exists(), "Results CSV directory not created"
    csv_files = list(results_dir.glob("*.csv"))
    assert len(csv_files) > 0, "No CSV files created"

    # Step 18: Generate visualizations
    kpms.generate_trajectory_plots(
        coordinates, results, project_dir, model_name, config()
    )
    trajectory_dir = Path(project_dir) / model_name / "trajectory_plots"
    assert trajectory_dir.exists(), "Trajectory plots directory not created"

    num_syllables = len(np.unique([v for v in results['syllable'].values() if v >= 0]))
    assert num_syllables > 0, "No syllables identified"

    # Check for trajectory plots
    pdf_plots = list(trajectory_dir.glob("*.pdf"))
    assert len(pdf_plots) > 0, "No trajectory PDFs created"

    # Grid movies
    kpms.generate_grid_movies(
        coordinates, results, project_dir, model_name,
        config=config(), fps=30, frame_path=None
    )
    grid_movies_dir = Path(project_dir) / model_name / "grid_movies"
    assert grid_movies_dir.exists(), "Grid movies directory not created"

    mp4_files = list(grid_movies_dir.glob("*.mp4"))
    assert len(mp4_files) > 0, "No grid movies created"

    # Similarity dendrogram
    kpms.generate_similarity_dendrogram(
        project_dir, model_name, config()
    )
    dendrogram_pdf = Path(project_dir) / model_name / "similarity_dendrogram.pdf"
    assert dendrogram_pdf.exists(), "Similarity dendrogram not created"

    print(f"\n✅ Complete workflow test passed!")
    print(f"   Model: {model_name}")
    print(f"   Syllables identified: {num_syllables}")
    print(f"   Trajectory plots: {len(pdf_plots)}")
    print(f"   Grid movies: {len(mp4_files)}")
    print(f"   CSV files: {len(csv_files)}")


@pytest.mark.quick
@pytest.mark.notebook
def test_project_setup(temp_project_dir, dlc_config):
    """Test project setup and configuration

    Expected duration: < 1 second
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Test setup
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    # Verify files created
    config_path = Path(project_dir, "config.yml")
    assert config_path.exists(), "Config file not created"

    # Update config with valid bodyparts before loading
    # (setup_project creates placeholders that need to be updated)
    kpms.update_config(
        project_dir,
        use_bodyparts=[
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ],
        anterior_bodyparts=['head', 'nose', 'right ear', 'left ear'],
        posterior_bodyparts=['spine4', 'spine3', 'spine2', 'spine1'],
    )

    # Test config loading after update
    config = kpms.load_config(project_dir)
    assert 'bodyparts' in config, "Config missing bodyparts"
    assert 'fps' in config, "Config missing fps"
    assert 'use_bodyparts' in config, "Config missing use_bodyparts"
    assert len(config['use_bodyparts']) == 8, "Wrong number of use_bodyparts"


@pytest.mark.quick
@pytest.mark.notebook
def test_load_keypoints(temp_project_dir, dlc_config, dlc_videos_dir):
    """Test keypoint loading from DLC data

    Expected duration: < 1 second
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    # Load keypoints from DLC videos directory (not project_dir)
    coordinates, confidences, bodyparts = kpms.load_keypoints(dlc_videos_dir, 'deeplabcut')

    # Verify data structure
    assert len(coordinates) > 0, "No coordinates loaded"
    assert len(confidences) > 0, "No confidences loaded"
    assert len(bodyparts) == 9, f"Expected 9 bodyparts, got {len(bodyparts)}"

    # Check data types
    first_recording = next(iter(coordinates.keys()))
    assert isinstance(coordinates[first_recording], np.ndarray), "Coordinates not numpy array"
    assert coordinates[first_recording].ndim == 3, "Coordinates wrong shape"


@pytest.mark.medium
@pytest.mark.notebook
def test_format_and_outlier_detection(temp_project_dir, dlc_config):
    """Test data formatting and outlier detection

    Expected duration: ~1 minute
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    config = lambda: kpms.load_config(project_dir)

    # Update config
    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ]
    })

    # Load and format
    coordinates, confidences, bodyparts = kpms.load_keypoints(project_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    # Test outlier detection
    outlier_params = {
        'num_points': 30, 'cutoff': 1,
        'use_bodyparts': config()['use_bodyparts']
    }
    data_clean = kpms.keypoint_distance_outliers(
        data, metadata, project_dir,
        generate_plots=True,
        **outlier_params
    )

    # Verify outputs
    assert 'coordinates' in data_clean, "Cleaned data missing coordinates"

    qa_dir = Path(project_dir) / "QA" / "plots" / "keypoint_distance_outliers"
    assert qa_dir.exists(), "QA directory not created"

    plot_files = list(qa_dir.glob("*.png"))
    assert len(plot_files) > 0, "No QA plots generated"


@pytest.mark.medium
@pytest.mark.notebook
def test_pca_fitting(temp_project_dir, dlc_config):
    """Test PCA model fitting

    Expected duration: ~5 seconds
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup and load data
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

    # Fit PCA
    pca = kpms.fit_pca(**data, **config())

    # Verify PCA
    pca_path = Path(project_dir) / "pca.p"
    assert pca_path.exists(), "PCA model not saved"

    # Test variance explained
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    assert latent_dim >= 3, f"Expected at least 3 PCs, got {latent_dim}"
    assert latent_dim <= 10, f"Too many PCs required: {latent_dim}"
