"""
Test suite for the keypoint-MoSeq colab workflow

This test suite validates the complete workflow from the colab notebook,
adapted for pytest with appropriate fixtures and assertions.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.notebook
def test_complete_workflow(
    temp_project_dir, dlc_config, dlc_videos_dir, reduced_iterations, kpms
):
    """Test the complete keypoint-MoSeq workflow end-to-end

    This test runs the full pipeline with reduced iterations suitable for CI/CD.
    Expected duration: ~15 minutes
    """
    from tests.conftest import compute_latent_dim, load_path_from_model

    project_dir = temp_project_dir

    # Step 1: Setup project
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )
    assert Path(project_dir, "config.yml").exists(), "Config file not created"

    # Step 2: Update config
    kpms.update_config(
        project_dir,
        use_bodyparts=[
            "spine4",
            "spine3",
            "spine2",
            "spine1",
            "head",
            "nose",
            "right ear",
            "left ear",
        ],
        anterior_bodyparts=["head", "nose", "right ear", "left ear"],
        posterior_bodyparts=["spine4", "spine3", "spine2", "spine1"],
        seg_length=5,
    )
    config = kpms.load_config(project_dir)

    # Step 3: Load keypoints
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        dlc_videos_dir, "deeplabcut"
    )
    assert len(coordinates) > 0, "No keypoints loaded"
    assert len(bodyparts) == 9, f"Expected 9 bodyparts, got {len(bodyparts)}"

    # Step 4: Outlier removal (before formatting)
    kpms.update_config(project_dir, outlier_scale_factor=6.0)
    coordinates, confidences = kpms.outlier_removal(
        coordinates, confidences, project_dir, overwrite=True, **config
    )
    qa_dir = Path(project_dir) / "QA" / "plots"
    assert qa_dir.exists(), "QA plots directory not created"

    # Step 5: Format data after outlier removal
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    assert "Y" in data, "Formatted data missing Y"
    assert "conf" in data, "Formatted data missing conf"

    # Step 6: Skip calibration (not needed for minimal dataset)
    # Manual calibration widget would go here in interactive mode

    # Step 7: Fit PCA
    pca = kpms.fit_pca(**data, **config)
    kpms.save_pca(pca, project_dir)
    pca_path = Path(project_dir) / "pca.p"
    assert pca_path.exists(), "PCA model not saved"

    # Step 8: Update latent dimensions
    latent_dim = compute_latent_dim(pca, variance_threshold=0.9)
    assert latent_dim >= 3, f"Expected at least 3 PCs, got {latent_dim}"
    kpms.update_config(project_dir, latent_dim=int(latent_dim))
    config = kpms.load_config(project_dir)

    # Step 9: Estimate hyperparameters
    sigmasq_loc = kpms.estimate_sigmasq_loc(
        data["Y"], data["mask"], filter_size=config["fps"]
    )
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)
    config = kpms.load_config(project_dir)

    # Step 10: Initialize model
    model = kpms.init_model(data, pca=pca, **config)
    assert model is not None, "Model initialization failed"

    # Step 11: Fit AR-HMM with reduced iterations
    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        ar_only=True,
        num_iters=reduced_iterations["ar_hmm_iters"],
    )

    # Step 12: Fit full model with reduced iterations
    model, _ = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        ar_only=False,
        num_iters=reduced_iterations["full_model_iters"],
    )

    # Step 13: Verify checkpoint was saved by fit_model
    checkpoint_path = load_path_from_model(project_dir, model_name, "checkpoint.h5")
    assert checkpoint_path.exists(), "Checkpoint file not created"

    # Step 14: Reindex syllables
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Step 15: Extract results
    results = kpms.extract_results(
        model, metadata, project_dir, model_name, config
    )
    example_model = results[metadata[0][0]]
    assert "syllable" in example_model, "Results missing syllable labels"

    results_h5_path = Path(project_dir) / model_name / "results.h5"
    assert results_h5_path.exists(), "Results HDF5 not created"

    # Validate results structure
    with h5py.File(results_h5_path, "r") as f:
        recording_keys = list(f.keys())
        assert len(recording_keys) > 0, "No recordings in results"

        first_recording = f[recording_keys[0]]
        # Verify required datasets are present
        required_datasets = {"syllable", "centroid", "heading", "latent_state"}
        actual_datasets = set(first_recording.keys())
        missing = required_datasets - actual_datasets
        assert not missing, f"Results missing datasets: {missing}"

    # Step 16: Save as CSV
    results_dir = load_path_from_model(project_dir, model_name, "results")
    csv_files_before = list(results_dir.glob("*.csv")) if results_dir.exists() else []

    kpms.save_results_as_csv(results, project_dir, model_name)
    assert results_dir.exists(), "Results CSV directory not created"

    csv_files_after = list(results_dir.glob("*.csv"))
    assert len(csv_files_after) > len(csv_files_before), "No new CSV files created"

    # Step 17: Generate visualizations
    # Add video_dir to config for visualization functions
    config["video_dir"] = dlc_videos_dir

    # Generate trajectory plots
    kpms.generate_trajectory_plots(
        coordinates=coordinates,
        results=results,
        project_dir=project_dir,
        model_name=model_name,
        **config,
    )
    trajectory_dir = load_path_from_model(project_dir, model_name, "trajectory_plots")
    assert trajectory_dir.exists(), "Trajectory plots directory not created"

    num_syllables = len(set(example_model["syllable"]))
    assert num_syllables > 0, "No syllables identified"

    # Check for trajectory plots
    pdf_plots = [f for f in trajectory_dir.glob("*.pdf")]
    assert len(pdf_plots) > 0, "No trajectory PDFs created"

    # Generate grid movies
    kpms.generate_grid_movies(
        coordinates=coordinates,
        results=results,
        project_dir=project_dir,
        model_name=model_name,
        frame_path=None,
        **config,
    )
    grid_movies_dir = load_path_from_model(project_dir, model_name, "grid_movies")
    assert grid_movies_dir.exists(), "Grid movies directory not created"

    mp4_files = [f for f in grid_movies_dir.glob("*.mp4")]
    assert len(mp4_files) > 0, "No grid movies created"

    # Generate similarity dendrogram
    kpms.plot_similarity_dendrogram(
        coordinates=coordinates,
        results=results,
        project_dir=project_dir,
        model_name=model_name,
        **config,
    )
    dendrogram_pdf = load_path_from_model(
        project_dir, model_name, "similarity_dendrogram.pdf"
    )
    assert dendrogram_pdf.exists(), "Similarity dendrogram not created"


@pytest.mark.quick
@pytest.mark.notebook
def test_project_setup(temp_project_dir, dlc_config, kpms):
    """Test project setup and configuration

    Expected duration: < 1 second
    """
    project_dir = temp_project_dir

    # Test setup
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )

    # Verify files created
    config_path = Path(project_dir, "config.yml")
    assert config_path.exists(), "Config file not created"

    # Update config with valid bodyparts before loading
    # (setup_project creates placeholders that need to be updated)
    kpms.update_config(
        project_dir,
        use_bodyparts=[
            "spine4",
            "spine3",
            "spine2",
            "spine1",
            "head",
            "nose",
            "right ear",
            "left ear",
        ],
        anterior_bodyparts=["head", "nose", "right ear", "left ear"],
        posterior_bodyparts=["spine4", "spine3", "spine2", "spine1"],
    )

    # Test config loading after update
    config = kpms.load_config(project_dir)
    expected_keys = {"bodyparts", "fps", "use_bodyparts"}
    assert expected_keys.issubset(config.keys()), f"Config missing keys: {expected_keys - config.keys()}"
    assert len(config["use_bodyparts"]) == 8, "Wrong number of use_bodyparts"


@pytest.mark.quick
@pytest.mark.notebook
def test_load_keypoints(temp_project_dir, dlc_config, dlc_videos_dir, kpms):
    """Test keypoint loading from DLC data

    Expected duration: < 1 second
    """
    project_dir = temp_project_dir
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )

    # Load keypoints from DLC videos directory (not project_dir)
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        dlc_videos_dir, "deeplabcut"
    )

    # Verify data structure
    assert len(coordinates) > 0, "No coordinates loaded"
    assert len(confidences) > 0, "No confidences loaded"
    assert len(bodyparts) == 9, f"Expected 9 bodyparts, got {len(bodyparts)}"

    # Check data types
    first_recording = next(iter(coordinates.keys()))
    assert isinstance(
        coordinates[first_recording], np.ndarray
    ), "Coordinates not numpy array"
    assert coordinates[first_recording].ndim == 3, "Coordinates wrong shape"


@pytest.mark.medium
@pytest.mark.notebook
def test_format_and_outlier_detection(
    temp_project_dir, dlc_config, dlc_videos_dir, kpms, update_kwargs
):
    """Test data formatting and outlier detection

    Expected duration: ~1 minute
    """
    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )

    # Update config using fixture
    kpms.update_config(project_dir, **update_kwargs)
    config = kpms.load_config(project_dir)

    # Load keypoints
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        dlc_videos_dir, "deeplabcut"
    )

    # Format data
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    assert "Y" in data, "Formatted data missing Y"

    # Test outlier removal (matches notebook API)
    kpms.update_config(project_dir, outlier_scale_factor=6.0)
    coordinates_clean, confidences_clean = kpms.outlier_removal(
        coordinates, confidences, project_dir, overwrite=True, **config
    )

    # Verify outputs
    assert len(coordinates_clean) > 0, "No coordinates after outlier removal"
    assert len(confidences_clean) > 0, "No confidences after outlier removal"

    qa_dir = Path(project_dir) / "QA" / "plots"
    assert qa_dir.exists(), "QA directory not created"


@pytest.mark.medium
@pytest.mark.notebook
def test_pca_fitting(temp_project_dir, dlc_config, dlc_videos_dir, kpms, update_kwargs):
    """Test PCA model fitting

    Expected duration: ~5 seconds
    """
    from tests.conftest import compute_latent_dim

    project_dir = temp_project_dir

    # Setup and load data
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )

    # Update config using fixture
    kpms.update_config(project_dir, **update_kwargs)
    config = kpms.load_config(project_dir)

    coordinates, confidences, _ = kpms.load_keypoints(
        dlc_videos_dir, "deeplabcut"
    )
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # Fit PCA
    pca = kpms.fit_pca(**data, **config)
    kpms.save_pca(pca, project_dir)

    # Verify PCA
    pca_path = Path(project_dir) / "pca.p"
    assert pca_path.exists(), "PCA model not saved"

    # Test variance explained using helper
    latent_dim = compute_latent_dim(pca, variance_threshold=0.9)
    assert latent_dim >= 3, f"Expected at least 3 PCs, got {latent_dim}"
    assert latent_dim <= 10, f"Too many PCs required: {latent_dim}"
