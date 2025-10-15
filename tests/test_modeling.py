"""
Test suite for keypoint-MoSeq modeling functionality

Tests model initialization, fitting, and checkpoint management.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.mark.medium
@pytest.mark.notebook
def test_model_initialization(prepared_model):
    """Test model initialization with hyperparameters

    Expected duration: <5 seconds (uses prepared_model fixture)
    """
    # Get prepared model from fixture
    model = prepared_model["model"]

    # Verify model was initialized
    assert model is not None, "Model initialization returned None"

    # Verify model structure (model is a dict, not an object)
    assert "states" in model, "Model missing states key"
    assert "params" in model, "Model missing params key"
    assert "hypparams" in model, "Model missing hypparams key"


@pytest.mark.integration
@pytest.mark.notebook
def test_model_fitting_sequence(prepared_model, reduced_iterations, kpms):
    """Test sequential model fitting: AR-HMM → full model

    Expected duration: ~10 minutes (uses prepared_model fixture)
    """
    # Get prepared model from fixture
    model = prepared_model["model"]
    data = prepared_model["data"]
    metadata = prepared_model["metadata"]
    project_dir = prepared_model["project_dir"]

    # Test AR-HMM fitting
    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        ar_only=True,
        num_iters=reduced_iterations["ar_hmm_iters"],
    )
    assert model is not None, "AR-HMM fitting failed"
    assert model_name is not None, "Model name is None"

    # Test full model fitting
    model_fitted, _ = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        ar_only=False,
        num_iters=reduced_iterations["full_model_iters"],
    )
    assert model_fitted is not None, "Full model fitting failed"


@pytest.mark.medium
@pytest.mark.notebook
def test_model_saving_and_loading(prepared_model, kpms):
    """Test model checkpoint saving and loading

    Expected duration: ~2 minutes (uses prepared_model fixture)
    """
    # Get prepared model from fixture
    model = prepared_model["model"]
    data = prepared_model["data"]
    metadata = prepared_model["metadata"]
    project_dir = prepared_model["project_dir"]

    # Quick fit - fit_model automatically saves checkpoint
    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        ar_only=True,
        num_iters=5,  # Very short for speed
    )

    assert model_name is not None, "Model name is None"

    # Check checkpoint file was created by fit_model
    checkpoint_path = Path(project_dir) / model_name / "checkpoint.h5"
    assert checkpoint_path.exists(), "Checkpoint not saved"

    # Verify checkpoint structure
    with h5py.File(checkpoint_path, "r") as f:
        assert "model_snapshots" in f, "Checkpoint missing model_snapshots group"
        assert "data" in f, "Checkpoint missing data group"
        assert "metadata" in f, "Checkpoint missing metadata group"

    # Test reindexing
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Checkpoint should still exist after reindexing
    assert checkpoint_path.exists(), "Checkpoint removed after reindexing"


@pytest.mark.quick
@pytest.mark.notebook
def test_hyperparameter_estimation(
    temp_project_dir, dlc_config, dlc_videos_dir, kpms, update_kwargs
):
    """Test hyperparameter estimation (sigmasq_loc)

    Expected duration: < 5 seconds
    """
    project_dir = temp_project_dir

    # Setup - use update_kwargs fixture for standard config
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )

    # Use different anterior/posterior for this test (testing edge case)
    kpms.update_config(
        project_dir,
        use_bodyparts=update_kwargs["use_bodyparts"],
        anterior_bodyparts=["head", "nose", "right ear", "left ear"],
        posterior_bodyparts=["spine4", "spine3", "spine2", "spine1"],
    )

    # Prepare data
    coordinates, confidences, _ = kpms.load_keypoints(
        dlc_videos_dir, "deeplabcut"
    )
    config = kpms.load_config(project_dir)
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # Fit PCA
    pca = kpms.fit_pca(**data, **config)

    # Estimate sigmasq_loc hyperparameter (this is what keypoint_moseq provides)
    sigmasq_loc = kpms.estimate_sigmasq_loc(
        data["Y"], data["mask"], filter_size=config["fps"]
    )

    # Verify estimate is reasonable
    assert isinstance(
        sigmasq_loc, (int, float, np.number)
    ), "sigmasq_loc should be numeric"
    assert sigmasq_loc > 0, "sigmasq_loc should be positive"
    assert sigmasq_loc < 100, "sigmasq_loc should be reasonable (< 100)"


@pytest.mark.quick
def test_config_update(temp_project_dir, dlc_config, kpms, update_kwargs):
    """Test configuration update and persistence

    Expected duration: < 1 second
    """
    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )

    # Update config with required bodyparts first (using standard config)
    kpms.update_config(
        project_dir,
        use_bodyparts=update_kwargs["use_bodyparts"],
        anterior_bodyparts=["head", "nose", "right ear", "left ear"],
        posterior_bodyparts=["spine4", "spine3", "spine2", "spine1"],
    )

    # Update config with a real parameter (latent_dim)
    test_value = 4
    kpms.update_config(project_dir, latent_dim=test_value)

    # Verify update persisted
    config = kpms.load_config(project_dir)
    assert "latent_dim" in config["ar_hypparams"], "Config update not persisted"
    assert (
        config["ar_hypparams"]["latent_dim"] == test_value
    ), "Config value mismatch"
