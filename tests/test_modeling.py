"""
Test suite for keypoint-MoSeq modeling functionality

Tests model initialization, fitting, and checkpoint management.
"""
import pytest
import numpy as np
from pathlib import Path
import h5py


@pytest.mark.medium
@pytest.mark.notebook
def test_model_initialization(temp_project_dir, dlc_config, dlc_videos_dir):
    """Test model initialization with hyperparameters

    Expected duration: ~30 seconds
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    kpms.update_config(
        project_dir,
        use_bodyparts=[
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ],
        anterior_bodyparts=['nose'],
        posterior_bodyparts=['spine4']
    )
    config = kpms.load_config(project_dir)

    # Load and format data
    coordinates, confidences, _ = kpms.load_keypoints(dlc_videos_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # Fit PCA
    pca = kpms.fit_pca(**data, **config)

    # Compute latent_dim manually
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = int(np.argmax(cumsum >= 0.9) + 1)
    kpms.update_config(project_dir, latent_dim=int(latent_dim))
    config = kpms.load_config(project_dir)

    # Estimate hyperparameters (sigmasq_loc)
    sigmasq_loc = kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config["fps"])
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)
    config = kpms.load_config(project_dir)

    # Initialize model
    model = kpms.init_model(data, pca=pca, **config)
    assert model is not None, "Model initialization returned None"

    # Verify model structure (model is a dict, not an object)
    assert 'states' in model, "Model missing states key"
    assert 'params' in model, "Model missing params key"
    assert 'hypparams' in model, "Model missing hypparams key"


@pytest.mark.integration
@pytest.mark.notebook
def test_ar_hmm_fitting(temp_project_dir, dlc_config, dlc_videos_dir, reduced_iterations):
    """Test AR-HMM fitting with reduced iterations

    Expected duration: ~2 minutes
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup and prepare data
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    kpms.update_config(
        project_dir,
        use_bodyparts=[
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ],
        anterior_bodyparts=['nose'],
        posterior_bodyparts=['spine4']
    )
    config = kpms.load_config(project_dir)

    coordinates, confidences, _ = kpms.load_keypoints(dlc_videos_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # Fit PCA and initialize model
    pca = kpms.fit_pca(**data, **config)

    # Compute latent_dim manually
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = int(np.argmax(cumsum >= 0.9) + 1)
    kpms.update_config(project_dir, latent_dim=int(latent_dim))
    config = kpms.load_config(project_dir)

    # Estimate hyperparameters
    sigmasq_loc = kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config["fps"])
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)
    config = kpms.load_config(project_dir)

    model = kpms.init_model(data, pca=pca, **config)

    # Fit AR-HMM only
    model_fitted, model_name = kpms.fit_model(model, data, metadata, project_dir, ar_only=True, num_iters=reduced_iterations['ar_hmm_iters']
    )

    assert model_fitted is not None, "AR-HMM fitting returned None"
    assert model_name is not None, "Model name is None"


@pytest.mark.integration
@pytest.mark.notebook
def test_full_model_fitting(temp_project_dir, dlc_config, dlc_videos_dir, reduced_iterations):
    """Test full model fitting with reduced iterations

    Expected duration: ~10 minutes
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    kpms.update_config(
        project_dir,
        use_bodyparts=[
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ],
        anterior_bodyparts=['nose'],
        posterior_bodyparts=['spine4']
    )
    config = kpms.load_config(project_dir)

    # Prepare data
    coordinates, confidences, _ = kpms.load_keypoints(dlc_videos_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # Fit PCA
    pca = kpms.fit_pca(**data, **config)

    # Compute latent_dim manually
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = int(np.argmax(cumsum >= 0.9) + 1)
    kpms.update_config(project_dir, latent_dim=int(latent_dim))
    config = kpms.load_config(project_dir)

    # Initialize and fit
    sigmasq_loc = kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config["fps"])
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)
    config = kpms.load_config(project_dir)

    model = kpms.init_model(data, pca=pca, **config)

    # AR-HMM
    model, model_name = kpms.fit_model(model, data, metadata, project_dir, ar_only=True, num_iters=reduced_iterations['ar_hmm_iters']
    )

    # Full model
    model_fitted, _ = kpms.fit_model(model, data, metadata, project_dir, ar_only=False, num_iters=reduced_iterations['full_model_iters']
    )

    assert model_fitted is not None, "Full model fitting returned None"


@pytest.mark.medium
@pytest.mark.notebook
def test_model_saving_and_loading(temp_project_dir, dlc_config, dlc_videos_dir, reduced_iterations):
    """Test model checkpoint saving and loading

    Expected duration: ~15 minutes
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup and fit model (abbreviated)
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    kpms.update_config(
        project_dir,
        use_bodyparts=[
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ],
        anterior_bodyparts=['nose'],
        posterior_bodyparts=['spine4']
    )
    config = kpms.load_config(project_dir)

    coordinates, confidences, _ = kpms.load_keypoints(dlc_videos_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    pca = kpms.fit_pca(**data, **config)

    # Compute latent_dim manually
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = int(np.argmax(cumsum >= 0.9) + 1)
    kpms.update_config(project_dir, latent_dim=int(latent_dim))
    config = kpms.load_config(project_dir)

    sigmasq_loc = kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config["fps"])
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)
    config = kpms.load_config(project_dir)

    model = kpms.init_model(data, pca=pca, **config)

    # Quick fit - fit_model automatically saves checkpoint
    model, model_name = kpms.fit_model(model, data, metadata, project_dir, ar_only=True, num_iters=5  # Very short for speed
    )

    assert model_name is not None, "Model name is None"

    # Check checkpoint file was created by fit_model
    checkpoint_path = Path(project_dir) / model_name / "checkpoint.h5"
    assert checkpoint_path.exists(), "Checkpoint not saved"

    # Verify checkpoint structure
    with h5py.File(checkpoint_path, 'r') as f:
        assert 'model' in f, "Checkpoint missing model group"

    # Test reindexing
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

    # Checkpoint should still exist after reindexing
    assert checkpoint_path.exists(), "Checkpoint removed after reindexing"


@pytest.mark.quick
@pytest.mark.notebook
def test_hyperparameter_estimation(temp_project_dir, dlc_config, dlc_videos_dir):
    """Test hyperparameter estimation (sigmasq_loc)

    Expected duration: < 5 seconds
    """
    import keypoint_moseq as kpms
    import numpy as np

    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    kpms.update_config(
        project_dir,
        use_bodyparts=[
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ],
        anterior_bodyparts=['head', 'nose', 'right ear', 'left ear'],
        posterior_bodyparts=['spine4', 'spine3', 'spine2', 'spine1'],
    )

    # Prepare data
    coordinates, confidences, _ = kpms.load_keypoints(dlc_videos_dir, 'deeplabcut')
    config = kpms.load_config(project_dir)
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # Fit PCA
    pca = kpms.fit_pca(**data, **config)

    # Estimate sigmasq_loc hyperparameter (this is what keypoint_moseq provides)
    sigmasq_loc = kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config["fps"])

    # Verify estimate is reasonable
    assert isinstance(sigmasq_loc, (int, float, np.number)), "sigmasq_loc should be numeric"
    assert sigmasq_loc > 0, "sigmasq_loc should be positive"
    assert sigmasq_loc < 100, "sigmasq_loc should be reasonable (< 100)"


@pytest.mark.quick
def test_config_update(temp_project_dir, dlc_config):
    """Test configuration update and persistence

    Expected duration: < 1 second
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    # Update config with required bodyparts first
    kpms.update_config(
        project_dir,
        use_bodyparts=[
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ],
        anterior_bodyparts=['head', 'nose', 'right ear', 'left ear'],
        posterior_bodyparts=['spine4', 'spine3', 'spine2', 'spine1'],
    )

    # Update config with a real parameter (latent_dim)
    test_value = 4
    kpms.update_config(project_dir, latent_dim=test_value)

    # Verify update persisted
    config = kpms.load_config(project_dir)
    assert 'latent_dim' in config['ar_hypparams'], "Config update not persisted"
    assert config['ar_hypparams']['latent_dim'] == test_value, "Config value mismatch"
