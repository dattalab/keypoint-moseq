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
def test_model_initialization(temp_project_dir, dlc_config):
    """Test model initialization with hyperparameters

    Expected duration: ~30 seconds
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    config = lambda: kpms.load_config(project_dir)

    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ]
    })

    # Load and format data
    coordinates, confidences, _ = kpms.load_keypoints(project_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    # Fit PCA
    pca = kpms.fit_pca(**data, **config())
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    config.update({'latent_dim': int(latent_dim)})

    # Estimate hyperparameters
    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    assert 'kappa' in hypparams, "Missing kappa hyperparameter"
    assert 'gamma' in hypparams, "Missing gamma hyperparameter"

    config.update(hypparams)

    # Initialize model
    model = kpms.init_model(pca=pca, **data, **config())
    assert model is not None, "Model initialization returned None"

    # Verify model structure
    assert hasattr(model, 'states'), "Model missing states attribute"


@pytest.mark.integration
@pytest.mark.notebook
def test_ar_hmm_fitting(temp_project_dir, dlc_config, reduced_iterations):
    """Test AR-HMM fitting with reduced iterations

    Expected duration: ~2 minutes
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup and prepare data
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

    # Fit PCA and initialize model
    pca = kpms.fit_pca(**data, **config())
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    config.update({'latent_dim': int(latent_dim)})

    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    config.update(hypparams)

    model = kpms.init_model(pca=pca, **data, **config())

    # Fit AR-HMM only
    model_fitted = kpms.fit_model(
        model, pca=pca, **data, **config(),
        ar_only=True,
        num_iters=reduced_iterations['ar_hmm_iters']
    )

    assert model_fitted is not None, "AR-HMM fitting returned None"


@pytest.mark.integration
@pytest.mark.notebook
def test_full_model_fitting(temp_project_dir, dlc_config, reduced_iterations):
    """Test full model fitting with reduced iterations

    Expected duration: ~10 minutes
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)
    config = lambda: kpms.load_config(project_dir)

    config.update({
        'use_bodyparts': [
            'spine4', 'spine3', 'spine2', 'spine1',
            'head', 'nose', 'right ear', 'left ear'
        ]
    })

    # Prepare data
    coordinates, confidences, _ = kpms.load_keypoints(project_dir, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    # Fit PCA
    pca = kpms.fit_pca(**data, **config())
    latent_dim = kpms.find_pcs_to_explain_variance(pca, 0.9)
    config.update({'latent_dim': int(latent_dim)})

    # Initialize and fit
    hypparams = kpms.estimate_hypparams(pca=pca, **data, **config())
    config.update(hypparams)

    model = kpms.init_model(pca=pca, **data, **config())

    # AR-HMM
    model = kpms.fit_model(
        model, pca=pca, **data, **config(),
        ar_only=True,
        num_iters=reduced_iterations['ar_hmm_iters']
    )

    # Full model
    model_fitted = kpms.fit_model(
        model, pca=pca, **data, **config(),
        num_iters=reduced_iterations['full_model_iters']
    )

    assert model_fitted is not None, "Full model fitting returned None"


@pytest.mark.medium
@pytest.mark.notebook
def test_model_saving_and_loading(temp_project_dir, dlc_config, reduced_iterations):
    """Test model checkpoint saving and loading

    Expected duration: ~15 minutes
    """
    import keypoint_moseq as kpms

    project_dir = temp_project_dir

    # Setup and fit model (abbreviated)
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

    # Quick fit
    model = kpms.fit_model(
        model, pca=pca, **data, **config(),
        ar_only=True,
        num_iters=5  # Very short for speed
    )

    # Save model
    model_name = kpms.save_model(
        model, project_dir, metadata=metadata,
        pca=pca, config=config()
    )

    assert model_name is not None, "Model name is None"

    # Check files exist
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
