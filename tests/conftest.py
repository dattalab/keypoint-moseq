"""
Pytest configuration and shared fixtures for keypoint-moseq tests
"""

import shutil
import tempfile
import warnings
from pathlib import Path

import gdown
import pytest
from matplotlib import MatplotlibDeprecationWarning

# - Warnings Configuration - #
# Ignore warnings from viz.py: "FigureCanvasAgg is non-interactive"
warnings.filterwarnings("ignore", category=UserWarning, module="keypoint_moseq")
# Bohek from Numpy 1.24: DeprecationWarning for np.bool8
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="bokeh")
# From JAX: 0 should be passed as minlength instead of None
# From JAX: shape requires ndarray or scalar, got None
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax_moseq")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow_probability")
# From matplotlib, 'mode' is deprecated, removed in Pillow 13 (2026-10-15)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
# From matplotlib, tostring_rgb is deprecated
warnings.simplefilter("ignore", MatplotlibDeprecationWarning)


def pytest_configure(config):
    """Configure pytest environment - set matplotlib to non-interactive backend"""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for tests


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--no-teardown",
        action="store_true",
        default=False,
        help="Preserve test outputs (don't cleanup temporary directories)",
    )
    parser.addoption(
        "--test-data-dir",
        action="store",
        default=None,
        help="Directory for test data (if not specified, uses temp dir)",
    )


@pytest.fixture
def no_teardown(request):
    """Check if --no-teardown flag is set"""
    return request.config.getoption("--no-teardown")


@pytest.fixture
def temp_project_dir(request, no_teardown):
    """Create a temporary project directory for testing

    If --no-teardown is specified, the directory is preserved after tests.
    """
    test_data_dir = request.config.getoption("--test-data-dir")

    if test_data_dir:
        # Use specified directory
        tmpdir = Path(test_data_dir) / f"test_{request.node.name}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        yield str(tmpdir)
        if not no_teardown:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        # Use system temp directory
        if no_teardown:
            # Create in /tmp with predictable name
            tmpdir = Path("/tmp") / f"kpms_test_{request.node.name}"
            tmpdir.mkdir(parents=True, exist_ok=True)
            yield str(tmpdir)
            print(f"\n[NO TEARDOWN] Test outputs preserved at: {tmpdir}")
        else:
            # Standard temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                yield tmpdir


@pytest.fixture(scope="session")
def dlc_example_project():
    """Path to the DLC example project

    This fixture returns the path to the DLC example data.
    The data is NEVER deleted during teardown - it's preserved as input data.
    Session-scoped since it's read-only data.
    """
    repo_root = Path(__file__).parent.parent
    dlc_path = repo_root / "docs" / "source" / "dlc_example_project"

    if not dlc_path.exists():
        pytest.skip("DLC example project not found at {dlc_path}")

    # Input data is never cleaned up - it's part of the repository
    return str(dlc_path)


@pytest.fixture(scope="session")
def dlc_config(dlc_example_project):
    """Path to DLC config file

    Session-scoped since it's read-only data.
    """
    config_path = Path(dlc_example_project) / "config.yaml"

    if not config_path.exists():
        pytest.skip("DLC config file not found")

    return str(config_path)


@pytest.fixture(scope="session")
def dlc_videos_dir(dlc_example_project):
    """Path to DLC videos directory

    Session-scoped since it's read-only data.
    """
    videos_path = Path(dlc_example_project) / "videos"

    if not videos_path.exists():
        pytest.skip("DLC videos directory not found")

    return str(videos_path)


@pytest.fixture(scope="session")
def notebook_output_dir():
    """Directory for notebook-generated outputs during testing"""
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / "tests" / "notebook_outputs"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)


@pytest.fixture(scope="session")
def test_data_cache():
    """Cache directory for downloaded test data"""
    cache_dir = Path.home() / ".cache" / "keypoint_moseq_tests"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_google_drive_file(file_id, output_path, use_cache=True):
    """Download a file from Google Drive

    Always checks if file exists before downloading. Downloaded data is
    preserved and never deleted during teardown.

    Args:
        file_id: Google Drive file ID
        output_path: Path where file should be saved
        use_cache: If True, skip download if file exists (default: True)

    Returns:
        Path to downloaded file
    """
    output_path = Path(output_path)

    # Always check if already exists - skip download if present
    if output_path.exists():
        if use_cache:
            print(f"Using cached file: {output_path}")
            return output_path
        else:
            print(
                f"File exists but use_cache=False, re-downloading: {output_path}"
            )

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download from Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive: {file_id}")
    gdown.download(url, str(output_path), quiet=False)

    return output_path


def unzip_file(zip_path, extract_to):
    """Extract a zip file

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to

    Returns:
        Path to extracted directory
    """
    import zipfile

    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to


@pytest.fixture(scope="session")
def dlc_test_data(test_data_cache):
    """Download and cache DLC test data from Google Drive

    This fixture downloads the minimal DLC dataset used in the colab notebook.
    The file ID is extracted from the colab notebook's google drive link.

    Note: Currently uses the local dlc_example_project. If external data
    is needed, implement download logic here.
    """
    # For now, return None - tests should use dlc_example_project fixture
    # This can be extended if external test data needs to be downloaded
    # TODO: Implement download logic if no example project is available in docs/source or tests/
    return None


@pytest.fixture(scope="session")
def reduced_iterations():
    """Configuration for reduced iteration counts for faster testing

    Returns dict with recommended iteration counts for CI/CD

    Note: pca_variance set to 0.80 (was 0.90) for speed. This reduces
    the number of PCA components, making model fitting ~30-40% faster
    while still capturing most variance. For production models, use 0.90.

    Session-scoped since it's just configuration data.
    """
    return {
        "ar_hmm_iters": 10,  # Reduced from 50
        "full_model_iters": 20,  # Reduced from 500
        "pca_variance": 0.80,  # 80% variance (was 0.90) - faster for tests
        "timeout_minutes": 30,  # Max test duration
    }


@pytest.fixture(scope="session")
def kpms():
    """Session-scoped fixture for keypoint_moseq package import

    Eliminates redundant imports in every test function.
    """
    import keypoint_moseq as kpms

    return kpms


@pytest.fixture(scope="session")
def update_kwargs():
    """Standard config update kwargs used across multiple tests

    Returns dict with common bodypart configurations.
    Use with: kpms.update_config(project_dir, **update_kwargs)

    Session-scoped since it's just configuration data.
    """
    return {
        "use_bodyparts": [
            "spine4",
            "spine3",
            "spine2",
            "spine1",
            "head",
            "nose",
            "right ear",
            "left ear",
        ],
        "anterior_bodyparts": ["nose"],
        "posterior_bodyparts": ["spine4"],
    }


@pytest.fixture(scope="module")
def module_project_dir(request):
    """Create a module-scoped temporary project directory

    This is used by fitted_model fixture to create a single project
    directory that's shared across all tests in the module.
    """
    # Use system temp directory with module name
    import tempfile

    tmpdir = Path(
        tempfile.mkdtemp(prefix=f"kpms_test_module_{request.module.__name__}_")
    )
    yield str(tmpdir)
    # Cleanup after all tests in module complete
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def prepared_model(
    module_project_dir,
    dlc_config,
    dlc_videos_dir,
    reduced_iterations,
    kpms,
    update_kwargs,
):
    """Module-scoped fixture providing initialized model ready for fitting

    Runs setup → load → format → PCA → hyperparams → init once per module.
    Tests can then fit the model with different parameters.

    Speed impact: Eliminates 1.5 min of duplicated setup per test.
    """
    project_dir = module_project_dir

    # Step 1: Setup project
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )

    # Step 2: Update config with standard bodyparts
    kpms.update_config(project_dir, **update_kwargs)
    config = kpms.load_config(project_dir)

    # Step 3: Load keypoints
    coordinates, confidences, _ = kpms.load_keypoints(
        dlc_videos_dir, "deeplabcut"
    )

    # Step 4: Format data
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # Step 5: Fit PCA
    pca = kpms.fit_pca(**data, **config)

    # Step 6: Compute latent dimensions
    latent_dim = compute_latent_dim(
        pca, variance_threshold=reduced_iterations["pca_variance"]
    )
    kpms.update_config(project_dir, latent_dim=int(latent_dim))
    config = kpms.load_config(project_dir)

    # Step 7: Estimate hyperparameters
    sigmasq_loc = kpms.estimate_sigmasq_loc(
        data["Y"], data["mask"], filter_size=config["fps"]
    )
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)
    config = kpms.load_config(project_dir)

    # Step 8: Initialize model (but don't fit yet)
    model = kpms.init_model(data, pca=pca, **config)

    # Return all intermediate results
    return {
        "project_dir": project_dir,
        "model": model,
        "data": data,
        "metadata": metadata,
        "pca": pca,
        "config": config,
        "coordinates": coordinates,
        "confidences": confidences,
    }


@pytest.fixture(scope="module")
def fitted_model(
    module_project_dir, dlc_config, dlc_videos_dir, reduced_iterations, kpms
):
    """Module-scoped fixture providing a fully fitted model

    This fixture runs the expensive workflow once per module:
    - Setup project
    - Load and format data
    - Fit PCA
    - Estimate hyperparameters
    - Initialize model
    - Fit AR-HMM and full model

    Returns dict with all intermediate results for reuse in tests.

    Speed impact: Reduces 10-15 min workflow to <1 min for dependent tests.
    """
    project_dir = module_project_dir

    # Step 1: Setup project
    kpms.setup_project(
        project_dir, deeplabcut_config=dlc_config, overwrite=True
    )

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
        anterior_bodyparts=["nose"],
        posterior_bodyparts=["spine4"],
    )
    config = kpms.load_config(project_dir)

    # Step 3: Load keypoints
    coordinates, confidences, _ = kpms.load_keypoints(
        dlc_videos_dir, "deeplabcut"
    )

    # Step 4: Format data
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # Step 5: Fit PCA
    pca = kpms.fit_pca(**data, **config)
    kpms.save_pca(pca, project_dir)

    # Step 6: Compute latent dimensions
    latent_dim = compute_latent_dim(
        pca, variance_threshold=reduced_iterations["pca_variance"]
    )
    kpms.update_config(project_dir, latent_dim=int(latent_dim))
    config = kpms.load_config(project_dir)

    # Step 7: Estimate hyperparameters
    sigmasq_loc = kpms.estimate_sigmasq_loc(
        data["Y"], data["mask"], filter_size=config["fps"]
    )
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)
    config = kpms.load_config(project_dir)

    # Step 8: Initialize model
    model = kpms.init_model(data, pca=pca, **config)

    # Step 9: Fit model
    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        ar_only=True,
        num_iters=reduced_iterations["ar_hmm_iters"],
    )
    model, _ = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        ar_only=False,
        num_iters=reduced_iterations["full_model_iters"],
    )

    # Return all intermediate results
    return {
        "project_dir": project_dir,
        "model": model,
        "model_name": model_name,
        "data": data,
        "metadata": metadata,
        "pca": pca,
        "config": config,
        "coordinates": coordinates,
        "confidences": confidences,
    }


# Helper functions


def compute_latent_dim(pca, variance_threshold=0.9):
    """Compute number of PCA components needed to explain variance threshold

    Args:
        pca: Fitted PCA object with explained_variance_ratio_ attribute
        variance_threshold: Target cumulative variance (default: 0.9 for 90%)

    Returns:
        int: Number of components needed
    """
    import numpy as np

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = int(np.argmax(cumsum >= variance_threshold) + 1)
    return latent_dim


def load_path_from_model(
    project_dir, model_name, filename, delete_existing=False
):
    """Construct standardized path to model output file

    Args:
        project_dir: Project directory path
        model_name: Model name (timestamp directory)
        filename: Target filename (e.g., 'checkpoint.h5', 'results.h5')

    Returns:
        Path: Absolute path to file
    """
    file_path = Path(project_dir) / model_name / filename

    if delete_existing and file_path.exists():
        file_path.unlink()

    return file_path


def assert_result_keys(results, expected_keys):
    """Assert that results dict contains all expected keys

    Args:
        results: Results dictionary to validate
        expected_keys: List or set of expected key names

    Raises:
        AssertionError: If any expected keys are missing
    """
    missing_keys = set(expected_keys) - set(results.keys())
    assert not missing_keys, f"Results missing keys: {missing_keys}"
