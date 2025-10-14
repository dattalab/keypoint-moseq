"""
Pytest configuration and shared fixtures for keypoint-moseq tests
"""
import os
import pytest
import tempfile
import shutil
import gdown
from pathlib import Path


def pytest_configure(config):
    """Configure pytest environment - set matplotlib to non-interactive backend"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for tests


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


@pytest.fixture
def dlc_example_project():
    """Path to the DLC example project

    This fixture returns the path to the DLC example data.
    The data is NEVER deleted during teardown - it's preserved as input data.
    """
    repo_root = Path(__file__).parent.parent
    dlc_path = repo_root / "docs" / "source" / "dlc_example_project"

    if not dlc_path.exists():
        pytest.skip("DLC example project not found at {dlc_path}")

    # Input data is never cleaned up - it's part of the repository
    return str(dlc_path)


@pytest.fixture
def dlc_config(dlc_example_project):
    """Path to DLC config file"""
    config_path = Path(dlc_example_project) / "config.yaml"

    if not config_path.exists():
        pytest.skip("DLC config file not found")

    return str(config_path)


@pytest.fixture
def dlc_videos_dir(dlc_example_project):
    """Path to DLC videos directory"""
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
            print(f"File exists but use_cache=False, re-downloading: {output_path}")

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

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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
    return None


@pytest.fixture
def reduced_iterations():
    """Configuration for reduced iteration counts for faster testing

    Returns dict with recommended iteration counts for CI/CD
    """
    return {
        "ar_hmm_iters": 10,      # Reduced from 50
        "full_model_iters": 20,  # Reduced from 500
        "pca_variance": 0.90,    # 90% variance explained
        "timeout_minutes": 30,   # Max test duration
    }
