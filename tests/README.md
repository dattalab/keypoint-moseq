# Keypoint-MoSeq Test Suite

This directory contains pytest-compatible tests for the keypoint-MoSeq package, converted from the official Jupyter notebooks.

## Structure

### Test Files

- `test_colab_workflow.py` - Complete workflow tests from colab notebook
- `test_modeling.py` - Model fitting and checkpoint management tests
- `test_analysis.py` - Result extraction and visualization tests
- `conftest.py` - Shared pytest fixtures and configuration
- `__init__.py` - Package initialization

### Original Notebooks (for reference)

- `notebook_colab.py` - Converted from `docs/keypoint_moseq_colab.ipynb`
- `notebook_modeling.py` - Converted from `docs/source/modeling.ipynb`
- `notebook_analysis.py` - Converted from `docs/source/analysis.ipynb`

Conversion command used:

```bash
jupytext --to py:percent <notebook>.ipynb -o tests/notebook_<name>.py
```

## Prerequisites

### Installation

Install keypoint-moseq with test dependencies:

```bash
pip install -e ".[test]"
```

This installs:

- pytest and plugins (pytest-cov, pytest-timeout, pytest-xdist)
- jupytext for notebook conversion
- h5py for HDF5 validation
- gdown for downloading test data from Google Drive

### Test Data

Tests use the DLC example project included in the repository:

- Location: `docs/source/dlc_example_project/`
- Contains: 10 minimal DLC tracking files
- Size: ~small (suitable for CI/CD)

**Important**: Input data is never deleted during test teardown.

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_colab_workflow.py

# Run specific test function
pytest tests/test_colab_workflow.py::test_project_setup
```

### By Test Category

Tests are marked by duration and type:

```bash
# Quick tests only (< 1 minute)
pytest tests/ -m quick

# Medium tests (1-5 minutes)
pytest tests/ -m medium

# Integration tests (5-15 minutes with reduced iterations)
pytest tests/ -m integration

# Notebook-derived tests
pytest tests/ -m notebook

# Exclude slow tests (for CI/CD)
pytest tests/ -m "not slow"
```

### Parallel Execution

Run tests in parallel with pytest-xdist:

```bash
# Use all available CPU cores
pytest tests/ -n auto

# Use specific number of workers
pytest tests/ -n 4
```

### Preserve Test Outputs

By default, test outputs are cleaned up. To preserve them:

```bash
# Preserve outputs in /tmp/kpms_test_<test_name>/
pytest tests/ --no-teardown

# Specify custom output directory
pytest tests/ --test-data-dir=/path/to/output
```

Example output locations:

- `/tmp/kpms_test_test_complete_workflow/`
- Contains: model checkpoints, results, plots, videos

### Timeout Configuration

Tests have a 30-minute default timeout configured in `pyproject.toml`.

Override for specific tests:

```bash
# Set custom timeout (in seconds)
pytest tests/ --timeout=3600

# Disable timeout
pytest tests/ --timeout=0
```

## Test Categories

### Quick Tests (< 1 minute)

- `test_project_setup` - Project initialization
- `test_load_keypoints` - Data loading
- `test_hyperparameter_estimation` - Hyperparam computation
- `test_config_update` - Configuration management
- `test_syllable_statistics` - Statistics computation

### Medium Tests (1-5 minutes)

- `test_format_and_outlier_detection` - Data QA
- `test_pca_fitting` - PCA model fitting
- `test_model_initialization` - Model setup
- `test_ar_hmm_fitting` - AR-HMM fitting (reduced iterations)

### Integration Tests (5-15 minutes)

- `test_complete_workflow` - End-to-end pipeline
- `test_full_model_fitting` - Complete model fitting
- `test_model_saving_and_loading` - Checkpoint management
- `test_result_extraction` - Result generation
- `test_csv_export` - CSV output
- `test_trajectory_plots` - Visualization
- `test_similarity_dendrogram` - Dendrogram generation

### Slow Tests (> 15 minutes)

- `test_grid_movies` - Video rendering (~20 minutes)

Run without slow tests:

```bash
pytest tests/ -m "not slow"
```

## Test Fixtures

Key fixtures available in `conftest.py`:

### Path Fixtures

- `temp_project_dir` - Temporary project directory (cleaned up unless --no-teardown)
- `dlc_example_project` - Path to DLC example data (never cleaned up)
- `dlc_config` - Path to DLC config.yaml
- `dlc_videos_dir` - Path to DLC videos directory
- `notebook_output_dir` - Directory for notebook outputs
- `test_data_cache` - Cache directory for downloaded data

### Configuration Fixtures

- `reduced_iterations` - Reduced iteration counts for fast testing:
  - `ar_hmm_iters`: 10 (vs 50 default)
  - `full_model_iters`: 20 (vs 500 default)
  - `pca_variance`: 0.90 (90% variance explained)
  - `timeout_minutes`: 30

### Utility Functions

- `download_google_drive_file()` - Download from Google Drive (skips if exists)
- `unzip_file()` - Extract zip archives

## Expected Test Durations

Based on actual execution with minimal DLC dataset:

| Test Suite | Duration | Notes |
|------------|----------|-------|
| Quick tests | < 2 min | All quick tests combined |
| Medium tests | 5-10 min | Includes PCA, outlier detection |
| Integration tests | 60-90 min | All integration tests |
| Complete workflow | ~15 min | Single full pipeline test |
| All tests (no slow) | ~90 min | Suitable for CI/CD |
| All tests (with slow) | ~110 min | Includes video rendering |

## CI/CD Recommendations

### Minimal Test Suite (Fast)

```bash
# Run only quick tests (~2 minutes)
pytest tests/ -m quick -n auto
```

### Standard Test Suite (Balanced)

```bash
# Run quick + medium tests (~15 minutes)
pytest tests/ -m "quick or medium" -n auto
```

### Full Test Suite (Comprehensive)

```bash
# Run all except slow tests (~90 minutes)
pytest tests/ -m "not slow" -n auto
```

### Nightly/Weekly Tests

```bash
# Run everything including slow tests (~110 minutes)
pytest tests/ -n auto
```

## Troubleshooting

### Test Failures

**Import errors**: Ensure package installed with test dependencies

```bash
pip install -e ".[test]"
```

**DLC data not found**: Verify DLC example project exists

```bash
ls docs/source/dlc_example_project/
```

**Timeout errors**: Increase timeout or run on faster hardware

```bash
pytest tests/ --timeout=3600
```

**JAX/GPU warnings**: Expected on CPU-only systems (tests run fine)

### Common Warnings

- `FigureCanvasAgg is non-interactive` - Expected for headless execution
- `os.fork() was called... may lead to deadlock` - JAX warning during video generation (harmless)
- `An NVIDIA GPU may be present... Falling back to cpu` - Expected without CUDA

### Preserving Outputs for Debugging

```bash
# Keep outputs and show print statements
pytest tests/test_colab_workflow.py::test_complete_workflow -s --no-teardown

# Check preserved outputs
ls /tmp/kpms_test_test_complete_workflow/
```

## Code Coverage

### Generate Coverage Report

Run tests with coverage measurement:

```bash
# Generate HTML and terminal coverage report
pytest tests/ --cov=keypoint_moseq --cov-report=html --cov-report=term -m "not slow"

# View HTML report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Commands

```bash
# Coverage with missing line numbers
pytest tests/ --cov=keypoint_moseq --cov-report=term-missing

# Coverage for specific module
pytest tests/ --cov=keypoint_moseq.analysis --cov-report=term

# Coverage with XML output (for CI/CD)
pytest tests/ --cov=keypoint_moseq --cov-report=xml --cov-report=term
```

## Test Development

### Adding New Tests

1. Create test file: `tests/test_<feature>.py`
2. Import required modules
3. Add pytest markers: `@pytest.mark.quick`, `@pytest.mark.integration`, etc.
4. Use fixtures: `def test_something(temp_project_dir, dlc_config):`
5. Add assertions: `assert result is not None, "Result should not be None"`
6. Document expected duration in docstring

### Test Template

```python
import pytest
from pathlib import Path

@pytest.mark.quick
@pytest.mark.notebook
def test_feature_name(temp_project_dir, dlc_config):
    """Test description

    Expected duration: < 1 minute
    """
    import keypoint_moseq as kpms

    # Setup
    project_dir = temp_project_dir
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config, overwrite=True)

    # Test logic
    result = kpms.some_function()

    # Assertions
    assert result is not None, "Result should not be None"
    assert Path(project_dir, "output.txt").exists(), "Output file not created"
```

## Additional Resources

For more information about keypoint-moseq:

- **Official Documentation**: <https://keypoint-moseq.readthedocs.io/>
- **GitHub Repository**: <https://github.com/dattalab/keypoint-moseq>
- **Paper**: Nature Methods (2024) - <https://www.nature.com/articles/s41592-024-02318-2>

For test development questions, refer to:

- Pytest documentation: <https://docs.pytest.org/>
- This README for test structure and conventions
- Example test functions in existing test files
