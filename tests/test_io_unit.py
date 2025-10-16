"""
Unit tests for keypoint_moseq.io module

Target coverage: 55% → 75% (from 249/455 statements to ~341/455)
Priority functions tested:
- Configuration management (_build_yaml, generate_config, check_config_validity, update_config)
- Path utilities (_get_path, _name_from_path)
- HDF5 operations (save_hdf5, load_hdf5)
- PCA persistence (save_pca, load_pca)
- Result extraction (extract_results)
"""

import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import pytest
import numpy as np
import yaml
import h5py
import joblib

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*os.fork.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*FigureCanvasAgg.*")

from keypoint_moseq.io import (
    _build_yaml,
    _get_path,
    _name_from_path,
    generate_config,
    check_config_validity,
    load_config,
    update_config,
    setup_project,
    save_pca,
    load_pca,
    save_hdf5,
    load_hdf5,
    extract_results,
    load_results,
    load_checkpoint,
    reindex_syllables_in_checkpoint,
    save_results_as_csv,
    save_keypoints,
)


@pytest.mark.quick
class TestBuildYaml:
    """Test _build_yaml helper function."""

    def test_basic_structure(self):
        """Test basic YAML structure generation."""
        sections = [
            ("TEST SECTION", {"key1": "value1", "key2": 123}),
        ]
        comments = {}

        result = _build_yaml(sections, comments)

        assert "TEST SECTION" in result
        assert "key1" in result
        assert "value1" in result
        assert "key2" in result

    def test_with_comments(self):
        """Test YAML generation with comments."""
        sections = [
            ("TEST", {"setting": "value"}),
        ]
        comments = {"setting": "This is a test comment"}

        result = _build_yaml(sections, comments)

        assert "# This is a test comment" in result
        assert "setting: value" in result

    def test_multiple_sections(self):
        """Test multiple sections."""
        sections = [
            ("SECTION1", {"a": 1}),
            ("SECTION2", {"b": 2}),
        ]
        comments = {}

        result = _build_yaml(sections, comments)

        assert "SECTION1" in result
        assert "SECTION2" in result
        assert "a: 1" in result
        assert "b: 2" in result


@pytest.mark.quick
class TestGetPath:
    """Test _get_path utility function."""

    def test_with_explicit_path(self):
        """Test when path is explicitly provided."""
        result = _get_path(
            project_dir="/proj",
            model_name="model",
            path="/explicit/path.h5",
            filename="default.h5",
        )
        assert result == "/explicit/path.h5"

    def test_without_path_constructs_from_parts(self):
        """Test path construction from project_dir and model_name."""
        result = _get_path(
            project_dir="/project",
            model_name="my_model",
            path=None,
            filename="results.h5",
        )
        assert result == "/project/my_model/results.h5"

    def test_missing_params_raises_error(self):
        """Test error when required params missing."""
        with pytest.raises(AssertionError, match="required"):
            _get_path(
                project_dir=None,
                model_name="model",
                path=None,
                filename="file.h5",
            )


@pytest.mark.quick
class TestNameFromPath:
    """Test _name_from_path utility function."""

    def test_basename_only(self):
        """Test extracting just the basename."""
        result = _name_from_path(
            "/path/to/file.csv",
            path_in_name=False,
            path_sep="-",
            remove_extension=True,
        )
        assert result == "file"

    def test_full_path_with_separator(self):
        """Test full path with custom separator."""
        result = _name_from_path(
            "/path/to/file.csv",
            path_in_name=True,
            path_sep="-",
            remove_extension=True,
        )
        assert result == "-path-to-file"

    def test_keep_extension(self):
        """Test keeping file extension."""
        result = _name_from_path(
            "/path/to/file.csv",
            path_in_name=False,
            path_sep="-",
            remove_extension=False,
        )
        assert result == "file.csv"


@pytest.mark.quick
class TestGenerateConfig:
    """Test generate_config function."""

    def test_creates_config_file(self, tmp_path):
        """Test that config file is created."""
        project_dir = str(tmp_path)
        generate_config(project_dir)

        config_path = tmp_path / "config.yml"
        assert config_path.exists()

    def test_config_has_required_sections(self, tmp_path):
        """Test that config has all required sections."""
        project_dir = str(tmp_path)
        generate_config(project_dir)

        with open(tmp_path / "config.yml") as f:
            content = f.read()

        assert "ANATOMY" in content
        assert "FITTING" in content
        assert "HYPER PARAMS" in content
        assert "OTHER" in content

    def test_custom_values_override_defaults(self, tmp_path):
        """Test that custom values override defaults."""
        project_dir = str(tmp_path)
        generate_config(project_dir, fps=60, verbose=True)

        config = yaml.safe_load(open(tmp_path / "config.yml"))
        assert config["fps"] == 60
        assert config["verbose"] is True

    def test_bodyparts_in_config(self, tmp_path):
        """Test that bodyparts are configured."""
        project_dir = str(tmp_path)
        generate_config(project_dir, bodyparts=["nose", "tail"])

        config = yaml.safe_load(open(tmp_path / "config.yml"))
        assert config["bodyparts"] == ["nose", "tail"]


@pytest.mark.quick
class TestCheckConfigValidity:
    """Test check_config_validity function."""

    def test_valid_config_returns_true(self):
        """Test that valid config returns True."""
        config = {
            "bodyparts": ["bp1", "bp2", "bp3"],
            "use_bodyparts": ["bp1", "bp2"],
            "skeleton": [["bp1", "bp2"]],
            "anterior_bodyparts": ["bp1"],
            "posterior_bodyparts": ["bp2"],
        }
        assert check_config_validity(config) is True

    def test_invalid_use_bodyparts(self, capsys):
        """Test detection of invalid use_bodyparts."""
        config = {
            "bodyparts": ["bp1", "bp2"],
            "use_bodyparts": ["bp1", "bp3"],  # bp3 not in bodyparts
            "skeleton": [],
            "anterior_bodyparts": ["bp1"],
            "posterior_bodyparts": ["bp1"],
        }
        result = check_config_validity(config)
        assert result is False
        captured = capsys.readouterr()
        assert "bp3" in captured.out

    def test_invalid_skeleton_bodypart(self, capsys):
        """Test detection of invalid skeleton bodypart."""
        config = {
            "bodyparts": ["bp1", "bp2"],
            "use_bodyparts": ["bp1", "bp2"],
            "skeleton": [["bp1", "bp3"]],  # bp3 not in bodyparts
            "anterior_bodyparts": ["bp1"],
            "posterior_bodyparts": ["bp2"],
        }
        result = check_config_validity(config)
        assert result is False
        captured = capsys.readouterr()
        assert "bp3" in captured.out

    def test_anterior_not_in_use(self, capsys):
        """Test detection of anterior bodypart not in use_bodyparts."""
        config = {
            "bodyparts": ["bp1", "bp2", "bp3"],
            "use_bodyparts": ["bp1", "bp2"],
            "skeleton": [],
            "anterior_bodyparts": ["bp3"],  # bp3 not in use_bodyparts
            "posterior_bodyparts": ["bp2"],
        }
        result = check_config_validity(config)
        assert result is False


@pytest.mark.quick
class TestLoadConfig:
    """Test load_config function."""

    def test_loads_valid_config(self, tmp_path):
        """Test loading a valid config file."""
        project_dir = str(tmp_path)
        generate_config(project_dir)

        config = load_config(project_dir, check_if_valid=False)

        assert "bodyparts" in config
        assert "fps" in config
        assert "trans_hypparams" in config

    def test_builds_indexes(self, tmp_path):
        """Test that anterior/posterior indexes are built."""
        project_dir = str(tmp_path)
        generate_config(
            project_dir,
            bodyparts=["bp1", "bp2", "bp3"],
            use_bodyparts=["bp1", "bp2", "bp3"],
            anterior_bodyparts=["bp1"],
            posterior_bodyparts=["bp3"],
        )

        config = load_config(project_dir, build_indexes=True)

        assert "anterior_idxs" in config
        assert "posterior_idxs" in config
        assert config["anterior_idxs"][0] == 0  # bp1 is at index 0
        assert config["posterior_idxs"][0] == 2  # bp3 is at index 2

    def test_skip_validity_check(self, tmp_path):
        """Test loading without validity check."""
        project_dir = str(tmp_path)
        # Create invalid config
        generate_config(
            project_dir,
            bodyparts=["bp1"],
            use_bodyparts=["bp2"],  # Invalid: bp2 not in bodyparts
        )

        # Should not raise error with check_if_valid=False
        config = load_config(project_dir, check_if_valid=False, build_indexes=False)
        assert config is not None


@pytest.mark.quick
class TestUpdateConfig:
    """Test update_config function."""

    def test_updates_top_level_key(self, tmp_path):
        """Test updating a top-level config key."""
        project_dir = str(tmp_path)
        generate_config(project_dir, fps=30)

        update_config(project_dir, fps=60)

        config = load_config(project_dir, check_if_valid=False)
        assert config["fps"] == 60

    def test_updates_hyperparam(self, tmp_path):
        """Test updating a hyperparameter."""
        project_dir = str(tmp_path)
        generate_config(project_dir)

        update_config(project_dir, kappa=1e5)

        config = load_config(project_dir, check_if_valid=False)
        assert config["trans_hypparams"]["kappa"] == 1e5

    def test_updates_multiple_keys(self, tmp_path):
        """Test updating multiple keys at once."""
        project_dir = str(tmp_path)
        generate_config(project_dir)

        update_config(project_dir, fps=45, verbose=True, kappa=1e4)

        config = load_config(project_dir, check_if_valid=False)
        assert config["fps"] == 45
        assert config["verbose"] is True
        assert config["trans_hypparams"]["kappa"] == 1e4


@pytest.mark.quick
class TestPCAPersistence:
    """Test PCA save/load functions."""

    def test_save_and_load_pca(self, tmp_path):
        """Test saving and loading PCA model."""
        from sklearn.decomposition import PCA

        project_dir = str(tmp_path)

        # Create real PCA object with fitted data
        X = np.random.randn(100, 20)
        pca = PCA(n_components=10)
        pca.fit(X)

        # Save PCA
        save_pca(pca, project_dir)

        # Load PCA
        loaded_pca = load_pca(project_dir)

        # Verify loaded
        assert loaded_pca is not None
        np.testing.assert_array_almost_equal(
            loaded_pca.components_, pca.components_
        )

    def test_save_with_custom_path(self, tmp_path):
        """Test saving PCA with custom path."""
        from sklearn.decomposition import PCA

        # Create real PCA object
        X = np.random.randn(50, 10)
        pca = PCA(n_components=5)
        pca.fit(X)

        custom_path = str(tmp_path / "custom_pca.p")
        save_pca(pca, str(tmp_path), pca_path=custom_path)

        assert Path(custom_path).exists()

    def test_load_nonexistent_raises_error(self, tmp_path):
        """Test loading nonexistent PCA raises error."""
        with pytest.raises(AssertionError, match="No PCA model found"):
            load_pca(str(tmp_path))


@pytest.mark.quick
class TestHDF5Operations:
    """Test HDF5 save/load functions."""

    def test_save_and_load_simple_dict(self, tmp_path):
        """Test saving and loading simple dictionary."""
        filepath = str(tmp_path / "test.h5")
        data = {
            "array": np.array([1, 2, 3]),
            "scalar": 42,
            "string": "test",
        }

        save_hdf5(filepath, data)
        loaded = load_hdf5(filepath)

        np.testing.assert_array_equal(loaded["array"], data["array"])
        assert loaded["scalar"] == data["scalar"]
        assert loaded["string"] == data["string"]

    def test_save_nested_dict(self, tmp_path):
        """Test saving nested dictionary structure."""
        filepath = str(tmp_path / "nested.h5")
        data = {
            "level1": {
                "level2": {
                    "array": np.array([1, 2, 3]),
                    "value": 123,
                }
            }
        }

        save_hdf5(filepath, data)
        loaded = load_hdf5(filepath)

        assert "level1" in loaded
        assert "level2" in loaded["level1"]
        np.testing.assert_array_equal(
            loaded["level1"]["level2"]["array"],
            data["level1"]["level2"]["array"],
        )

    def test_save_with_datapath(self, tmp_path):
        """Test saving to specific path within HDF5."""
        filepath = str(tmp_path / "datapath.h5")
        data = {"value": 42}

        save_hdf5(filepath, data, datapath="custom/path")

        with h5py.File(filepath, "r") as f:
            assert "custom" in f
            assert "path" in f["custom"]

    def test_exist_ok_false_prevents_overwrite(self, tmp_path):
        """Test that exist_ok=False prevents overwriting."""
        filepath = str(tmp_path / "exists.h5")

        save_hdf5(filepath, {"data": 1})

        with pytest.raises(AssertionError, match="already exists"):
            save_hdf5(filepath, {"data": 2}, exist_ok=False)

    def test_exist_ok_true_allows_append(self, tmp_path):
        """Test that exist_ok=True allows appending."""
        filepath = str(tmp_path / "append.h5")

        save_hdf5(filepath, {"data1": 1})
        save_hdf5(filepath, {"data2": 2}, exist_ok=True)

        loaded = load_hdf5(filepath)
        assert "data1" in loaded
        assert "data2" in loaded


@pytest.mark.quick
class TestExtractResults:
    """Test extract_results function."""

    def test_extract_results_structure(self, tmp_path):
        """Test that extract_results creates correct structure."""
        # Mock model with states
        model = {
            "states": {
                "x": np.random.randn(10, 5),
                "z": np.zeros((10, 2), dtype=int),
                "v": np.random.randn(10, 2),
                "h": np.random.randn(10),
            }
        }
        metadata = (["recording1"], np.array([[0, 10]]))

        with patch("jax.device_get", side_effect=lambda x: x):
            with patch("keypoint_moseq.io.unbatch") as mock_unbatch:
                # Mock unbatch to return simple dict
                mock_unbatch.return_value = {"recording1": np.random.randn(10, 5)}

                results = extract_results(
                    model,
                    metadata,
                    save_results=False,
                )

        assert "recording1" in results
        assert "syllable" in results["recording1"]
        assert "latent_state" in results["recording1"]
        assert "centroid" in results["recording1"]
        assert "heading" in results["recording1"]

    def test_save_results_to_file(self, tmp_path):
        """Test saving results to file."""
        model = {
            "states": {
                "x": np.random.randn(10, 5),
                "z": np.zeros((10, 2), dtype=int),
                "v": np.random.randn(10, 2),
                "h": np.random.randn(10),
            }
        }
        metadata = (["rec1"], np.array([[0, 10]]))
        project_dir = str(tmp_path)
        model_name = "test_model"

        # Create model directory
        os.makedirs(os.path.join(project_dir, model_name))

        with patch("jax.device_get", side_effect=lambda x: x):
            with patch("keypoint_moseq.io.unbatch") as mock_unbatch:
                mock_unbatch.return_value = {"rec1": np.random.randn(10, 5)}

                extract_results(
                    model,
                    metadata,
                    project_dir=project_dir,
                    model_name=model_name,
                    save_results=True,
                )

        results_path = Path(project_dir) / model_name / "results.h5"
        assert results_path.exists()


@pytest.mark.quick
class TestLoadResults:
    """Test load_results function."""

    def test_load_results_from_default_path(self, tmp_path):
        """Test loading results from default path."""
        project_dir = str(tmp_path)
        model_name = "test_model"
        results_path = tmp_path / model_name / "results.h5"
        results_path.parent.mkdir(parents=True)

        # Create mock results file
        test_data = {"rec1": {"syllable": np.array([0, 1, 2])}}
        save_hdf5(str(results_path), test_data)

        loaded = load_results(project_dir=project_dir, model_name=model_name)

        assert "rec1" in loaded
        np.testing.assert_array_equal(loaded["rec1"]["syllable"], test_data["rec1"]["syllable"])


@pytest.mark.quick
class TestSaveResultsAsCsv:
    """Test save_results_as_csv function."""

    def test_creates_csv_files(self, tmp_path):
        """Test that CSV files are created."""
        results = {
            "recording1": {
                "syllable": np.array([0, 1, 2, 1, 0]),
                "centroid": np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]),
                "heading": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            }
        }

        save_dir = str(tmp_path / "csv_results")
        save_results_as_csv(results, save_dir=save_dir)

        csv_path = Path(save_dir) / "recording1.csv"
        assert csv_path.exists()

    def test_csv_contains_correct_columns(self, tmp_path):
        """Test that CSV has correct column structure."""
        import pandas as pd

        results = {
            "rec1": {
                "syllable": np.array([0, 1, 2]),
                "centroid": np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]]),
                "heading": np.array([0.1, 0.2, 0.3]),
                "latent_state": np.random.randn(3, 5),
            }
        }

        save_dir = str(tmp_path / "csv_test")
        save_results_as_csv(results, save_dir=save_dir)

        df = pd.read_csv(Path(save_dir) / "rec1.csv")

        assert "syllable" in df.columns
        assert "centroid x" in df.columns
        assert "centroid y" in df.columns
        assert "heading" in df.columns
        assert "latent_state 0" in df.columns

    def test_path_separator_replacement(self, tmp_path):
        """Test that path separators are replaced."""
        results = {
            "path/to/recording": {
                "syllable": np.array([0, 1, 2]),
            }
        }

        save_dir = str(tmp_path / "csv_pathsep")
        save_results_as_csv(results, save_dir=save_dir, path_sep="_")

        csv_path = Path(save_dir) / "path_to_recording.csv"
        assert csv_path.exists()


@pytest.mark.quick
class TestSaveKeypoints:
    """Test save_keypoints function."""

    def test_saves_coordinates_only(self, tmp_path):
        """Test saving coordinates without confidences."""
        import pandas as pd

        coordinates = {
            "rec1": np.random.randn(10, 3, 2),  # 10 frames, 3 keypoints, 2D
        }
        bodyparts = ["bp1", "bp2", "bp3"]

        save_dir = str(tmp_path / "keypoints")
        save_keypoints(save_dir, coordinates, bodyparts=bodyparts)

        csv_path = Path(save_dir) / "rec1.csv"
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert "bp1_x" in df.columns
        assert "bp1_y" in df.columns
        assert "bp2_x" in df.columns

    def test_saves_with_confidences(self, tmp_path):
        """Test saving coordinates with confidences."""
        import pandas as pd

        coordinates = {
            "rec1": np.random.randn(10, 2, 2),
        }
        confidences = {
            "rec1": np.random.rand(10, 2),
        }
        bodyparts = ["bp1", "bp2"]

        save_dir = str(tmp_path / "keypoints_conf")
        save_keypoints(save_dir, coordinates, confidences=confidences, bodyparts=bodyparts)

        df = pd.read_csv(Path(save_dir) / "rec1.csv")
        assert "bp1_conf" in df.columns
        assert "bp2_conf" in df.columns

    def test_3d_coordinates(self, tmp_path):
        """Test saving 3D coordinates."""
        import pandas as pd

        coordinates = {
            "rec1": np.random.randn(5, 2, 3),  # 5 frames, 2 keypoints, 3D
        }
        bodyparts = ["bp1", "bp2"]

        save_dir = str(tmp_path / "keypoints_3d")
        save_keypoints(save_dir, coordinates, bodyparts=bodyparts)

        df = pd.read_csv(Path(save_dir) / "rec1.csv")
        assert "bp1_x" in df.columns
        assert "bp1_y" in df.columns
        assert "bp1_z" in df.columns


@pytest.mark.quick
class TestSetupProject:
    """Test setup_project function."""

    def test_creates_project_directory(self, tmp_path):
        """Test that project directory is created."""
        project_dir = str(tmp_path / "new_project")
        setup_project(project_dir)

        assert Path(project_dir).exists()
        assert (Path(project_dir) / "config.yml").exists()

    def test_existing_directory_no_overwrite(self, tmp_path, capsys):
        """Test that existing directory is not overwritten without flag."""
        project_dir = str(tmp_path / "existing")
        setup_project(project_dir)

        # Try to setup again without overwrite
        setup_project(project_dir, overwrite=False)

        captured = capsys.readouterr()
        assert "already exists" in captured.out

    def test_existing_directory_with_overwrite(self, tmp_path):
        """Test that existing directory can be overwritten with flag."""
        project_dir = str(tmp_path / "existing")
        setup_project(project_dir, fps=30)

        # Setup again with overwrite and different fps
        setup_project(project_dir, fps=60, overwrite=True)

        config = load_config(project_dir, check_if_valid=False)
        assert config["fps"] == 60

    def test_with_custom_options(self, tmp_path):
        """Test setup with custom configuration options."""
        project_dir = str(tmp_path / "custom")
        setup_project(
            project_dir,
            fps=45,
            bodyparts=["nose", "tail", "back"],
            verbose=True
        )

        config = load_config(project_dir, check_if_valid=False)
        assert config["fps"] == 45
        assert config["bodyparts"] == ["nose", "tail", "back"]
        assert config["verbose"] is True


@pytest.mark.quick
class TestCheckpointOperations:
    """Test checkpoint loading and reindexing."""

    def test_load_checkpoint_with_explicit_path(self, tmp_path):
        """Test loading checkpoint from explicit path."""
        checkpoint_path = str(tmp_path / "checkpoint.h5")

        # Create mock checkpoint
        model_data = {
            "params": {"pi": np.eye(3)},
            "states": {"z": np.array([0, 1, 2])},
        }
        data = {"Y": np.random.randn(100, 10)}
        metadata = {"keys": ["rec1"], "bounds": np.array([[0, 100]])}

        save_hdf5(checkpoint_path, {"model_snapshots": {"50": model_data}}, exist_ok=True)
        save_hdf5(checkpoint_path, {"data": data}, exist_ok=True)
        save_hdf5(checkpoint_path, {"metadata": metadata}, exist_ok=True)

        model, loaded_data, loaded_metadata, iteration = load_checkpoint(
            path=checkpoint_path
        )

        assert iteration == 50
        assert "params" in model
        assert "states" in model

    def test_load_checkpoint_from_project_dir(self, tmp_path):
        """Test loading checkpoint using project_dir and model_name."""
        project_dir = str(tmp_path)
        model_name = "test_model"
        checkpoint_path = tmp_path / model_name / "checkpoint.h5"
        checkpoint_path.parent.mkdir(parents=True)

        # Create minimal checkpoint
        model_data = {
            "params": {"pi": np.eye(2)},
            "states": {"z": np.array([0, 1])},
        }

        save_hdf5(str(checkpoint_path), {"model_snapshots": {"100": model_data}}, exist_ok=True)
        save_hdf5(str(checkpoint_path), {"data": {"Y": np.random.randn(10, 5)}}, exist_ok=True)
        save_hdf5(str(checkpoint_path), {"metadata": {"keys": ["rec1"], "bounds": np.array([[0, 10]])}}, exist_ok=True)

        model, data, metadata, iteration = load_checkpoint(
            project_dir=project_dir,
            model_name=model_name
        )

        assert iteration == 100

    def test_load_checkpoint_specific_iteration(self, tmp_path):
        """Test loading specific iteration from checkpoint."""
        checkpoint_path = str(tmp_path / "multi_snapshot.h5")

        # Create checkpoint with multiple snapshots
        for it in [10, 20, 30]:
            model_data = {
                "params": {"value": it},
                "states": {"z": np.array([it])},
            }
            save_hdf5(checkpoint_path, {f"model_snapshots/{it}": model_data}, exist_ok=True)

        save_hdf5(checkpoint_path, {"data": {"Y": np.random.randn(10, 5)}}, exist_ok=True)
        save_hdf5(checkpoint_path, {"metadata": {"keys": ["rec1"], "bounds": np.array([[0, 10]])}}, exist_ok=True)

        # Load iteration 20
        model, _, _, iteration = load_checkpoint(path=checkpoint_path, iteration=20)

        assert iteration == 20
        assert model["params"]["value"] == 20

    def test_reindex_syllables_modifies_checkpoint(self, tmp_path):
        """Test that reindex_syllables modifies checkpoint in place."""
        checkpoint_path = str(tmp_path / "reindex.h5")
        num_states = 5

        # Create checkpoint with model snapshot
        model_data = {
            "params": {
                "betas": np.arange(num_states),
                "pi": np.eye(num_states),
                "Ab": np.arange(num_states),
                "Q": np.arange(num_states),
            },
            "states": {
                "z": np.array([0, 1, 2, 3, 4, 0, 1]),
            }
        }

        save_hdf5(checkpoint_path, {"model_snapshots": {"50": model_data}}, exist_ok=True)
        save_hdf5(checkpoint_path, {"data": {"mask": np.ones(7, dtype=bool)}}, exist_ok=True)

        # Reindex with custom index (reverse order)
        custom_index = np.array([4, 3, 2, 1, 0])
        returned_index = reindex_syllables_in_checkpoint(
            path=checkpoint_path,
            index=custom_index
        )

        np.testing.assert_array_equal(returned_index, custom_index)

        # Load and verify reindexing happened
        reindexed_model = load_hdf5(checkpoint_path, "model_snapshots/50")

        # betas should be reordered
        np.testing.assert_array_equal(
            reindexed_model["params"]["betas"],
            np.array([4, 3, 2, 1, 0])
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
