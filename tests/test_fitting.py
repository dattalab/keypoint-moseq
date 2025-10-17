"""
Unit tests for keypoint_moseq.fitting module

Target coverage: 36% → 80% (from 241/671 statements to ~536/671)
Functions tested:
- _wrapped_resample()
- _set_parallel_flag()
- init_model()
- fit_model()
- apply_model()
- estimate_syllable_marginals()
- update_hypparams()
- expected_marginal_likelihoods()
"""

import os
import tempfile
import warnings
from unittest.mock import Mock, patch

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from keypoint_moseq.fitting import (
    StopResampling,
    _set_parallel_flag,
    _wrapped_resample,
    apply_model,
    estimate_syllable_marginals,
    expected_marginal_likelihoods,
    fit_model,
    init_model,
    update_hypparams,
)

# Suppress JAX/matplotlib warnings for clean test output
warnings.filterwarnings("ignore", category=UserWarning, message=".*os.fork.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*FigureCanvasAgg.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*NVIDIA GPU.*")


@pytest.mark.quick
class TestWrappedResample:
    """Test _wrapped_resample function."""

    def test_successful_resample(self):
        """Test successful resampling without NaNs or interrupts."""
        # Mock resample function that returns updated model
        mock_resample = Mock(return_value={"states": {"x": jnp.array([1.0])}})
        data = {"Y": jnp.array([1.0])}
        model = {"states": {"x": jnp.array([0.5])}}

        with patch("keypoint_moseq.fitting.check_for_nans") as mock_check:
            mock_check.return_value = (False, {}, [])
            result = _wrapped_resample(mock_resample, data, model)

        assert "states" in result
        mock_resample.assert_called_once_with(data, **model)

    def test_keyboard_interrupt(self):
        """Test KeyboardInterrupt handling."""
        mock_resample = Mock(side_effect=KeyboardInterrupt)
        data = {"Y": jnp.array([1.0])}
        model = {"states": {"x": jnp.array([0.5])}}

        with pytest.raises(StopResampling):
            _wrapped_resample(mock_resample, data, model)

    def test_nan_detection(self):
        """Test NaN detection during resampling."""
        mock_resample = Mock(return_value={"states": {"x": jnp.array([np.nan])}})
        data = {"Y": jnp.array([1.0])}
        model = {"states": {"x": jnp.array([0.5])}}

        with patch("keypoint_moseq.fitting.check_for_nans") as mock_check:
            mock_check.return_value = (
                True,
                {"x": np.nan},
                ["NaN found in states.x"],
            )
            with pytest.warns(UserWarning, match="Early termination.*NaNs"):
                with pytest.raises(StopResampling):
                    _wrapped_resample(mock_resample, data, model)

    def test_with_progress_bar(self):
        """Test with progress bar parameter."""
        mock_resample = Mock(return_value={"states": {"x": jnp.array([1.0])}})
        mock_pbar = Mock()
        data = {"Y": jnp.array([1.0])}
        model = {"states": {"x": jnp.array([0.5])}}

        with patch("keypoint_moseq.fitting.check_for_nans") as mock_check:
            mock_check.return_value = (False, {}, [])
            result = _wrapped_resample(mock_resample, data, model, pbar=mock_pbar)

        assert "states" in result

    def test_nan_with_progress_bar_closes(self):
        """Test that progress bar is closed when NaN detected."""
        mock_resample = Mock(return_value={"states": {"x": jnp.array([np.nan])}})
        mock_pbar = Mock()
        data = {"Y": jnp.array([1.0])}
        model = {"states": {"x": jnp.array([0.5])}}

        with patch("keypoint_moseq.fitting.check_for_nans") as mock_check:
            mock_check.return_value = (True, {}, ["NaN detected"])
            with pytest.warns(UserWarning):
                with pytest.raises(StopResampling):
                    _wrapped_resample(mock_resample, data, model, pbar=mock_pbar)

        mock_pbar.close.assert_called_once()


@pytest.mark.quick
class TestSetParallelFlag:
    """Test _set_parallel_flag function."""

    def test_force_true(self):
        """Test force=True always returns True."""
        result = _set_parallel_flag("force")
        assert result is True

    def test_none_with_gpu(self):
        """Test None with GPU backend."""
        with patch("jax.default_backend", return_value="gpu"):
            result = _set_parallel_flag(None)
            assert result is True

    def test_none_with_cpu(self):
        """Test None with CPU backend."""
        with patch("jax.default_backend", return_value="cpu"):
            result = _set_parallel_flag(None)
            assert result is False

    def test_explicit_true_with_cpu_warns(self):
        """Test explicit True with CPU backend raises warning."""
        with patch("jax.default_backend", return_value="cpu"):
            with pytest.warns(UserWarning, match="CPU-bound"):
                result = _set_parallel_flag(True)
                assert result is True

    def test_explicit_false(self):
        """Test explicit False returns False."""
        result = _set_parallel_flag(False)
        assert result is False


@pytest.mark.quick
class TestInitModel:
    """Test init_model function."""

    def test_standard_model(self):
        """Test standard keypoint-SLDS model initialization."""
        data = {"Y": jnp.ones((10, 5, 2))}

        with patch("keypoint_moseq.fitting.keypoint_slds.init_model") as mock_init:
            mock_init.return_value = {"model": "standard"}
            result = init_model(data, location_aware=False)

        assert result == {"model": "standard"}
        mock_init.assert_called_once()

    def test_location_aware_model(self):
        """Test location-aware model initialization."""
        data = {"Y": jnp.ones((10, 5, 2))}
        trans_hypparams = {"num_states": 50}

        with patch("keypoint_moseq.fitting.allo_keypoint_slds.init_model") as mock_init:
            mock_init.return_value = {"model": "allow"}
            result = init_model(
                data,
                location_aware=True,
                trans_hypparams=trans_hypparams,
            )

        assert result == {"model": "allow"}
        mock_init.assert_called_once()

    def test_location_aware_allo_hypparams(self):
        """Test that location-aware model sets allo_hypparams."""
        data = {"Y": jnp.ones((10, 5, 2))}
        trans_hypparams = {"num_states": 30}

        with patch("keypoint_moseq.fitting.allo_keypoint_slds.init_model") as mock_init:
            mock_init.return_value = {"model": "allow"}
            _ = init_model(
                data,
                location_aware=True,
                trans_hypparams=trans_hypparams,
            )

        # Check that allo_hypparams were passed
        call_kwargs = mock_init.call_args[1]
        assert "allo_hypparams" in call_kwargs
        assert call_kwargs["allo_hypparams"]["num_states"] == 30


@pytest.mark.quick
class TestUpdateHypparams:
    """Test update_hypparams function."""

    def test_update_scalar_hyperparam(self):
        """Test updating a scalar hyperparameter."""
        model = {
            "hypparams": {
                "trans_hypparams": {"kappa": 1e3},
                "ar_hypparams": {"nlags": 3},
            }
        }

        result = update_hypparams(model, kappa=1e4)

        assert result["hypparams"]["trans_hypparams"]["kappa"] == 1e4

    def test_update_multiple_hypparams(self):
        """Test updating multiple hyperparameters."""
        model = {
            "hypparams": {
                "trans_hypparams": {"kappa": 1e3, "gamma": 1e2},
                "ar_hypparams": {"nlags": 3},
            }
        }

        result = update_hypparams(model, kappa=5e3, gamma=5e2)

        assert result["hypparams"]["trans_hypparams"]["kappa"] == 5e3
        assert result["hypparams"]["trans_hypparams"]["gamma"] == 5e2

    def test_type_conversion_warning(self):
        """Test warning when type conversion is needed."""
        model = {
            "hypparams": {
                "trans_hypparams": {"kappa": 1000.0},  # float
            }
        }

        with pytest.warns(UserWarning, match="will be cast"):
            result = update_hypparams(model, kappa=2000)  # int

        assert result["hypparams"]["trans_hypparams"]["kappa"] == 2000.0

    def test_non_scalar_hyperparam_not_updated(self):
        """Test that non-scalar hyperparameters are not updated."""
        model = {
            "hypparams": {
                "trans_hypparams": {
                    "kappa": 1e3,
                    "matrix_param": np.array([[1, 2], [3, 4]]),
                },
            }
        }

        # Should print message but not raise error
        result = update_hypparams(model, matrix_param=np.array([[5, 6], [7, 8]]))

        # Original matrix should be unchanged
        np.testing.assert_array_equal(
            result["hypparams"]["trans_hypparams"]["matrix_param"],
            np.array([[1, 2], [3, 4]]),
        )

    def test_unknown_hyperparam_warns(self):
        """Test warning for unknown hyperparameter."""
        model = {
            "hypparams": {
                "trans_hypparams": {"kappa": 1e3},
            }
        }

        with pytest.warns(UserWarning, match="not found"):
            _ = update_hypparams(model, unknown_param=999)

    def test_missing_hypparams_raises(self):
        """Test error when model has no hypparams."""
        model = {"states": {}, "params": {}}

        with pytest.raises(AssertionError, match="does not contain any hyperparams"):
            update_hypparams(model, kappa=1e4)


@pytest.mark.quick
class TestFitModelParameters:
    """Test fit_model parameter validation and setup."""

    def test_explicit_model_name_used(self):
        """Test that explicit model name is used instead of auto-generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._create_mock_model()
            data = self._create_mock_data()
            metadata = (["rec1"], np.array([[0, 100]]))

            # This simpler test just checks the directory is created with the right name
            test_name = "my_custom_model"

            with patch("keypoint_moseq.fitting._wrapped_resample") as mock_resample:
                mock_resample.return_value = model
                with patch("keypoint_moseq.fitting.save_hdf5"):
                    with patch(
                        "keypoint_moseq.fitting.device_put_as_scalar"
                    ) as mock_device:
                        mock_device.return_value = model
                        _, returned_name = fit_model(
                            model,
                            data,
                            metadata,
                            project_dir=tmpdir,
                            model_name=test_name,
                            num_iters=0,  # No iterations, just check setup
                        )

            assert returned_name == test_name
            assert os.path.exists(os.path.join(tmpdir, test_name))

    def test_save_every_n_iters_none_no_save(self):
        """Test save_every_n_iters=None disables saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._create_mock_model()
            data = self._create_mock_data()
            metadata = (["rec1"], np.array([[0, 100]]))

            with patch("keypoint_moseq.fitting._wrapped_resample") as mock_resample:
                mock_resample.return_value = model
                with patch("keypoint_moseq.fitting.save_hdf5") as mock_save:
                    with patch(
                        "keypoint_moseq.fitting.device_put_as_scalar"
                    ) as mock_device:
                        mock_device.return_value = model
                        result, _ = fit_model(
                            model,
                            data,
                            metadata,
                            project_dir=tmpdir,
                            save_every_n_iters=None,
                            num_iters=2,
                        )

                # save_hdf5 should not be called
                mock_save.assert_not_called()

    def test_progress_plots_require_saving(self):
        """Test warning when progress plots requested but saving disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._create_mock_model()
            data = self._create_mock_data()
            metadata = (["rec1"], np.array([[0, 100]]))

            with pytest.warns(UserWarning, match="Progress plots"):
                with patch("keypoint_moseq.fitting._wrapped_resample") as mock_resample:
                    mock_resample.return_value = model
                    with patch(
                        "keypoint_moseq.fitting.device_put_as_scalar"
                    ) as mock_device:
                        mock_device.return_value = model
                        fit_model(
                            model,
                            data,
                            metadata,
                            project_dir=tmpdir,
                            save_every_n_iters=0,
                            generate_progress_plots=True,
                            num_iters=1,
                        )

    def test_ar_only_mode(self):
        """Test AR-only fitting mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._create_mock_model()
            data = self._create_mock_data()
            metadata = (["rec1"], np.array([[0, 100]]))

            with patch("keypoint_moseq.fitting._wrapped_resample") as mock_resample:
                mock_resample.return_value = model
                with patch(
                    "keypoint_moseq.fitting.device_put_as_scalar"
                ) as mock_device:
                    mock_device.return_value = model
                    fit_model(
                        model,
                        data,
                        metadata,
                        project_dir=tmpdir,
                        save_every_n_iters=None,
                        ar_only=True,
                        num_iters=1,
                    )

            # Check ar_only was passed
            call_kwargs = mock_resample.call_args[1]
            assert call_kwargs["ar_only"] is True

    def test_location_aware_uses_allo_resample(self):
        """Test location_aware mode uses allow resample function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._create_mock_model()
            data = self._create_mock_data()
            metadata = (["rec1"], np.array([[0, 100]]))

            with patch(
                "keypoint_moseq.fitting.allo_keypoint_slds.resample_model"
            ) as mock_allo:
                mock_allo.return_value = model
                with patch(
                    "keypoint_moseq.fitting.device_put_as_scalar"
                ) as mock_device:
                    mock_device.return_value = model
                    fit_model(
                        model,
                        data,
                        metadata,
                        project_dir=tmpdir,
                        save_every_n_iters=None,
                        location_aware=True,
                        num_iters=1,
                    )

            assert mock_allo.called

    # Helper methods
    def _create_mock_model(self):
        """Create a minimal mock model."""
        return {
            "states": {"x": jnp.ones((10, 5)), "z": jnp.zeros(10, dtype=int)},
            "params": {"Ab": jnp.eye(5)},
            "hypparams": {"trans_hypparams": {"num_states": 10}},
            "seed": 0,
        }

    def _create_mock_data(self):
        """Create minimal mock data."""
        return {
            "Y": jnp.ones((10, 5, 2)),
            "mask": jnp.ones((10, 5), dtype=bool),
        }


@pytest.mark.quick
class TestApplyModelBasics:
    """Test apply_model basic functionality."""

    def test_save_results_requires_params(self):
        """Test that save_results=True requires project_dir and model_name."""
        model = self._create_mock_model()
        data = self._create_mock_data()
        metadata = (["rec1"], np.array([[0, 100]]))

        with pytest.raises(AssertionError, match="requires either"):
            apply_model(
                model,
                data,
                metadata,
                save_results=True,
                # Missing project_dir and model_name
            )

    def test_results_path_override(self):
        """Test that results_path overrides project_dir/model_name."""
        model = self._create_mock_model()
        data = self._create_mock_data()
        metadata = (["rec1"], np.array([[0, 100]]))

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_results.h5")

            with patch("keypoint_moseq.fitting._wrapped_resample") as mock_resample:
                mock_resample.return_value = model
                with patch("keypoint_moseq.fitting.init_model") as mock_init:
                    mock_init.return_value = model
                    with patch(
                        "keypoint_moseq.fitting.extract_results"
                    ) as mock_extract:
                        mock_extract.return_value = {"rec1": {}}
                        with patch("jax.device_put") as mock_device:
                            mock_device.return_value = data
                            apply_model(
                                model,
                                data,
                                metadata,
                                save_results=True,
                                results_path=custom_path,
                                num_iters=1,
                            )

            # Check extract_results was called - check positional args
            call_args = mock_extract.call_args[0]
            # extract_results(model, metadata, project_dir, model_name, save_results, results_path)
            # Custom path should be the last positional arg
            assert call_args[-1] == custom_path

    def test_return_model_option(self):
        """Test return_model=True returns both results and model."""
        model = self._create_mock_model()
        data = self._create_mock_data()
        metadata = (["rec1"], np.array([[0, 100]]))

        with patch("keypoint_moseq.fitting._wrapped_resample") as mock_resample:
            mock_resample.return_value = model
            with patch("keypoint_moseq.fitting.init_model") as mock_init:
                mock_init.return_value = model
                with patch("keypoint_moseq.fitting.extract_results") as mock_extract:
                    mock_extract.return_value = {"rec1": {}}
                    with patch("jax.device_put") as mock_device:
                        mock_device.return_value = data
                        results, returned_model = apply_model(
                            model,
                            data,
                            metadata,
                            save_results=False,
                            return_model=True,
                            num_iters=1,
                        )

        assert "rec1" in results
        assert returned_model == model

    def test_location_aware_apply(self):
        """Test location_aware mode in apply_model."""
        model = self._create_mock_model()
        data = self._create_mock_data()
        metadata = (["rec1"], np.array([[0, 100]]))

        with patch(
            "keypoint_moseq.fitting.allo_keypoint_slds.resample_model"
        ) as mock_allo:
            mock_allo.return_value = model
            with patch("keypoint_moseq.fitting.init_model") as mock_init:
                mock_init.return_value = model
                with patch("keypoint_moseq.fitting.extract_results") as mock_extract:
                    mock_extract.return_value = {"rec1": {}}
                    with patch("jax.device_put") as mock_device:
                        mock_device.return_value = data
                        apply_model(
                            model,
                            data,
                            metadata,
                            save_results=False,
                            location_aware=True,
                            num_iters=1,
                        )

        assert mock_allo.called

    # Helper methods
    def _create_mock_model(self):
        """Create a minimal mock model."""
        return {
            "states": {"x": jnp.ones((10, 5)), "z": jnp.zeros(10, dtype=int)},
            "params": {"Ab": jnp.eye(5)},
            "hypparams": {"trans_hypparams": {"num_states": 10}},
            "seed": 0,
        }

    def _create_mock_data(self):
        """Create minimal mock data."""
        return {
            "Y": jnp.ones((10, 5, 2)),
            "mask": jnp.ones((10, 5), dtype=bool),
        }


@pytest.mark.quick
class TestEstimateSyllableMarginals:
    """Test estimate_syllable_marginals function."""

    def test_basic_marginal_estimation(self):
        """Test basic marginal estimation."""
        model = self._create_mock_model()
        data = self._create_mock_data()
        # Bounds must match data shape
        metadata = (["rec1"], np.array([[0, 100]]))

        with patch("keypoint_moseq.fitting._wrapped_resample") as mock_resample:
            mock_resample.return_value = model
            with patch("keypoint_moseq.fitting.init_model") as mock_init:
                mock_init.return_value = model
                with patch(
                    "keypoint_moseq.fitting.stateseq_marginals"
                ) as mock_marginals:
                    # Return marginals for 10 states
                    mock_marginals.return_value = jnp.ones((100, 10))
                    with patch("keypoint_moseq.fitting.get_nlags") as mock_nlags:
                        mock_nlags.return_value = 3
                        with patch("keypoint_moseq.fitting.unbatch") as mock_unbatch:
                            # Return unbatched result directly
                            mock_unbatch.return_value = {"rec1": np.ones((97, 10))}
                            with patch("jax.device_put") as mock_device:
                                mock_device.return_value = data
                                result = estimate_syllable_marginals(
                                    model,
                                    data,
                                    metadata,
                                    burn_in_iters=2,
                                    num_samples=2,
                                    steps_per_sample=1,
                                )

        assert "rec1" in result
        assert result["rec1"].shape[1] == 10  # num_syllables

    def test_return_samples_option(self):
        """Test return_samples=True returns both marginals and samples."""
        model = self._create_mock_model()
        data = self._create_mock_data()
        metadata = (["rec1"], np.array([[0, 100]]))

        with patch("keypoint_moseq.fitting._wrapped_resample") as mock_resample:
            mock_resample.return_value = model
            with patch("keypoint_moseq.fitting.init_model") as mock_init:
                mock_init.return_value = model
                with patch(
                    "keypoint_moseq.fitting.stateseq_marginals"
                ) as mock_marginals:
                    mock_marginals.return_value = jnp.ones((100, 10))
                    with patch("keypoint_moseq.fitting.get_nlags") as mock_nlags:
                        mock_nlags.return_value = 3
                        with patch("keypoint_moseq.fitting.unbatch") as mock_unbatch:
                            # Return unbatched result directly (call count = 2, one for marginals, one for samples)
                            mock_unbatch.side_effect = [
                                {"rec1": np.ones((97, 10))},  # marginals
                                {"rec1": np.ones((97, 2))},  # samples
                            ]
                            with patch("numpy.moveaxis") as mock_moveaxis:
                                # Mock moveaxis to avoid shape issues
                                mock_moveaxis.return_value = np.ones((100, 100, 2))
                                with patch("jax.device_put") as mock_device:
                                    mock_device.return_value = data
                                    marginals, samples = estimate_syllable_marginals(
                                        model,
                                        data,
                                        metadata,
                                        burn_in_iters=1,
                                        num_samples=2,
                                        steps_per_sample=1,
                                        return_samples=True,
                                    )

        assert "rec1" in marginals
        assert "rec1" in samples

    def test_location_aware_marginals(self):
        """Test location_aware mode in marginal estimation."""
        model = self._create_mock_model()
        data = self._create_mock_data()
        metadata = (["rec1"], np.array([[0, 100]]))

        with patch(
            "keypoint_moseq.fitting.allo_keypoint_slds.resample_model"
        ) as mock_allo:
            mock_allo.return_value = model
            with patch("keypoint_moseq.fitting.init_model") as mock_init:
                mock_init.return_value = model
                with patch(
                    "keypoint_moseq.fitting.stateseq_marginals"
                ) as mock_marginals:
                    mock_marginals.return_value = jnp.ones((100, 10))
                    with patch("keypoint_moseq.fitting.get_nlags") as mock_nlags:
                        mock_nlags.return_value = 3
                        with patch("keypoint_moseq.fitting.unbatch") as mock_unbatch:
                            # Return unbatched result directly
                            mock_unbatch.return_value = {"rec1": np.ones((97, 10))}
                            with patch("jax.device_put") as mock_device:
                                mock_device.return_value = data
                                estimate_syllable_marginals(
                                    model,
                                    data,
                                    metadata,
                                    location_aware=True,
                                    burn_in_iters=1,
                                    num_samples=1,
                                )

        assert mock_allo.called

    # Helper methods
    def _create_mock_model(self):
        """Create a minimal mock model."""
        return {
            "states": {"x": jnp.ones((100, 5)), "z": jnp.zeros(100, dtype=int)},
            "params": {"Ab": jnp.eye(5)},
            "hypparams": {"trans_hypparams": {"num_states": 10}},
            "seed": 0,
        }

    def _create_mock_data(self):
        """Create minimal mock data."""
        return {
            "Y": jnp.ones((100, 5, 2)),
            "mask": jnp.ones((100, 5), dtype=bool),
        }


@pytest.mark.quick
class TestExpectedMarginalLikelihoods:
    """Test expected_marginal_likelihoods function."""

    def test_with_checkpoint_paths(self):
        """Test with explicit checkpoint paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two mock checkpoints
            checkpoint_paths = []
            for i in range(2):
                path = os.path.join(tmpdir, f"checkpoint_{i}.h5")
                checkpoint_paths.append(path)
                self._create_mock_checkpoint(path)

            with patch("keypoint_moseq.fitting.load_checkpoint") as mock_load:
                mock_load.return_value = self._create_mock_checkpoint_data()
                with patch(
                    "keypoint_moseq.fitting.marginal_log_likelihood"
                ) as mock_mll:
                    mock_mll.return_value = jnp.array(-100.0)
                    scores, std_errors = expected_marginal_likelihoods(
                        checkpoint_paths=checkpoint_paths
                    )

            assert len(scores) == 2
            assert len(std_errors) == 2

    def test_with_project_dir_and_names(self):
        """Test with project_dir and model_names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model directories
            model_names = ["model_1", "model_2"]
            for name in model_names:
                model_dir = os.path.join(tmpdir, name)
                os.makedirs(model_dir)
                checkpoint = os.path.join(model_dir, "checkpoint.h5")
                self._create_mock_checkpoint(checkpoint)

            with patch("keypoint_moseq.fitting.load_checkpoint") as mock_load:
                mock_load.return_value = self._create_mock_checkpoint_data()
                with patch(
                    "keypoint_moseq.fitting.marginal_log_likelihood"
                ) as mock_mll:
                    mock_mll.return_value = jnp.array(-100.0)
                    scores, std_errors = expected_marginal_likelihoods(
                        project_dir=tmpdir,
                        model_names=model_names,
                    )

            assert len(scores) == 2
            assert len(std_errors) == 2

    def test_requires_params(self):
        """Test that function requires either checkpoint_paths or project_dir+model_names."""
        with pytest.raises(AssertionError, match="Must provide either"):
            expected_marginal_likelihoods()

    # Helper methods
    def _create_mock_checkpoint(self, path):
        """Create a minimal HDF5 checkpoint file."""
        with h5py.File(path, "w") as f:
            f.create_dataset("model/states/x", data=np.ones((10, 5)))
            f.create_dataset("model/params/Ab", data=np.eye(5))

    def _create_mock_checkpoint_data(self):
        """Create mock data returned by load_checkpoint."""
        model = {
            "states": {"x": jnp.ones((10, 5))},
            "params": {
                "Ab": jnp.eye(5),
                "Q": jnp.eye(5) * 0.1,
                "pi": jnp.ones(10) / 10,
            },
        }
        data = {"mask": jnp.ones((10, 5), dtype=bool)}
        metadata = (["rec1"], np.array([[0, 10]]))
        iteration = 100
        return model, data, metadata, iteration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
