"""Out-of-memory errors from JAX are re-raised with the settings that matter.

The tests replace jax-moseq's resampling and initialization with functions
that raise the error XLA raises when an allocation fails, since a real
out-of-memory needs a GPU and a dataset too large for it.
"""

import pytest

import keypoint_moseq as kpms

OUT_OF_MEMORY = "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 91.03GiB."


def raise_out_of_memory(*args, **kwargs):
    raise RuntimeError(OUT_OF_MEMORY)


def total_frames(coordinates):
    return sum(len(v) for v in coordinates.values())


def check_message(error, stage, data_text):
    message = str(error.value)
    assert stage in message
    assert "91.03GiB" in message
    assert data_text in message
    assert "set_mixed_map_iters" in message
    # mixed_map reads the setting when a function is first compiled, so it
    # only applies to a fit started in a new kernel
    assert "restart the kernel" in message
    assert "FAQs.html#out-of-memory" in message
    assert isinstance(error.value.__cause__, RuntimeError)


def test_fitting_out_of_memory_explains_settings(
    tutorial_outlier_removed, tutorial_formatted_data, tutorial_model, tmp_path, monkeypatch
):
    _, coordinates, _ = tutorial_outlier_removed
    data, metadata = tutorial_formatted_data
    monkeypatch.setattr(kpms.fitting.keypoint_slds, "resample_model", raise_out_of_memory)

    with pytest.raises(MemoryError) as error:
        kpms.fit_model(
            tutorial_model, data, metadata, str(tmp_path), ar_only=True, num_iters=1
        )

    check_message(error, "AR-HMM fitting", f"{total_frames(coordinates):,} frames")


def test_applying_model_out_of_memory_explains_settings(
    tutorial_outlier_removed, tutorial_formatted_data, tutorial_model, monkeypatch
):
    project_dir, coordinates, _ = tutorial_outlier_removed
    data, metadata = tutorial_formatted_data
    config = kpms.load_config(project_dir)
    monkeypatch.setattr(kpms.fitting.keypoint_slds, "resample_model", raise_out_of_memory)

    with pytest.raises(MemoryError) as error:
        kpms.apply_model(
            tutorial_model, data, metadata, num_iters=1, save_results=False, **config
        )

    check_message(error, "applying the model", f"{total_frames(coordinates):,} frames")


def test_initialization_out_of_memory_explains_settings(
    tutorial_outlier_removed, tutorial_formatted_data, tutorial_pca, monkeypatch
):
    project_dir, coordinates, _ = tutorial_outlier_removed
    data, metadata = tutorial_formatted_data
    pca, _ = tutorial_pca
    config = kpms.load_config(project_dir)
    monkeypatch.setattr(kpms.fitting.keypoint_slds, "init_model", raise_out_of_memory)

    with pytest.raises(MemoryError) as error:
        kpms.init_model(data, metadata=metadata, pca=pca, **config)

    check_message(error, "model initialization", f"{total_frames(coordinates):,} frames")


def test_initialization_out_of_memory_without_metadata_describes_segments(
    tutorial_outlier_removed, tutorial_formatted_data, tutorial_pca, monkeypatch
):
    project_dir, _, _ = tutorial_outlier_removed
    data, _ = tutorial_formatted_data
    pca, _ = tutorial_pca
    config = kpms.load_config(project_dir)
    monkeypatch.setattr(kpms.fitting.keypoint_slds, "init_model", raise_out_of_memory)

    with pytest.raises(MemoryError) as error:
        kpms.init_model(data, pca=pca, **config)

    check_message(error, "model initialization", f"{data['mask'].shape[0]:,} segments")


def test_other_runtime_errors_pass_through(
    tutorial_formatted_data, tutorial_model, tmp_path, monkeypatch
):
    data, metadata = tutorial_formatted_data

    def raise_other(*args, **kwargs):
        raise RuntimeError("INTERNAL: some other XLA failure")

    monkeypatch.setattr(kpms.fitting.keypoint_slds, "resample_model", raise_other)

    with pytest.raises(RuntimeError, match="some other XLA failure"):
        kpms.fit_model(
            tutorial_model, data, metadata, str(tmp_path), ar_only=True, num_iters=1
        )
