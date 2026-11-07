"""A model saved in one floating-point precision works in a session that runs
JAX in the other.

keypoint-MoSeq 0.8.4 ran JAX in 64-bit mode, so its checkpoints hold float64
arrays; a session with 64-bit mode off runs in float32, and the reverse
holds for float32 checkpoints in a 64-bit session. Each test runs once per
session precision with a model or checkpoint in the other one. Integer arrays
other than the random seed (a uint32 PRNG key) follow the session's integer
width, and scalar hyperparameters stay Python numbers, which is what
`update_hypparams` edits.

The fit_model, apply_model and estimate_syllable_marginals tests replace
jax-moseq's resampling functions with one that records the dtypes of the
model it receives and then stops, so nothing is resampled.
"""

import jax
import numpy as np
import pytest

import keypoint_moseq as kpms


def in_precision(tree, x64):
    """Numpy copy of `tree` with floating and signed-integer arrays in 64-bit
    (x64=True) or 32-bit precision. Other arrays and Python scalars are kept."""
    floating, integer = (np.float64, np.int64) if x64 else (np.float32, np.int32)

    def cast(leaf):
        if not isinstance(leaf, (np.ndarray, jax.Array)):
            return leaf
        leaf = np.asarray(leaf)
        if np.issubdtype(leaf.dtype, np.floating):
            return leaf.astype(floating)
        if np.issubdtype(leaf.dtype, np.signedinteger):
            return leaf.astype(integer)
        return leaf

    return jax.tree_util.tree_map(cast, tree)


def wrong_dtypes(tree, x64):
    """Paths of arrays in `tree` whose dtype differs from the session's:
    float64/int64 with x64, float32/int32 without, uint32 for the seed."""
    floating, integer = (np.float64, np.int64) if x64 else (np.float32, np.int32)
    wrong = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf, (np.ndarray, jax.Array)):
            continue
        name = jax.tree_util.keystr(path)
        if name.startswith("['seed']"):
            expected = np.uint32
        elif np.issubdtype(leaf.dtype, np.floating):
            expected = floating
        elif np.issubdtype(leaf.dtype, np.integer):
            expected = integer
        else:
            continue
        if leaf.dtype != expected:
            wrong.append(f"{name}: {leaf.dtype}")
    return wrong


class Sampled(Exception):
    """Raised by the stand-in resampling function after its first call."""


@pytest.fixture
def sampled_models(monkeypatch):
    """Model passed to each call of jax-moseq's resampling functions."""
    models = []

    def record(data, **kwargs):
        keys = ["seed", "states", "params", "hypparams", "noise_prior"]
        models.append({key: kwargs[key] for key in keys})
        raise Sampled

    for module in [kpms.fitting.keypoint_slds, kpms.fitting.allo_keypoint_slds]:
        monkeypatch.setattr(module, "resample_model", record)
    return models


def test_load_checkpoint_matches_session_precision(
    session_x64, one_recording, tmp_path
):
    config, data, metadata, models = one_recording
    model = in_precision(models[False], not session_x64)
    saved_data = in_precision(data, not session_x64)
    path = str(tmp_path / "checkpoint.h5")
    kpms.save_hdf5(
        path,
        {"model_snapshots": {"0": model}, "metadata": metadata, "data": saved_data},
    )

    loaded, loaded_data, loaded_metadata, _ = kpms.load_checkpoint(path=path)

    assert wrong_dtypes(loaded, session_x64) == []
    assert wrong_dtypes(loaded_data, session_x64) == []
    for key in ["Ab", "Q", "Cd", "pi"]:
        np.testing.assert_allclose(
            loaded["params"][key], model["params"][key], rtol=1e-6, err_msg=key
        )
    np.testing.assert_array_equal(loaded["states"]["z"], model["states"]["z"])
    np.testing.assert_array_equal(loaded["seed"], model["seed"])
    np.testing.assert_array_equal(loaded_metadata[0], metadata[0])
    np.testing.assert_array_equal(loaded_metadata[1], metadata[1])

    kappa = loaded["hypparams"]["trans_hypparams"]["kappa"]
    assert type(kappa) is float
    updated = kpms.update_hypparams(loaded, kappa=kappa * 10)
    assert updated["hypparams"]["trans_hypparams"]["kappa"] == kappa * 10


def fit(model, data, metadata, config, tmp_path, **kwargs):
    kpms.fit_model(
        model, data, metadata, str(tmp_path), num_iters=1,
        generate_progress_plots=False, **kwargs,
    )


def apply(model, data, metadata, config, tmp_path, **kwargs):
    kpms.apply_model(
        model, data, metadata, num_iters=1, save_results=False, **config, **kwargs
    )


def estimate_marginals(model, data, metadata, config, tmp_path, **kwargs):
    kpms.estimate_syllable_marginals(
        model, data, metadata, burn_in_iters=1, num_samples=1, **config, **kwargs
    )


@pytest.mark.parametrize(
    "run", [fit, apply, estimate_marginals], ids=["fit_model", "apply_model", "marginals"]
)
@pytest.mark.parametrize(
    "location_aware", [False, True], ids=["keypoint_slds", "location_aware"]
)
def test_model_in_other_precision_runs_in_session_precision(
    run, location_aware, session_x64, one_recording, sampled_models, tmp_path
):
    config, data, metadata, models = one_recording
    # the data and config as a session in this precision makes them, and the
    # model in the other precision
    config = in_precision(config, session_x64)
    data = in_precision(data, session_x64)
    model = in_precision(models[location_aware], not session_x64)

    with pytest.raises(Sampled):
        run(model, data, metadata, config, tmp_path, location_aware=location_aware)

    (sampled,) = sampled_models
    assert wrong_dtypes(sampled, session_x64) == []
    assert type(sampled["hypparams"]["trans_hypparams"]["kappa"]) is float


CONVERSION_NOTICE = "the precision JAX is running in"


def conversion_notices(text):
    """Number of conversion notices in printed output (wrapped by `fill`)."""
    return " ".join(text.split()).count(CONVERSION_NOTICE)


def save_checkpoint(path, model, data, metadata):
    kpms.save_hdf5(
        str(path), {"model_snapshots": {"0": model}, "metadata": metadata, "data": data}
    )
    return str(path)


def test_conversion_notice_once_per_mismatched_model(
    session_x64, one_recording, sampled_models, capsys, tmp_path
):
    """Loading a checkpoint in the other precision prints the notice once,
    naming both precisions, and applying the loaded model prints nothing more."""
    config, data, metadata, models = one_recording
    path = save_checkpoint(
        tmp_path / "checkpoint.h5",
        in_precision(models[False], not session_x64),
        in_precision(data, not session_x64),
        metadata,
    )
    saved, session = ("32", "64") if session_x64 else ("64", "32")

    model, loaded_data, loaded_metadata, _ = kpms.load_checkpoint(path=path)
    printed = " ".join(capsys.readouterr().out.split())

    assert conversion_notices(printed) == 1
    assert (
        f"This model was saved in {saved}-bit precision and is converted to "
        f"{session}-bit, the precision JAX is running in." in printed
    )

    with pytest.raises(Sampled):
        kpms.apply_model(
            model, loaded_data, loaded_metadata, num_iters=1, save_results=False,
            **in_precision(config, session_x64),
        )
    assert conversion_notices(capsys.readouterr().out) == 0


def test_no_conversion_notice_for_a_matching_model(
    session_x64, one_recording, sampled_models, capsys, tmp_path
):
    config, data, metadata, models = one_recording
    path = save_checkpoint(
        tmp_path / "checkpoint.h5",
        in_precision(models[False], session_x64),
        in_precision(data, session_x64),
        metadata,
    )

    model, loaded_data, loaded_metadata, _ = kpms.load_checkpoint(path=path)
    with pytest.raises(Sampled):
        kpms.apply_model(
            model, loaded_data, loaded_metadata, num_iters=1, save_results=False,
            **in_precision(config, session_x64),
        )

    assert conversion_notices(capsys.readouterr().out) == 0


def test_conversion_notice_once_when_applying_a_mismatched_model(
    session_x64, one_recording, sampled_models, capsys
):
    """A model in the other precision handed straight to apply_model prints
    the notice once; data from format_data is converted without one."""
    config, data, metadata, models = one_recording

    with pytest.raises(Sampled):
        kpms.apply_model(
            in_precision(models[False], not session_x64), data, metadata,
            num_iters=1, save_results=False, **in_precision(config, session_x64),
        )

    assert conversion_notices(capsys.readouterr().out) == 1


def test_init_model_on_data_formatted_in_the_other_precision(
    session_x64, one_recording, tutorial_pca, capsys
):
    """format_data returns arrays in the session's precision. Data formatted
    in a session of the other precision runs in init_model, without a
    conversion notice, and the model is in the session's precision."""
    config, data, metadata, _ = one_recording
    pca, _ = tutorial_pca

    model = kpms.init_model(
        in_precision(data, not session_x64),
        metadata=metadata,
        pca=pca,
        **in_precision(config, session_x64),
    )

    assert wrong_dtypes(model, session_x64) == []
    assert conversion_notices(capsys.readouterr().out) == 0
