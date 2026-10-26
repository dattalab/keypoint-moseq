"""The jitter given to fit_model, apply_model and estimate_syllable_marginals
is the jitter that pose trajectories are resampled with, for the
keypoint-SLDS and the location-aware model alike.

The tests replace jax-moseq's resampling functions with one that records
the jitter it would have run with and then stops, so nothing is resampled.
A call that passes no jitter records the replaced function's default.
"""

import inspect

import pytest

import keypoint_moseq as kpms

DEFAULT_JITTER = 0.001  # `fit_model`'s default


class Sampled(Exception):
    """Raised by the stand-in resampling function after its first call."""


@pytest.fixture
def sampled_jitters(monkeypatch):
    """Jitter of each call to jax-moseq's resampling functions."""
    jitters = []

    def recorder(resample_model):
        default = inspect.signature(resample_model).parameters["jitter"].default

        def record(*args, **kwargs):
            jitters.append(kwargs.get("jitter", default))
            raise Sampled

        return record

    for module in [kpms.fitting.keypoint_slds, kpms.fitting.allo_keypoint_slds]:
        monkeypatch.setattr(module, "resample_model", recorder(module.resample_model))
    return jitters


@pytest.fixture(scope="module")
def one_recording(tutorial_outlier_removed, tutorial_pca):
    """Config, data, metadata and an initialized keypoint-SLDS and
    location-aware model for one tutorial recording."""
    project_dir, coordinates, confidences = tutorial_outlier_removed
    pca, _ = tutorial_pca
    config = kpms.load_config(project_dir)
    name = sorted(coordinates)[0]
    data, metadata = kpms.format_data(
        {name: coordinates[name]}, {name: confidences[name]}, **config
    )
    models = {
        location_aware: kpms.init_model(
            data, metadata=metadata, pca=pca, location_aware=location_aware, **config
        )
        for location_aware in [False, True]
    }
    return config, data, metadata, models


def fit(model, data, metadata, config, tmp_path, **kwargs):
    kpms.fit_model(model, data, metadata, str(tmp_path), num_iters=1, **kwargs)


def apply(model, data, metadata, config, tmp_path, **kwargs):
    kpms.apply_model(model, data, metadata, num_iters=1, save_results=False, **config, **kwargs)


def estimate_marginals(model, data, metadata, config, tmp_path, **kwargs):
    kpms.estimate_syllable_marginals(
        model, data, metadata, burn_in_iters=1, num_samples=1, **config, **kwargs
    )


FUNCTIONS = pytest.mark.parametrize(
    "run", [fit, apply, estimate_marginals], ids=["fit_model", "apply_model", "marginals"]
)
MODELS = pytest.mark.parametrize(
    "location_aware", [False, True], ids=["keypoint_slds", "location_aware"]
)


@FUNCTIONS
@MODELS
def test_default_jitter(run, location_aware, one_recording, sampled_jitters, tmp_path):
    config, data, metadata, models = one_recording

    with pytest.raises(Sampled):
        run(models[location_aware], data, metadata, config, tmp_path, location_aware=location_aware)

    assert sampled_jitters == [DEFAULT_JITTER]


@FUNCTIONS
@MODELS
def test_jitter_reaches_the_sampler(run, location_aware, one_recording, sampled_jitters, tmp_path):
    config, data, metadata, models = one_recording

    with pytest.raises(Sampled):
        run(
            models[location_aware],
            data,
            metadata,
            config,
            tmp_path,
            location_aware=location_aware,
            jitter=0.05,
        )

    assert sampled_jitters == [0.05]
