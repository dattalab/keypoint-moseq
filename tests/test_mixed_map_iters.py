"""kpms chooses how many passes jax-moseq's mixed_map splits the segments into.

With the parallel Kalman sampler (the default on a GPU), each pass takes 5
segments, the fastest size, so N segments take ceil(N / 5) passes. On a GPU,
kpms compiles the pose step and the syllable step for the data before the
first resampling step and reads their memory; if the larger needs more than
80% of JAX's memory pool, the passes get smaller, down to 1 segment each.
The sequential sampler (the default on CPU) runs all segments in 1 pass. A
value the user set with `set_mixed_map_iters` is kept.

The tests patch `jax.default_backend` to stand in for a GPU, and each starts
from jax-moseq's default of 1, as in a new session.
"""

from math import ceil

import jax
import numpy as np
import pytest
from jax_moseq.utils import get_mixed_map_iters, set_mixed_map_iters

import keypoint_moseq as kpms
from keypoint_moseq import fitting


@pytest.fixture(autouse=True)
def new_session(monkeypatch):
    set_mixed_map_iters(1)
    monkeypatch.setattr(fitting, "_kpms_mixed_map_iters", None, raising=False)
    monkeypatch.setattr(fitting, "_checked_passes", {}, raising=False)
    yield
    set_mixed_map_iters(1)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")


def segments(n):
    return {"mask": np.ones((n, 10))}


@pytest.mark.parametrize("n_segments", [1, 4, 5, 6, 11, 70, 71, 1000])
def test_parallel_sampler_uses_passes_of_5_segments(gpu, n_segments):
    parallel = fitting._set_parallel_flag(None)
    fitting._set_mixed_map_iters(parallel, segments(n_segments))

    passes = get_mixed_map_iters()
    assert parallel
    assert passes == ceil(n_segments / 5)
    assert ceil(n_segments / passes) <= 5


def test_each_call_counts_the_segments_of_its_own_data(gpu):
    fitting._set_mixed_map_iters(True, segments(70))
    fitting._set_mixed_map_iters(True, segments(20))

    assert get_mixed_map_iters() == 4


def test_sequential_sampler_on_cpu_uses_1_pass():
    parallel = fitting._set_parallel_flag(None)
    fitting._set_mixed_map_iters(parallel, segments(70))

    assert not parallel
    assert get_mixed_map_iters() == 1


def test_sequential_sampler_on_a_gpu_uses_1_pass(gpu):
    fitting._set_mixed_map_iters(True, segments(70))
    fitting._set_mixed_map_iters(False, segments(70))

    assert get_mixed_map_iters() == 1


def test_a_value_the_user_set_is_kept(gpu):
    set_mixed_map_iters(4)

    fitting._set_mixed_map_iters(True, segments(70))

    assert get_mixed_map_iters() == 4


def unchanged_model(data, seed, states, params, hypparams, noise_prior, **kwargs):
    return dict(
        seed=seed, states=states, params=params, hypparams=hypparams,
        noise_prior=noise_prior,
    )


def test_apply_model_on_a_gpu_uses_passes_of_5_segments(
    gpu, one_recording, monkeypatch
):
    config, data, metadata, models = one_recording
    monkeypatch.setattr(fitting, "_gpu_memory_limit", lambda: None)
    monkeypatch.setattr(fitting.keypoint_slds, "resample_model", unchanged_model)

    kpms.apply_model(
        models[False], data, metadata, num_iters=1, save_results=False, **config
    )

    assert get_mixed_map_iters() == ceil(data["mask"].shape[0] / 5)


def test_expected_marginal_likelihoods_on_a_gpu_uses_passes_of_5_segments(
    gpu, monkeypatch
):
    model = {
        "states": {"x": np.zeros((12, 10, 2))},
        "params": {"Ab": np.zeros(1), "Q": np.zeros(1), "pi": np.zeros(1)},
    }
    monkeypatch.setattr(
        fitting,
        "_load_checkpoint",
        lambda path: (model, segments(12), None, None, False),
    )
    monkeypatch.setattr(fitting, "marginal_log_likelihood", lambda *args: np.float64(0.0))

    kpms.expected_marginal_likelihoods(checkpoint_paths=["a.h5", "b.h5"])

    assert get_mixed_map_iters() == 3


# The memory check, with each candidate's memory given by a function of the
# number of passes in place of compiling the pose step.


def linear_pose_bytes(n_segments, fixed, per_segment, compiled):
    def pose_bytes(passes):
        compiled.append(passes)
        return fixed + per_segment * ceil(n_segments / passes)

    return pose_bytes


def test_memory_check_keeps_5_segments_when_they_fit():
    compiled = []
    pose_bytes = linear_pose_bytes(70, 100, 50, compiled)

    passes = fitting._passes_within_memory(70, pose_bytes, 200, budget=10**6)

    assert passes == 14
    assert compiled == [14]


def test_memory_check_shrinks_passes_until_they_fit():
    compiled = []
    # 3 segments need 250 bytes and 4 need 300
    pose_bytes = linear_pose_bytes(70, 100, 50, compiled)

    passes = fitting._passes_within_memory(70, pose_bytes, 200, budget=280)

    assert passes == ceil(70 / 3)
    assert len(compiled) <= 3


def test_memory_check_stops_at_1_segment_per_pass():
    compiled = []
    pose_bytes = linear_pose_bytes(70, 100, 50, compiled)

    passes = fitting._passes_within_memory(70, pose_bytes, 200, budget=120)

    assert passes == 70
    assert len(compiled) <= 2


def test_memory_check_skips_the_pose_step_when_the_syllable_step_does_not_fit():
    compiled = []
    pose_bytes = linear_pose_bytes(70, 100, 50, compiled)

    passes = fitting._passes_within_memory(70, pose_bytes, 10**6, budget=10**5)

    assert passes == 70
    assert compiled == []


# The memory check on compiled programs, with JAX's memory pool given.


@pytest.mark.parametrize("location_aware", [False, True])
def test_compiled_memory_check_keeps_5_segments_on_a_large_gpu(
    gpu, one_recording, monkeypatch, location_aware
):
    _, data, _, models = one_recording
    monkeypatch.setattr(fitting, "_gpu_memory_limit", lambda: 2**40)
    n_segments = data["mask"].shape[0]

    fitting._set_mixed_map_iters(True, data, models[location_aware], location_aware)

    assert get_mixed_map_iters() == ceil(n_segments / 5)
    assert len(fitting._checked_passes) == 1


def test_compiled_memory_check_uses_1_segment_per_pass_on_a_small_gpu(
    gpu, one_recording, monkeypatch
):
    _, data, _, models = one_recording
    monkeypatch.setattr(fitting, "_gpu_memory_limit", lambda: 2**20)

    fitting._set_mixed_map_iters(True, data, models[False])

    assert get_mixed_map_iters() == data["mask"].shape[0]
