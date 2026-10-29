"""kpms chooses how many batches jax-moseq's mixed_map splits the data into.

The parallel Kalman sampler (the default on a GPU) needs 6-9 times less
memory split into 32 batches, at the same speed. Each batch of the sequential
sampler (the default on CPU) costs time and saves little memory, so it runs
as one batch. A value the user set with `set_mixed_map_iters` is kept.

The tests patch `jax.default_backend` to stand in for a GPU, and each starts
from jax-moseq's default of 1, as in a new session.
"""

import jax
import pytest
from jax_moseq.utils import get_mixed_map_iters, set_mixed_map_iters

import keypoint_moseq as kpms
from keypoint_moseq import fitting


@pytest.fixture(autouse=True)
def new_session(monkeypatch):
    set_mixed_map_iters(1)
    monkeypatch.setattr(fitting, "_kpms_mixed_map_iters", None, raising=False)
    yield
    set_mixed_map_iters(1)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")


def test_parallel_sampler_on_a_gpu_uses_32_batches(gpu):
    parallel = fitting._set_parallel_flag(None)

    assert parallel
    assert get_mixed_map_iters() == 32


def test_sequential_sampler_on_cpu_uses_1_batch():
    parallel = fitting._set_parallel_flag(None)

    assert not parallel
    assert get_mixed_map_iters() == 1


def test_sequential_sampler_on_a_gpu_uses_1_batch(gpu):
    fitting._set_parallel_flag(None)
    fitting._set_parallel_flag(False)

    assert get_mixed_map_iters() == 1


def test_a_value_the_user_set_is_kept(gpu):
    set_mixed_map_iters(4)

    fitting._set_parallel_flag(None)

    assert get_mixed_map_iters() == 4


def test_expected_marginal_likelihoods_on_a_gpu_uses_32_batches(gpu):
    kpms.expected_marginal_likelihoods(checkpoint_paths=[])

    assert get_mixed_map_iters() == 32
