import os
import re
import numpy as np
import tqdm
import h5py
import jax
import jax.numpy as jnp
import warnings
from textwrap import fill
from datetime import datetime
from itertools import zip_longest

from keypoint_moseq.viz import plot_progress
from keypoint_moseq.io import save_hdf5, load_hdf5, extract_results, load_checkpoint
from keypoint_moseq.util import get_recording_lengths
from jax_moseq.models import allo_keypoint_slds, keypoint_slds
from jax_moseq.models.arhmm import stateseq_marginals, marginal_log_likelihood
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils import check_for_nans, device_put_as_scalar, unbatch
from jax_moseq.utils import get_mixed_map_gpus, get_mixed_map_iters


class StopResampling(Exception):
    pass


OUT_OF_MEMORY_FAQ = "https://keypoint-moseq.readthedocs.io/en/stable/FAQs.html#out-of-memory"


def _is_out_of_memory(error):
    """Whether `error` is the error XLA raises when an allocation fails."""
    return isinstance(error, RuntimeError) and "RESOURCE_EXHAUSTED" in str(error)


def _out_of_memory_error(
    error, stage, data, hypparams, parallel_message_passing=None, metadata=None
):
    """Explain an XLA out-of-memory error in terms of the data and settings.

    XLA's message names only the allocation that failed. Users need to know
    how much data they asked the device to hold and which settings reduce
    memory use, and those settings live in jax-moseq, not in the fitting call.

    Parameters
    ----------
    error: RuntimeError
        The `RESOURCE_EXHAUSTED` error raised by XLA.

    stage: str
        What was running, e.g. "AR-HMM fitting".

    data: dict or None
        Data passed to the model, with a `mask` of shape (n_segments,
        segment_length). None if unavailable.

    hypparams: dict
        Model hyperparameters; `ar_hypparams` supplies `latent_dim` and
        `num_states` when present.

    parallel_message_passing: bool, default=None
        The setting in effect, or None if it does not apply to `stage`.

    metadata: tuple (keys, bounds), default=None
        Segment metadata from `format_data`, used to count the recordings'
        frames. Without it the data are described by their segments.

    Returns
    -------
    MemoryError
    """
    device = jax.devices()[0]
    on_gpu = device.platform == "gpu"
    stats = device.memory_stats() if hasattr(device, "memory_stats") else None
    limit = (stats or {}).get("bytes_limit")
    requested = re.search(r"allocate\s+([\d.]+\s*[KMGT]?i?B)", str(error))
    ar_hypparams = (hypparams or {}).get("ar_hypparams", {})
    iters, gpus = get_mixed_map_iters(), get_mixed_map_gpus()

    lines = [f"Ran out of {'GPU ' if on_gpu else ''}memory during {stage}.", ""]
    device_text = f"{device.device_kind} ({device.platform}"
    device_text += f", {limit / 2**30:.1f} GiB available to JAX)" if limit else ")"
    if requested:
        lines.append(f"  XLA tried to allocate {requested.group(1)} on {device_text}.")
    else:
        lines.append(f"  Device: {device_text}.")
    if data is not None:
        n_segments, segment_length = np.asarray(data["mask"]).shape
        segments = f"{n_segments:,} segments of {segment_length:,} frames"
        if metadata is not None:
            n_frames = sum(get_recording_lengths(metadata).values())
            data_text = f"  Data: {n_frames:,} frames, split into {segments}"
        else:
            data_text = f"  Data: {segments}"
        if "latent_dim" in ar_hypparams:
            data_text += (
                f" (latent_dim={ar_hypparams['latent_dim']}, "
                f"num_states={ar_hypparams.get('num_states')})"
            )
        lines.append(data_text + ".")
    settings = f"mixed_map_iters={iters}, mixed_map_gpus={gpus}"
    if parallel_message_passing is not None:
        settings = f"parallel_message_passing={parallel_message_passing}, " + settings
    lines += [f"  Settings: {settings}.", "", "Ways to fit within memory:"]
    if on_gpu:
        lines.append(
            "  - If another notebook or script is using this GPU, shut it down. JAX "
            "reserves 75% of GPU memory when keypoint_moseq is imported, so a "
            "second process has little left."
        )
    lines.append(
        "  - Split the data into batches that are processed one after another: "
        "restart the kernel, then run "
        f"`from jax_moseq.utils import set_mixed_map_iters; set_mixed_map_iters({2 * iters})` "
        "before fitting. Memory use falls roughly in proportion to the number "
        "of batches, and fitting slows by about the same factor."
    )
    if parallel_message_passing:
        lines.append(
            "  - Pass `parallel_message_passing=False`, which uses 4-6 times less "
            "memory and is slower by a similar factor."
        )
    if on_gpu:
        lines.append(
            "  - Spread the work across several GPUs (Linux only): restart the "
            "kernel, then call `jax_moseq.utils.set_mixed_map_gpus` before "
            "fitting."
        )
    lines += [
        "  - Fit the model on a subset of the recordings, then use `apply_model` "
        "for the rest.",
        "",
        f"See {OUT_OF_MEMORY_FAQ}",
    ]
    return MemoryError("\n".join(lines))


def _wrapped_resample(
    resample_func, data, model, pbar=None, stage=None, metadata=None, **resample_options
):
    try:
        model = resample_func(data, **model, **resample_options)
    except KeyboardInterrupt:
        print("Early termination of fitting: user interruption")
        raise StopResampling()
    except RuntimeError as error:
        if not _is_out_of_memory(error):
            raise
        if pbar is not None:
            pbar.close()
        if stage is None:
            if resample_options.get("states_only"):
                stage = "applying the model"
            elif resample_options.get("ar_only"):
                stage = "AR-HMM fitting"
            else:
                stage = "full-model fitting"
        raise _out_of_memory_error(
            error,
            stage,
            data,
            model.get("hypparams"),
            resample_options.get("parallel_message_passing"),
            metadata,
        ) from error

    any_nans, nan_info, messages = check_for_nans(model)

    if any_nans:
        if pbar is not None:
            pbar.close()
        warning_text = ["\nEarly termination of fitting: NaNs encountered"]
        for msg in messages:
            warning_text.append("  - {}".format(msg))
        warning_text.append(
            "\nFor additional information, see https://keypoint-moseq.readthedocs.io/en/stable/FAQs.html#nans-during-fitting"
        )
        warnings.warn("\n".join(warning_text))
        raise StopResampling()

    return model


def _set_parallel_flag(parallel_message_passing):
    if parallel_message_passing == "force":
        parallel_message_passing = True
    elif parallel_message_passing is None:
        parallel_message_passing = jax.default_backend() != "cpu"
    elif parallel_message_passing and jax.default_backend() == "cpu":
        warnings.warn(
            fill(
                "Setting parallel_message_passing to True when JAX is CPU-bound can "
                "result in long jit times without speed increase for calculations. "
                '(To suppress this message, set parallel_message_passing="force")'
            )
        )
    return parallel_message_passing


def init_model(
    *args,
    location_aware=False,
    allo_hypparams=None,
    trans_hypparams=None,
    metadata=None,
    **kwargs,
):
    """Initialize a model. Wrapper for `jax_moseq.models.keypoint_slds.init_model`
    and `jax_moseq.models.allo_keypoint_slds.init_model`.

    Parameters
    ----------
    location_aware : bool, default=False
        If True, the model will be initialized using the location-aware version
        of the keypoint-SLDS model (`jax_moseq.models.allo_keypoint_slds`).

    allo_hypparams : dict, default=None
        Hyperparameters for the `allo_keypoint_slds` model. Keys left out take
        their default values.

    trans_hypparams : dict
        Transition hyperparameters, required when `location_aware=True`
        (usually passed with `**config()`).

    metadata : tuple (keys, bounds), default=None
        Segment metadata from :py:func:`keypoint_moseq.util.format_data`. Used only
        to report the number of frames if initialization runs out of memory.

    Returns
    -------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    Raises
    ------
    MemoryError
        If the device runs out of memory, with the data size and the
        settings that reduce memory use.
    """
    try:
        if location_aware:
            num_states = trans_hypparams["num_states"]
            allo_hypparams = {
                "alpha0_v": 10,
                "beta0_v": 0.1,
                "lambda0_v": 1,
                "alpha0_h": 10,
                "beta0_h": 0.1,
                "lambda0_h": 1,
                "num_states": num_states,
                **(allo_hypparams or {}),
            }

            return allo_keypoint_slds.init_model(
                *args,
                allo_hypparams=allo_hypparams,
                trans_hypparams=trans_hypparams,
                **kwargs,
            )
        else:
            return keypoint_slds.init_model(
                *args, trans_hypparams=trans_hypparams, **kwargs
            )
    except RuntimeError as error:
        if not _is_out_of_memory(error):
            raise
        data = kwargs.get("data", args[0] if args else None)
        hypparams = kwargs.get("hypparams") or {"ar_hypparams": kwargs.get("ar_hypparams", {})}
        raise _out_of_memory_error(
            error, "model initialization", data, hypparams, metadata=metadata
        ) from error


def _check_checkpoint_data(checkpoint_path, model_name, data, metadata):
    """Raise if an existing checkpoint was saved from data with other
    recording names, segment bounds or array shapes than `data` and
    `metadata`, listing each difference."""
    with h5py.File(checkpoint_path, "r") as f:
        if "data" not in f or "metadata" not in f:
            return
        saved_shapes = {k: tuple(f["data"][k].shape) for k in f["data"]}
    saved_keys, saved_bounds = load_hdf5(checkpoint_path, "metadata")
    keys, bounds = metadata

    def listed(names):
        names = sorted(names)
        more = f", and {len(names) - 5} more" if len(names) > 5 else ""
        return ", ".join(names[:5]) + more

    def segment(s):
        return "none" if s is None else f"{s[0]} [{s[1]}, {s[2]}]"

    sections = []
    saved_names, names = set(map(str, saved_keys)), set(map(str, keys))
    if saved_names != names:
        sections.append(
            f"  Recording names (checkpoint: {len(saved_names)}, data: {len(names)})"
        )
        if saved_names - names:
            sections.append(f"    only in checkpoint: {listed(saved_names - names)}")
        if names - saved_names:
            sections.append(f"    only in data: {listed(names - saved_names)}")

    saved_segments = [
        (str(k), *b) for k, b in zip(saved_keys, np.asarray(saved_bounds).tolist())
    ]
    segments = [(str(k), *b) for k, b in zip(keys, np.asarray(bounds).tolist())]
    if saved_segments != segments:
        i, (saved, current) = next(
            (i, pair)
            for i, pair in enumerate(zip_longest(saved_segments, segments))
            if pair[0] != pair[1]
        )
        sections += [
            f"  Segment bounds (checkpoint: {len(saved_segments)} segments, "
            f"data: {len(segments)} segments)",
            f"    first difference at segment {i}: checkpoint {segment(saved)}, "
            f"data {segment(current)}",
        ]

    shapes = {k: tuple(v.shape) for k, v in data.items() if k in saved_shapes}
    changed = [k for k in shapes if shapes[k] != saved_shapes[k]]
    if changed:
        width = max(len(k) for k in changed) + 1
        sections.append("  Array shapes")
        sections += [
            f"    {(k + ':').ljust(width)} checkpoint {saved_shapes[k]}, data {shapes[k]}"
            for k in changed
        ]

    if not sections:
        return
    wrap = lambda text: fill(text, break_long_words=False, break_on_hyphens=False)
    lines = [
        wrap(
            f"fit_model was given model_name '{model_name}', and a checkpoint for "
            f"that model already exists at {checkpoint_path}, saved from different "
            "data. Continuing a checkpoint is only valid on the data it was fit "
            "to, so fit_model checked the recording names, segment bounds and "
            "array shapes saved in the checkpoint against the data passed in, and "
            "found these differences:"
        ),
        "",
        *sections,
        "",
        wrap(
            f"To fit this data, delete {os.path.dirname(checkpoint_path)} and run "
            "again, or pass a different model_name. To label new recordings with a "
            "model that is already fit, use apply_model instead."
        ),
    ]
    raise ValueError("\n".join(lines))


def fit_model(
    model,
    data,
    metadata,
    project_dir=None,
    model_name=None,
    num_iters=50,
    start_iter=0,
    verbose=False,
    ar_only=False,
    parallel_message_passing=None,
    jitter=0.001,
    generate_progress_plots=True,
    save_every_n_iters=25,
    location_aware=False,
    fps=None,
    **kwargs,
):
    """Fit a model to data.

    This method optionally:
        - saves checkpoints of the model and data at regular intervals
        - plots of the model's progress during fitting (see
          :py:func:`keypoint_moseq.viz.plot_progress`)

    If a checkpoint for `model_name` already exists, it must have been saved
    from the same data (recording names, segment bounds and array shapes),
    and all model snapshots after `start_iter` are deleted.

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    data: dict
        Data for model fitting (see :py:func:`keypoint_moseq.util.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.util.format_data`).

    project_dir : str, default=None
        Project directory; required unless `save_every_n_iters=None`.

    model_name : str, default=None
        Name of the model. If None, the model is named using the current date
        and time.

    num_iters : int, default=50
        Index of the last iteration. Fitting runs `num_iters - start_iter + 1`
        resampling steps and saves the last as snapshot `num_iters`, so to
        continue a fit until its last snapshot is N iterations later, pass
        `num_iters=start_iter + N`.

    start_iter : int, default=0
        Index of the starting iteration, which is non-zero when continuing a
        previous fit.

    verbose : bool, default=False
        If True, print the model's progress during fitting.

    ar_only : bool, default=False
        If True, fit an AR-HMM model using the latent trajectory defined by
        `model['states']['x']` (see
        :py:func:`jax_moseq.models.arhmm.gibbs.resample_model`).
        Otherwise fit a full keypoint-SLDS model (see
        :py:func:`jax_moseq.models.keypoint_slds.gibbs.resample_model`)

    save_every_n_iters : int, default=25
        Save the current model every `save_every_n_iters`. To only save the
        final model, set `save_every_n_iters=-1`. To save nothing, set
        `save_every_n_iters=None`.

    generate_progress_plots : bool, default=True
        If True, generate plots of the model's progress during fitting. The latest
        plot is saved to `{project_dir}/{model_name}/fitting_progress.pdf`.

    parallel_message_passing : bool | string, default=None
        Use parallel implementation of Kalman sampling, which can be faster but
        has a significantly longer jit time. If None, will be set automatically
        based on the backend (True for GPU, False for CPU). A warning will be
        raised if `parallel_message_passing=True` and JAX is CPU-bound. Set to
        'force' to skip this check.

    jitter : float, default=0.001
        Amount to boost the diagonal of the dynamics covariance matrix when
        resampling pose trajectories. Increasing this value can help prevent
        NaNs during fitting.

    location_aware : bool, default=False
        If True, the model will be fit using the location-aware version of the
        keypoint-SLDS model (`jax_moseq.models.allo_keypoint_slds`).

    fps : float, default=None
        Frame rate of the recordings. If given, the progress plots show
        syllable durations in seconds; otherwise in frames. The notebook passes
        it with the rest of the config (`**config()`), which also sets
        `verbose`; other config entries are ignored.

    Returns
    -------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    model_name : str
        Name of the model.

    Raises
    ------
    ValueError
        If a checkpoint for `model_name` exists and was saved from different
        data.
    MemoryError
        If fitting runs out of memory. The message gives the data size and the
        settings that reduce memory use.
    """
    if generate_progress_plots and save_every_n_iters == 0:
        warnings.warn(
            fill(
                "The `generate_progress_plots` option requires that "
                "`save_every_n_iters` be greater than 0. Progress plots will "
                "not be generated."
            )
        )
        generate_progress_plots = False

    if model_name is None:
        model_name = str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

    if save_every_n_iters is not None:
        savedir = os.path.join(project_dir, model_name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        print(fill(f"Outputs will be saved to {savedir}"))

        checkpoint_path = os.path.join(savedir, "checkpoint.h5")
        if not os.path.exists(checkpoint_path):
            save_hdf5(
                checkpoint_path,
                {
                    "model_snapshots": {f"{start_iter}": model},
                    "metadata": metadata,
                    "data": data,
                },
            )
        else:
            _check_checkpoint_data(checkpoint_path, model_name, data, metadata)
            # delete model snapshots later than start_iter
            with h5py.File(checkpoint_path, "a") as f:
                for k in list(f["model_snapshots"].keys()):
                    if int(k) > start_iter:
                        del f["model_snapshots"][k]

    parallel_message_passing = _set_parallel_flag(parallel_message_passing)
    model = device_put_as_scalar(model)

    if location_aware:
        resample_func = allo_keypoint_slds.resample_model
    else:
        resample_func = keypoint_slds.resample_model

    with tqdm.trange(start_iter, num_iters + 1, ncols=72) as pbar:
        for iteration in pbar:
            try:
                model = _wrapped_resample(
                    resample_func,
                    data,
                    model,
                    pbar=pbar,
                    metadata=metadata,
                    ar_only=ar_only,
                    verbose=verbose,
                    jitter=jitter,
                    parallel_message_passing=parallel_message_passing,
                )
            except StopResampling:
                break

            if save_every_n_iters is not None and iteration > start_iter:
                if iteration == num_iters or (
                    save_every_n_iters > 0 and iteration % save_every_n_iters == 0
                ):
                    save_hdf5(
                        checkpoint_path,
                        model,
                        f"model_snapshots/{iteration}",
                        exist_ok=True,
                    )
                    if generate_progress_plots:
                        plot_progress(
                            model,
                            data,
                            checkpoint_path,
                            iteration,
                            project_dir,
                            model_name,
                            savefig=True,
                            fps=fps,
                        )

    return model, model_name


def apply_model(
    model,
    data,
    metadata,
    project_dir=None,
    model_name=None,
    num_iters=None,
    ar_only=False,
    save_results=True,
    verbose=False,
    results_path=None,
    parallel_message_passing=None,
    return_model=False,
    location_aware=False,
    overwrite=False,
    jitter=0.001,
    **kwargs,
):
    """Apply a model to new data.

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    data: dict
        Data for model fitting (see :py:func:`keypoint_moseq.util.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.util.format_data`).

    project_dir : str, default=None
        Path to the project directory. Required if `save_results=True` and
        `results_path=None`.

    model_name : str, default=None
        Name of the model. Required if `save_results=True` and
        `results_path=None`.

    num_iters : int, default=None
        Number of iterations to run the model, 500 if None. With
        `ar_only=True`, a single iteration runs, since each one draws the
        syllable labels independently given the same pose trajectories.

    ar_only : bool, default=False
        See :py:func:`keypoint_moseq.fitting.fit_model`.

    save_results : bool, default=True
        If True, the model outputs will be saved to disk (see
        :py:func:`keypoint_moseq.io.extract_results` for the output format).

    verbose : bool, default=False
        Whether to print progress updates.

    results_path : str, default=None
        Optional path for saving model outputs.

    parallel_message_passing : bool | string, default=None
        Use parallel implementation of Kalman sampling, which can be faster
        but has a significantly longer jit time. If None, will be set
        automatically based on the backend (True for GPU, False for CPU). A
        warning will be raised if `parallel_message_passing=True` and JAX is
        CPU-bound. Set to 'force' to skip this check.

    return_model : bool, default=False
        Whether to return the model after fitting.

    location_aware : bool, default=False
        If True, apply the location-aware version of the keypoint-SLDS model
        (`jax_moseq.models.allo_keypoint_slds`). Set this to the value the
        model was fit with.

    overwrite : bool, default=False
        If True, overwrite existing results for recordings that are
        already in the results file. This is required when applying a
        model to data that includes recordings from the original
        training set.

    jitter : float, default=0.001
        See :py:func:`keypoint_moseq.fitting.fit_model`.

    Returns
    -------
    results : dict
        Dictionary of model outputs for the recordings in `data` (for results
        format, see :py:func:`keypoint_moseq.io.extract_results`). When saved,
        they are added to the results file alongside any recordings already
        there.

    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed. Only returned if `return_model=True`.
    """
    parallel_message_passing = _set_parallel_flag(parallel_message_passing)
    data = jax.device_put(data)

    if save_results:
        if results_path is None:
            assert project_dir is not None and model_name is not None, fill(
                "The `save_results` option requires either a `results_path` "
                "or the `project_dir` and `model_name` arguments"
            )
            results_path = os.path.join(project_dir, model_name, "results.h5")

    model = init_model(
        data=data,
        metadata=metadata,
        seed=model["seed"],
        params=model["params"],
        hypparams=model["hypparams"],
        location_aware=location_aware,
        **kwargs,
    )

    if location_aware:
        resample_func = allo_keypoint_slds.resample_model
    else:
        resample_func = keypoint_slds.resample_model

    if num_iters is None:
        num_iters = 1 if ar_only else 500
    elif ar_only and num_iters > 1:
        warnings.warn(
            fill(
                "With `ar_only=True`, `apply_model` only resamples syllable "
                "labels, and each iteration redraws them independently from the "
                "same distribution given the same pose trajectories, so more "
                "iterations do not improve them. Stopping after 1 iteration "
                f"instead of running {num_iters}. Set `num_iters=1` to silence "
                "this warning."
            )
        )
        num_iters = 1

    with tqdm.trange(num_iters, desc="Applying model", ncols=72) as pbar:
        for iteration in pbar:
            try:
                model = _wrapped_resample(
                    resample_func,
                    data,
                    model,
                    pbar=pbar,
                    metadata=metadata,
                    ar_only=ar_only,
                    states_only=True,
                    verbose=verbose,
                    jitter=jitter,
                    parallel_message_passing=parallel_message_passing,
                )
            except StopResampling:
                break

    results = extract_results(
        model, metadata, project_dir, model_name, save_results, results_path,
        overwrite=overwrite,
    )

    if return_model:
        return results, model
    else:
        return results


def estimate_syllable_marginals(
    model,
    data,
    metadata,
    burn_in_iters=200,
    num_samples=100,
    steps_per_sample=10,
    return_samples=False,
    verbose=False,
    parallel_message_passing=None,
    location_aware=False,
    jitter=0.001,
    **kwargs,
):
    """Estimate marginal distributions over syllables.

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    data: dict
        Data for model fitting (see :py:func:`keypoint_moseq.util.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.util.format_data`).

    burn_in_iters : int, default=200
        Number of resampling iterations to run before collecting samples.

    num_samples : int, default=100
        Number of samples to collect for marginalization.

    steps_per_sample : int, default=10
        Number of resampling iterations to run between collecting samples.

    return_samples : bool, default=False
        Whether to store and return sampled syllable sequences. May require
        significant RAM.

    verbose : bool, default=False
        Whether to print progress updates.

    parallel_message_passing : bool | string, default=None
        Use parallel implementation of Kalman sampling, which can be faster
        but has a significantly longer jit time. If None, will be set
        automatically based on the backend (True for GPU, False for CPU). A
        warning will be raised if `parallel_message_passing=True` and JAX is
        CPU-bound. Set to 'force' to skip this check.

    location_aware : bool, default=False
        If True, use the location-aware version of the keypoint-SLDS model
        (`jax_moseq.models.allo_keypoint_slds`), whose marginals include
        heading and centroid dynamics. Set this to the value the model was
        fit with.

    jitter : float, default=0.001
        See :py:func:`keypoint_moseq.fitting.fit_model`.

    Returns
    -------
    marginal_estimates : dict
        Estimated marginal distributions over syllables in the form of a
        dictionary mapping recording names to arrays of shape
        ``(num_timepoints, num_syllables)``.

    samples : dict
        Sampled syllable sequences in the form of a dictionary mapping
        recording names to arrays of shape ``(num_timepoints, num_samples)``.
        Only returned if `return_samples=True`.
    """
    parallel_message_passing = _set_parallel_flag(parallel_message_passing)
    data = jax.device_put(data)

    model = init_model(
        data=data,
        metadata=metadata,
        seed=model["seed"],
        params=model["params"],
        hypparams=model["hypparams"],
        location_aware=location_aware,
        **kwargs,
    )

    num_syllables = model["hypparams"]["trans_hypparams"]["num_states"]
    marginal_estimates = np.zeros((*model["states"]["z"].shape, num_syllables))
    samples = []

    if location_aware:
        resample_func = allo_keypoint_slds.resample_model
        marginals_func = allo_keypoint_slds.stateseq_marginals
    else:
        resample_func = keypoint_slds.resample_model
        marginals_func = stateseq_marginals

    total_iters = burn_in_iters + num_samples * steps_per_sample
    with tqdm.trange(total_iters, desc="Applying model", ncols=72) as pbar:
        for iteration in pbar:
            try:
                model = _wrapped_resample(
                    resample_func,
                    data,
                    model,
                    pbar=pbar,
                    metadata=metadata,
                    stage="estimating syllable marginals",
                    states_only=True,
                    verbose=verbose,
                    jitter=jitter,
                    parallel_message_passing=parallel_message_passing,
                )
            except StopResampling:
                break

            if (
                iteration >= burn_in_iters
                and (iteration - burn_in_iters) % steps_per_sample == 0
            ):
                marginal_estimates += np.array(
                    marginals_func(
                        mask=data["mask"], **model["states"], **model["params"]
                    )
                )
                if return_samples:
                    samples.append(np.array(model["states"]["z"]))

    nlags = get_nlags(model["params"]["Ab"])
    keys, bounds = metadata
    # a segment no longer than nlags has no syllables of its own
    starts = np.minimum(bounds[:, 0] + nlags, bounds[:, 1])
    bounds = np.stack([starts, bounds[:, 1]], axis=1)
    marginal_estimates = unbatch(marginal_estimates / num_samples, keys, bounds)
    marginal_estimates = {
        k: np.pad(v[nlags:], ((nlags, 0), (0, 0)), mode="edge")
        for k, v in marginal_estimates.items()
    }
    if return_samples:
        samples = unbatch(np.moveaxis(samples, 0, 2), keys, bounds)
        samples = {
            k: np.pad(v[nlags:], ((nlags, 0), (0, 0)), mode="edge")
            for k, v in samples.items()
        }
        return marginal_estimates, samples
    else:
        return marginal_estimates


def update_hypparams(model_dict, **kwargs):
    """Edit the hyperparameters of a model.

    Hyperparameters are stored as a nested dictionary in the `hypparams` key of
    the model dictionary. This function allows the user to update the
    hyperparameters of a model by passing in keyword arguments with the same
    name as the hyperparameter. The hyperparameter will be updated if it is a
    scalar value.

    Parameters
    ----------
    model_dict : dict
        Model dictionary.

    kwargs : dict
        Keyword arguments mapping hyperparameter names to new values.

    Returns
    -------
    model_dict : dict
        Model dictionary with updated hyperparameters.
    """
    assert "hypparams" in model_dict, fill(
        "The inputted model/checkpoint does not contain any hyperparams"
    )

    not_updated = list(kwargs.keys())

    for hypparms_group in model_dict["hypparams"]:
        for k, v in kwargs.items():
            if k in model_dict["hypparams"][hypparms_group]:
                old_value = model_dict["hypparams"][hypparms_group][k]
                if not np.isscalar(old_value):
                    print(
                        fill(
                            f"{k} cannot be updated since it is not a scalar hyperparam"
                        )
                    )
                else:
                    if not isinstance(v, type(old_value)):
                        warnings.warn(
                            f"'{k}' with {type(v)} will be cast to {type(old_value)}"
                        )

                    model_dict["hypparams"][hypparms_group][k] = type(old_value)(v)
                    not_updated.remove(k)

    if len(not_updated) > 0:
        warnings.warn(fill(f"The following hypparams were not found {not_updated}"))

    return model_dict


def expected_marginal_likelihoods(
    project_dir=None, model_names=None, checkpoint_paths=None
):
    """Calculate the expected marginal likelihood score for each model.

    The score is calculated as follows, where theta^i denotes the
    autoregressive parameters and transition matrix for the i'th model,
    x^i denotes the latent trajectories for the i'th model, and the
    number of models is N:

    score(theta^i) = 1/(N-1) sum_{j ≠ i} P(x^j | theta^i)

    Parameters
    ----------
    project_dir : str
        Path to the project directory. Required if ``checkpoint_paths`` is None.

    model_names : list of str
        Names of the models to compare. Required if ``checkpoint_paths`` is None.

    checkpoint_paths : list of str
        Paths to the checkpoints to compare. Required if ``model_names`` and
        ``project_dir`` are None.

    Returns
    -------
    scores : numpy array
        Expected marginal likelihood score for each model.

    standard_errors : numpy array
        Standard error of the expected marginal likelihood score for each model.
    """
    if checkpoint_paths is None:
        assert project_dir is not None and model_names is not None, fill(
            "Must provide either `checkpoint_paths` or `project_dir` and `model_names`"
        )
        checkpoint_paths = [
            os.path.join(project_dir, model_name, "checkpoint.h5")
            for model_name in model_names
        ]

    xs, params = [], []
    for checkpoint_path in checkpoint_paths:
        model, data, _, _ = load_checkpoint(path=checkpoint_path)
        xs.append(model["states"]["x"])
        params.append(model["params"])

    num_models = len(xs)
    mlls = np.zeros((num_models, num_models))
    for i in tqdm.trange(num_models, ncols=72):
        for j in range(num_models):
            if i != j:
                mlls[i, j] = marginal_log_likelihood(
                    jnp.array(data["mask"]),
                    jnp.array(xs[j]),
                    jnp.array(params[i]["Ab"]),
                    jnp.array(params[i]["Q"]),
                    jnp.array(params[i]["pi"]),
                ).item()

    scores = mlls.sum(1) / (num_models - 1)
    variances = (mlls**2).sum(1) / (num_models - 1) - scores**2
    standard_errors = np.sqrt(variances / (num_models - 1))
    return scores, standard_errors
