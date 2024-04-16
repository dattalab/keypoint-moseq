import os
import numpy as np
import tqdm
import h5py
import jax
import jax.numpy as jnp
import warnings
from textwrap import fill
from datetime import datetime

from keypoint_moseq.viz import plot_progress
from keypoint_moseq.io import save_hdf5, extract_results, load_checkpoint
from jax_moseq.models import allo_keypoint_slds, keypoint_slds
from jax_moseq.models.arhmm import stateseq_marginals, marginal_log_likelihood
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils import check_for_nans, device_put_as_scalar, unbatch


class StopResampling(Exception):
    pass


def _wrapped_resample(resample_func, data, model, pbar=None, **resample_options):
    try:
        model = resample_func(data, **model, **resample_options)
    except KeyboardInterrupt:
        print("Early termination of fitting: user interruption")
        raise StopResampling()

    any_nans, nan_info, messages = check_for_nans(model)

    if any_nans:
        if pbar is not None:
            pbar.close()
        warning_text = ["\nEarly termination of fitting: NaNs encountered"]
        for msg in messages:
            warning_text.append("  - {}".format(msg))
        warning_text.append(
            "\nFor additional information, see https://keypoint-moseq.readthedocs.io/en/latest/troubleshooting.html#nans-during-fitting"
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
    *args, location_aware=False, allo_hypparams=None, trans_hypparams=None, **kwargs
):
    """Initialize a model. Wrapper for `jax_moseq.models.keypoint_slds.init_model`
    and `jax_moseq.models.allo_keypoint_slds.init_model`.

    Parameters
    ----------
    location_aware : bool, default=False
        If True, the model will be initialized using the location-aware version
        of the keypoint-SLDS model (`jax_moseq.models.allo_keypoint_slds`).

    allo_hypparams : dict, default=None
        Hyperparameters for the `allo_keypoint_slds` model. If None, default
        hyperparameters will be used.

    Returns
    -------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.
    """
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
    **kwargs,
):
    """Fit a model to data.

    This method optionally:
        - saves checkpoints of the model and data at regular intervals
        - plots of the model's progress during fitting (see
          :py:func:`jax_moseq.viz.plot_progress`)

    Note that if a checkpoint file already exists, all model snapshots after
    `start_iter` will be deleted.

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    data: dict
        Data for model fitting (see :py:func:`keypoint_moseq.io.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.io.format_data`).

    project_dir : str, default=None
        Project directory; required if `save_every_n_iters>0`.

    model_name : str, default=None
        Name of the model. If None, the model is named using the current date
        and time.

    num_iters : int, default=50
        Number of Gibbs sampling iterations to run.

    start_iter : int, default=0
        Index of the starting iteration, which is non-zero when continuing a
        previous fit.

    verbose : bool, default=True
        If True, print the model's progress during fitting.

    ar_only : bool, default=False
        If True, fit an AR-HMM model using the latent trajectory defined by
        `model['states']['x']` (see
        :py:func:`jax_moseq.models.arhmm.resample_model`).
        Otherwise fit a full keypoint-SLDS model (see
        :py:func:`jax_moseq.models.keypoint_slds.resample_model`)

    save_every_n_iters : int, default=25
        Save the current model every `save_every_n_iters`. To only save the
        final model, set `save_every_n_iter=-1`. To save nothing, set
        `save_every_n_iters=None`.

    generate_progress_plots : bool, default=True
        If True, generate plots of the model's progress during fitting. Plots
        are saved to `{project_dir}/{model_name}/plots/`.

    parallel_message_passing : bool | string, default=None,
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

    Returns
    -------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    model_name : str
        Name of the model.
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
        else:  # delete model snapshots later than start_iter
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
                    save_hdf5(checkpoint_path, model, f"model_snapshots/{iteration}")
                    if generate_progress_plots:
                        plot_progress(
                            model,
                            data,
                            checkpoint_path,
                            iteration,
                            project_dir,
                            model_name,
                            savefig=True,
                        )

    return model, model_name


def apply_model(
    model,
    data,
    metadata,
    project_dir=None,
    model_name=None,
    num_iters=500,
    ar_only=False,
    save_results=True,
    verbose=False,
    results_path=None,
    parallel_message_passing=None,
    return_model=False,
    location_aware=False,
    **kwargs,
):
    """Apply a model to new data.

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    data: dict
        Data for model fitting (see :py:func:`keypoint_moseq.io.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.io.format_data`).

    project_dir : str, default=None
        Path to the project directory. Required if `save_results=True` and
        `results_path=None`.

    model_name : str, default=None
        Name of the model. Required if `save_results=True` and
        `results_path=None`.

    num_iters : int, default=500
        Number of iterations to run the model.

    ar_only : bool, default=False
        See :py:func:`keypoint_moseq.fitting.fit_model`.

    save_results : bool, default=True
        If True, the model outputs will be saved to disk (see
        :py:func:`keypoint_moseq.io.extract_results` for the output format).

    verbose : bool, default=False
        Whether to print progress updates.

    results_path : str, default=None
        Optional path for saving model outputs.

    parallel_message_passing : bool | string, default=None,
        Use parallel implementation of Kalman sampling, which can be faster
        but has a significantly longer jit time. If None, will be set
        automatically based on the backend (True for GPU, False for CPU). A
        warning will be raised if `parallel_message_passing=True` and JAX is
        CPU-bound. Set to 'force' to skip this check.

    return_model : bool, default=False
        Whether to return the model after fitting.

    location_aware : bool, default=False
        If True, the model will be fit using the location-aware version of the
        keypoint-SLDS model (`jax_moseq.models.allo_keypoint_slds`).

    Returns
    -------
    results : dict
        Dictionary of model outputs (for results format, see
        :py:func:`keypoint_moseq.io.extract_results`).

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

    with tqdm.trange(num_iters, desc="Applying model", ncols=72) as pbar:
        for iteration in pbar:
            try:
                model = _wrapped_resample(
                    resample_func,
                    data,
                    model,
                    pbar=pbar,
                    ar_only=ar_only,
                    states_only=True,
                    verbose=verbose,
                    parallel_message_passing=parallel_message_passing,
                )
            except StopResampling:
                break

    results = extract_results(
        model, metadata, project_dir, model_name, save_results, results_path
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
    **kwargs,
):
    """Estimate marginal distributions over syllables.

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, noise
        prior, and random seed.

    data: dict
        Data for model fitting (see :py:func:`keypoint_moseq.io.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.io.format_data`).

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

    parallel_message_passing : bool | string, default=None,
        Use parallel implementation of Kalman sampling, which can be faster
        but has a significantly longer jit time. If None, will be set
        automatically based on the backend (True for GPU, False for CPU). A
        warning will be raised if `parallel_message_passing=True` and JAX is
        CPU-bound. Set to 'force' to skip this check.

    location_aware : bool, default=False
        If True, the model will be fit using the location-aware version of the
        keypoint-SLDS model (`jax_moseq.models.allo_keypoint_slds`).

    Returns
    -------
    marginal_estimates : dict
        Estimated marginal distributions over syllables in the form of a
        dictionary mapping recoriding names to arrays of shape
        ``(num_timepoints, num_syllables)``.

    samples : dict
        Sampled syllable sequences in the form of a dictionary mapping
        recording names to arrays of shape ``(num_samples, num_timepoints)``.
        Only returned if `return_samples=True`.
    """
    parallel_message_passing = _set_parallel_flag(parallel_message_passing)
    data = jax.device_put(data)

    model = init_model(
        data=data,
        seed=model["seed"],
        params=model["params"],
        hypparams=model["hypparams"],
        **kwargs,
    )

    num_syllables = model["hypparams"]["trans_hypparams"]["num_states"]
    marginal_estimates = np.zeros((*model["states"]["z"].shape, num_syllables))
    samples = []

    if location_aware:
        resample_func = allo_keypoint_slds.resample_model
    else:
        resample_func = keypoint_slds.resample_model

    total_iters = burn_in_iters + num_samples * steps_per_sample
    with tqdm.trange(total_iters, desc="Applying model", ncols=72) as pbar:
        for iteration in pbar:
            try:
                model = _wrapped_resample(
                    resample_func,
                    data,
                    model,
                    pbar=pbar,
                    states_only=True,
                    verbose=verbose,
                    parallel_message_passing=parallel_message_passing,
                )
            except StopResampling:
                break

            if (
                iteration >= burn_in_iters
                and (iteration - burn_in_iters) % steps_per_sample == 0
            ):
                marginal_estimates += np.array(
                    stateseq_marginals(
                        model["states"]["x"], data["mask"], **model["params"]
                    )
                )
                if return_samples:
                    samples.append(np.array(model["states"]["z"]))

    nlags = get_nlags(model["params"]["Ab"])
    keys, bounds = metadata
    bounds = bounds + np.array([nlags, 0])
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

    The score is calculated as follows, where $\theta^{(i)}$ denotes the
    autoregressive parameters and transition matrix for the i'th model,
    $x^{(i)}$ denotes the latent trajectories for the i'th model, and the
    number of models iss $N$

    .. math::
        \text{score}(\theta^{(i)}) = \frac{1}{(N-1)} \sum_{j \neq i} P(x^{(j)} | \theta^{(i)})$

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
