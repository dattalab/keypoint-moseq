import os
import numpy as np
import tqdm
import jax
import warnings
from textwrap import fill
from datetime import datetime
import shutil

from keypoint_moseq.viz import plot_progress
from keypoint_moseq.io import save_checkpoint, format_data, save_hdf5, save_kappa_scan_checkpoint
from keypoint_moseq.util import reindex_by_frequency
from jax_moseq.models.keypoint_slds import estimate_coordinates, resample_model, init_model
from jax_moseq.utils import check_for_nans, unbatch

from keypoint_moseq.util import get_durations

class StopResampling(Exception):
    pass

def _update_history(history, iteration, model, include_states=True): 
    
    model_snapshot = {
        'params': jax.device_get(model['params']),
        'seed': jax.device_get(model['seed'])}
    
    if include_states: 
        model_snapshot['states'] = jax.device_get(model['states'])
        
    history[iteration] = model_snapshot
    return history


def _wrapped_resample(data, model, pbar=None, **resample_options):
    try: 
        model = resample_model(data, **model, **resample_options)
    except KeyboardInterrupt: 
        print('Early termination of fitting: user interruption')
        raise StopResampling()

    any_nans, nan_info, messages = check_for_nans(model)
    
    if any_nans:
        if pbar is not None: pbar.close()
        warning_text = ['\nEarly termination of fitting: NaNs encountered']
        for msg in messages: warning_text.append('  - {}'.format(msg))
        warning_text.append('\nFor additional information, see https://keypoint-moseq.readthedocs.io/en/latest/troubleshooting.html#nans-during-fitting')
        warnings.warn('\n'.join(warning_text))
        raise StopResampling()
    
    return model


def _keep_duration_history(duration_history, current_model, data, window):
    """
    Maintain recent median durations and signal if the fitting has converged.
    window >= 2"""
    assert type(window) == int and window >= 2, fill(
        f"Early stopping window must be an integer greater than 1, but got {window}.")
    z = np.array(jax.device_get(current_model['states']['z']))
    mask = np.array(data['mask'])
    duration_history = np.array([*duration_history[-window + 1:], np.median(get_durations(z,mask))])
    return duration_history
    
def _check_converged(metric_history, tol, window):
    return (abs(np.diff(metric_history)) <= tol).sum() == window - 1


def fit_model(model,
              data,
              labels,
              start_iter=0,
              history=None,
              verbose=False,
              num_iters=50,
              ar_only=False,
              name=None,
              project_dir=None,
              save_data=True,
              save_states=True,
              save_history=True,
              save_every_n_iters=10,
              history_every_n_iters=10,
              states_in_history=True,
              plot_every_n_iters=10,  
              save_progress_figs=True,
              tol = None,
              **kwargs):

    """
    Fit a model to data.
    
    This method optionally:
        - saves checkpoints of the model and data at regular intervals
          (see :py:func:`jax_moseq.io.save_checkpoint`)
        - plots of the model's progress during fitting (see 
          :py:func:`jax_moseq.viz.plot_progress`)
        - saves a history of the model's states and parameters at 
          regular intervals

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, 
        noise prior, and random seed. 

    data: dict, labels: list of tuples
        See :py:func:`keypoint_moseq.io.format_data`

    start_iter : int, default=0
        Index of the starting iteration, which is non-zero when continuing
        a previous fit.

    history : dict, default=None
        Dictionary containing the history of the model's states and
        parameters, (see Returns for the format). If None, a new empty
        history is created.

    verbose : bool, default=True
        If True, print the model's progress during fitting.

    num_iters : int, default=50
        Number of Gibbs sampling iterations to run.

    ar_only : bool, default=False
        If True, fit an AR-HMM model using the latent trajectory
        defined by `model['states']['x']` (see 
        :py:func:`jax_moseq.models.arhmm.resample_model`).
        Otherwise fit a full keypoint-SLDS model
        (see :py:func:`jax_moseq.models.keypoint_slds.resample_model`)

    name : str, default=None
        Name of the model. If None, rhe model is named using the current
        date and time.

    project_dir : str, default=None
        Project directory; required if `save_every_n_iters>0` or
        `save_progress_figs=True`.

    save_data : bool, default=True
        If True, include the data in the checkpoint.

    save_states : bool, default=True
        If True, include the model's states in the checkpoint.

    save_history : bool, default=True
        If True, include the model's history in the checkpoint.

    save_every_n_iters : int, default=10
        Save a checkpoint every `save_every_n_iters`

    history_every_n_iters : int, default=10
        Update the model's history every `history_every_n_iters`. E.g.,
        if `history_every_n_iters=10`, the history will contain snapshots 
        from iterations 0, 10, 20, etc. If `history_every_n_iters=0`, no
        history is saved.

    states_in_history : bool, default=True
        If True, include the model's states in the history.

    plot_every_n_iters : int, default=10
        Plot the model's progress every `plot_every_n_iters`. If 
        `plot_every_n_iters=0`, no plots are generated.

    save_progress_figs : bool, default=True
        If True, save the progress plots to disk.

    tol : int, deafult=None
        Enable early stopping if the median syllable duration has not changed by more than
        `tol` in any of the last 4 resampling steps. If `None`, no check will be performed.
        If the early stopping criterion is reached, histories, plots, and model checkpionts
        will be updated unless `history_every_n_iters` or the other correspondsiding parameters
        are zero.
        
    Returns
    -------
    model : dict
        Model dictionary containing states, parameters, hyperparameters,
        noise prior, and random seed.

    history : dict
        Snapshots of the model over the course of fitting, represented as
        a dictionary with keys corresponding to the iteration number and
        values corresponding to copies of the `model` dictionary.

    name : str
        Name of the model.
    """
    if save_every_n_iters>0 or save_progress_figs:
        assert project_dir, fill(
            'To save checkpoints or progress plots during fitting, provide '
            'a `project_dir`. Otherwise set `save_every_n_iters=0` and '
            '`save_progress_figs=False`')
        if name is None: 
            name = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        savedir = os.path.join(project_dir,name)
        if not os.path.exists(savedir): os.makedirs(savedir)
        print(fill(f'Outputs will be saved to {savedir}'))

    if history is None: history = {}
    if tol is not None: med_durs_history = []

    with tqdm.trange(start_iter, num_iters+1) as pbar:
        for iteration in pbar:

            try: model = _wrapped_resample(
                data, model, pbar=pbar, ar_only=ar_only, verbose=verbose)        
            except StopResampling: break

            if tol is not None:
                med_durs_history = _keep_duration_history(med_durs_history, model, data, window = 5)
                early_stop = _check_converged(med_durs_history, tol, window = 5)
            else: early_stop = False

            if history_every_n_iters>0 and ((iteration%history_every_n_iters)==0 or early_stop):
                history = _update_history(history, iteration, model, 
                                        include_states=states_in_history)
                
            if plot_every_n_iters>0 and ((iteration%plot_every_n_iters)==0 or early_stop):
                plot_progress(model, data, history, iteration, name=name, 
                            savefig=save_progress_figs, project_dir=project_dir)

            if save_every_n_iters>0 and ((iteration%save_every_n_iters)==0 or early_stop):
                save_checkpoint(model, data, history, labels, iteration, name=name,
                                project_dir=project_dir,save_history=save_history, 
                                save_states=save_states, save_data=save_data)
                
            if early_stop:
                print('Early termination of fitting: converged (see `tol` parameter).')
                break

                
    return model, history, name
    
    
def resume_fitting(*, params, hypparams, labels, iteration, mask, num_iters,
                   Y, conf, seed, noise_prior=None, states=None, **kwargs):
    """Resume fitting a model from a checkpoint."""
    
    num_iters = num_iters + iteration
    data = jax.device_put({'Y':Y, 'mask':mask, 'conf':conf})    
    model = init_model(data, states, params, hypparams,
                       noise_prior, seed, **kwargs)

    return fit_model(model, data, labels, start_iter=iteration+1, 
                     num_iters=num_iters, **kwargs)


def extract_results(*, params, states, labels, save_results=True, 
                    project_dir=None, name=None, results_path=None, 
                    **kwargs): 
    """
    Extract model outputs and [optionally] save them to disk.

    Model outputs are saved to disk as a .h5 file, either at `results_path`
    if it is specified, or at `{project_dir}/{name}/results.h5` if it is not.
    If a .h5 file with the given path already exists, the outputs will be added
    to it. The results have the following structure::

        results.h5
        ├──session_name1
        │  ├──syllables             # model state sequence (z), shape=(T,)
        │  ├──syllables_reindexed   # states reindexed by frequency, shape=(T,)
        │  ├──estimated_coordinates # model predicted coordinates, shape=(T,n_keypoints,dim)
        │  ├──latent_state          # model latent state (x), shape=(T,latent_dim)
        │  ├──centroid              # model centroid (v), shape=(T,dim)
        │  └──heading               # model heading (h), shape=(T,)
        ⋮

    Parameters
    ----------
    params : dict
        Model parameters.

    states : dict, default=None
        Dictionary of saved states. 

    labels: list
        Row labels for `states`
        (see :py:func:`keypoint_moseq.util.batch`).

    save_results : bool, default=True
        If True, the model outputs will be saved to disk.
        
    project_dir : str, default=None
        Path to the project directory. Required if `save_results=True`
        and `results_path=None`.

    name : str, default=None
        Name of the model. Required if `save_results=True`
        and `results_path=None`.

    results_path : str, default=None
        Optional path for saving model outputs.

    Returns
    -------
    results_dict : dict
        Dictionary of model outputs with the same structure as the
        results `.h5` file.
    """
    if save_results:
        if results_path is None: 
            assert project_dir is not None and name is not None, fill(
                'The `save_results` option requires either a `results_path` '
                'or the `project_dir` and `name` arguments')
            results_path = os.path.join(project_dir,name,'results.h5')

    # estimate keypoint coords from latent state, centroid and heading
    estimated_coords = jax.device_get(estimate_coordinates(**states, **params))
    states = jax.device_get(states)
    
    # extract syllables; repeat first syllable an extra `nlags` times
    nlags = states['x'].shape[1] - states['z'].shape[1]
    lagged_labels = [(key,start+nlags,end) for key,start,end in labels]
    syllables = unbatch(states['z'], lagged_labels)
    syllables = {k: np.pad(z[nlags:], (nlags,0), mode='edge') for k,z in syllables.items()}
    syllables_reindexed = reindex_by_frequency(syllables)
    
    # extract estimated coords, latent state, centroid, and heading
    estimated_coords = unbatch(estimated_coords, labels)
    latent_state = unbatch(states['x'], labels)
    centroid = unbatch(states['v'], labels)
    heading = unbatch(states['h'], labels)

    results_dict = {
        session_name : {
            'syllables' : syllables[session_name],
            'syllables_reindexed' : syllables_reindexed[session_name],
            'estimated_coordinates' : estimated_coords[session_name],
            'latent_state' : latent_state[session_name],
            'centroid' : centroid[session_name],
            'heading' : heading[session_name]
        } for session_name in syllables.keys()}
    
    if save_results: 
        save_hdf5(results_path, results_dict)
        print(fill(f'Saved results to {results_path}'))
        
    return results_dict


def apply_model(*, params, coordinates, confidences=None, num_iters=20, 
                ar_only=False, save_results=True, verbose=False,
                project_dir=None, name=None, results_path=None, **kwargs): 
    """
    Apply a model to new data.

    Parameters
    ----------
    params : dict
        Model parameters.

    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays
        of shape (n_frames, n_bodyparts, 2)

    confidences: dict, default=None
        Dictionary mapping filenames to keypoint confidences as ndarrays
        of shape (n_frames, n_bodyparts)

    num_iters : int, default=20
        Number of iterations to run the model. 

    ar_only : bool, default=False
        See :py:func:`keypoint_moseq.fitting.fit_model`.

    save_results : bool, default=True
        If True, the model outputs will be saved to disk (see 
        :py:func:`keypoint_moseq.fitting.extract_results` for
        the output format).

    verbose : bool, default=False
        Whether to print progress updates.

    project_dir : str, default=None
        Path to the project directory. Required if `save_results=True`
        and `results_path=None`.

    name : str, default=None
        Name of the model. Required if `save_results=True`
        and `results_path=None`.

    results_path : str, default=None
        Optional path for saving model outputs.

    Returns
    -------
    results_dict : dict
        Dictionary of model outputs (for results format, see
        :py:func:`keypoint_moseq.fitting.extract_results`).
    """
    if save_results:
        if results_path is None: 
            assert project_dir is not None and name is not None, fill(
                'The `save_results` option requires either a `results_path` '
                'or the `project_dir` and `name` arguments')
            results_path = os.path.join(project_dir,name,'results.h5')
     
    data, labels = format_data(coordinates, confidences=confidences, **kwargs)
    model = init_model(data=data, params=params, verbose=verbose, **kwargs)

    with tqdm.trange(num_iters, desc='Applying model') as pbar:
        for iteration in pbar:
            try: model = _wrapped_resample(
                    data, model, pbar=pbar, ar_only=ar_only, 
                    states_only=True, verbose=verbose)
            except StopResampling: break

    return extract_results(
        **model, labels=labels, save_results=save_results, name=name,
        project_dir=project_dir, results_path=results_path)


def revert(checkpoint, iteration):
    """
    Revert a checkpoint to an earlier iteration.

    The checkpoint will revert to latest iteration stored in the 
    fitting history that is less than or equal to `iteration`.
    The model parameters, seed, and iteration number will be updated
    accordingly. The fitting history will be truncated to remove
    all iterations after the reverted iteration.

    Parameters
    ----------
    checkpoint : dict
        See :py:func:`keypoint_moseq.io.save_checkpoint`.

    iteration : int
        Iteration to revert to.

    Returns
    -------
    checkpoint : dict
        Reverted checkpoint.
    """
    assert len(checkpoint['history'])>0, fill(
        'No history was saved during fitting')
    
    use_iter = max([i for i in checkpoint['history'] if i <= iteration])
    print(f'Reverting to iteration {use_iter}')
    
    model_snapshot =  checkpoint['history'][use_iter]
    checkpoint['params'] = model_snapshot['params']
    checkpoint['seed'] = model_snapshot['seed']
    checkpoint['iteration'] = use_iter
    
    if 'states' in model_snapshot: 
        checkpoint['states'] = model_snapshot['states']
    else: checkpoint['states'] = None
        
    for i in list(checkpoint['history'].keys()):
        if i > use_iter: del checkpoint['history'][i]

    return checkpoint
    
    
def update_hypparams(model_dict, **kwargs):
    """
    Edit the hyperparameters of a model.

    Hyperparameters are stored as a nested dictionary in the
    `hypparams` key of the model dictionary. This function
    allows the user to update the hyperparameters of a model
    by passing in keyword arguments with the same name as the
    hyperparameter. The hyperparameter will be updated if it
    is a scalar value.

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
    assert 'hypparams' in model_dict, fill(
        'The inputted model/checkpoint does not contain any hyperparams')
    
    not_updated = list(kwargs.keys())
    
    for hypparms_group in model_dict['hypparams']:
        for k,v in kwargs.items():
    
            if k in model_dict['hypparams'][hypparms_group]:
                
                old_value = model_dict['hypparams'][hypparms_group][k]
                
                if not np.isscalar(old_value): print(fill(
                    f'{k} cannot be updated since it is not a scalar hyperparam'))
                 
                else:
                    if not isinstance(v, type(old_value)): warnings.warn(fill(
                        f'{v} has type {type(v)} which differs from the current '
                        f'value of {k} which has type {type(old_value)}. {v} will '
                        f'will be cast to {type(old_value)}'))
                                     
                    model_dict['hypparams'][hypparms_group][k] = type(old_value)(v)
                    not_updated.remove(k)

    if len(not_updated)>0: warnings.warn(fill(
        f'The following hypparams were not found {not_updated}'))

    return model_dict


def kappa_scan(
    data,
    labels,
    scan_name = None,
    model_name_fmt = None,
    save_models = 'all',
    save_durations = True,
    save_progress_figs = True,
    project_dir = None,
    init_model_kwargs = {},
    fit_model_kwargs = {},
    plot_model_progress = None,
    scan_verbose = False,
    warning_convergence_tol = 2,
    **config,
    ):
    """
    Fit models on a range of kappa values to find desired median syllable duration.
    
    See also the configuration parameter `kappa_scan` for specifications on the scan to be
    performed: `'min_kappa'`, `'max_kappa'`, `'n_iter'`, and `'target_duration'`.

    Parameters
    ----------
    data: dict, labels: list of tuples
        See :py:func:`keypoint_moseq.io.format_data`
    
    scan_name: str, default=None
        Name of the scan. If None, the scan is named using the current
        date and time.

    model_name_fmt: str, default=None
        Format string for the name of each trained model. If None, the models are named
        using the date and time at the beginning of their fitting procedure. The keys
        available in the format string are `scan_name` (as given by the `scan_name parameter),
        `scan_iter` (the current integer iteration number of the scan), and `current_time` (
        string-formatted datetime at the start of the model fit).

    save_models: str or bool, default='all'
        Policy for saving results of individual model fits. If `'all``, then output files
        of :py:func:`keypoint_moseq.fit_model` for all iterations will be retained.
        If `'best'`, output files for only the iteration resulting in medan duration closest to
        the target value. If `False`, no model data will be saved, except for the median duration
        histories in a summary file for the scan (see `save_durations`), and `save_every_n_iters` 
        will be overridden to 0 in `fit_model_kwargs`. Setting to `True` is equivalent to `'all'`.

    save_durations: bool, default=True
        Whether to save summary file mapping kappas to median durations and the median duration
        histories from training.

    save_progess_figs:
        Passed on to parameter `save_progeess_figs` of :py:func:`keypoint_moseq.fit_model`, and
        determines whether to save model progress figures generated at the end of each model fit (
        see parameter `plot_model_progress`).

    project_dir : str, default=None
        Project directory; required if any of `save_models`, `save_durations`, or `plot_model_progress`
        are `True`, or if `save_every_n_iters>0` or `save_progress_figs=True`in parameter
        `fit_model_kwargs`.

    init_model_kwargs : dict, default={}
        Keyword arguments for :py:func:`keypoint_moseq.init_model` when called at the start of
        each scan iteration.

    fit_model_kwargs : dict, default={}
        Keyword arguments for :py:func:`keypoint_moseq.fit_model` when called at during each
        each scan iteration. Entries may be overwritten for `'name'` (see parameter `model_name_fmt`),
        `'project_dir'` (see parameter `project_dir`) ,  `'plot_every_n_iters'` (see paramete
        `plot_model_progress`), `'save_progress_figs'` (see parameter `save_progress_figs`),
        and `'save_every_n_iters'` (see parameter `save_models`).

    plot_model_progress : int, str, bool, default=None
        Policy for plotting model progeess. May be an integer to be passed on to
        :py:func:`keypoint_moseq.fit_model` (overriding any value in `fit_model_kwargs`). Otherwise, if
        `'each'`, one model progress plot will be generated at the end of each fitting process, or if `False`
        then no model progreess plots will be generated. If `None`, the value in `fit_model_kwargs` or the
        :py:func:`keypoint_moseq.fit_model` take precedence. Plots are saved according to `save_progess_figs`.

    scan_verbose: bool, default=False
        Verbosity level for scan information. Does not interaction with config option `verbose` or parameter
        `verbose` of :py:func:`keypoint_moseq.fit_model`.

    warning_convergence_tol: int, default=2
        After each fittting process, a warning will be issued if median durations changed more than
        `warning_convergence_tol` over any of its last 4 resampling iterations.

    config: dict, default={}
        Project configuration to be passed as keyword arguments to :py:func:`keypoint_moseq.init_model`, 
        and containing an entry `'kappa_scan'`: a dictionary with keys 'min_kappa'`, `'max_kappa'`, 
        `'n_iter'`, and `'target_duration'` specifying the scan to be performed.

    Returns
    -------
    best: dict
        Dictionary containing keys `'i'`, `'kappa'`, holding the iteration that yielded the closest
        median duration to the target duration and corresponding value of kappa, respectively.
    
    kappas: np.ndarray
        Values of the hyperparameter `kappa` at each iteration of the scan.

    median_durations: np.ndarray
        Resulting median syllable durations at each iteration of the scan.

    med_dur_histories: List[np.ndarray]
        History of model durations over the fitting process for each trained model to verify convergence.
        See parameters `fit_model_kwargs` of this function and `history_every_n_iter` of
        :py:func:`keypoint_moseq.fit_model`.
    """

    # --- Check arguments

    if scan_name is None:
        scan_name = f"scan-{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    if model_name_fmt is None:
        model_name_fmt = ("{scan_name}-iter{scan_iter}-{current_time}")

    if save_models is True: save_models = 'all'
    assert (save_models == 'all' or save_models == 'best' or 
            save_models is False), fill(
        "Parameter `save_models` must be one of 'all', 'best', or a boolean.")
    
    plot_every_n_fit_iters = 0
    if isinstance(plot_model_progress, int):
        plot_every_n_fit_iters = plot_model_progress
    else:
        assert plot_model_progress is False or plot_model_progress == 'each', fill(
            "Parameter `plot_model_progress` must be None, False, 'each', or an "
            "integer to be passsed on to `plot_every_n_iters`.")
    
    if save_durations or save_models is not False:
        if project_dir is None:
            assert project_dir, fill(
                'To save durations or models from of a kappa scan, provide '
                'a `project_dir`. Otherwise set `save_models=False`, '
                '`save_durations=False`, and `plot_model_progress=False`')


    # --- Set up variables for iteration

    scan_config = config['kappa_scan']
    kappas = np.logspace(
        np.log10(scan_config['min_kappa']),
        np.log10(scan_config['max_kappa']),
        scan_config['n_iter'])
    median_durations = []
    med_dur_histories = []
    model_names = []
    best_i = None

    

    # --- Perform kappa scan

    for i_kappa, kappa in enumerate(kappas):

        # Set up fit parameters
        model_name = model_name_fmt.format(
            scan_name = scan_name,
            scan_iter = i_kappa,
            current_time = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S')))
        fit_kwargs = {**fit_model_kwargs, **dict(
            name = model_name,
            project_dir = project_dir,
            plot_every_n_iters = plot_every_n_fit_iters,
            save_progress_figs = save_progress_figs,
            save_every_n_iters = ('will-be-popped' if save_models is not False else 0)
        )}
        if plot_model_progress is None: fit_kwargs.pop('plot_every_n_iters')
        if save_models == 'all' or save_models == 'best': fit_kwargs.pop('save_every_n_iters')
        

        # Initialize a model fit with the desired kappa
        if scan_verbose: print(f"Initializing model with kappa={kappa}.")
        model = init_model(data, **{
            **config,
            **init_model_kwargs,
            **dict(
                trans_hypparams = {
                    **config['trans_hypparams'],
                    'kappa': kappa
                }
            )})

        # Fit the initialized model
        if scan_verbose: print(f"Fitting said model with kappa={kappa}.")
        model, history, model_name = fit_model(
            model, data, labels,
            **fit_kwargs)
        model_names.append(model_name)
        
        # Calculate resulting median duration
        if scan_verbose: print(f"Calculating duration distribution.")
        z = np.array(model['states']['z'])
        mask = np.array(data['mask'])
        median_duration = np.median(get_durations(z,mask))
        median_durations.append(median_duration)

        # calculate durations through history to check for convergence
        history_iters = np.array(sorted(history.keys()))
        past_stateseqs = [history[i]['states']['z'] 
                          for i in history_iters 
                          if 'states' in history[i]]
        med_dur_history = [np.median(get_durations(z,mask)) for z in past_stateseqs]
        med_dur_histories.append(med_dur_history)


        # Keep track of best model and delete unecessary model directories

        def _delete_model_with_name(name):
            if project_dir is None: return
            path = os.path.join(project_dir, name)
            if os.path.exists(path): shutil.rmtree(path)

        if best_i is None or (
            abs(median_duration - scan_config['target_duration']) < 
            abs(median_durations[best_i] - scan_config['target_duration'])):
            if save_models == 'best' and best_i is not None:
                # Delete previous best model
                _delete_model_with_name(model_name)
            best_i = i_kappa
        elif save_models == 'best':
            # Delete this model, it was not better than the previous best
            _delete_model_with_name(model_name)


        # Plot and save results of fit

        if plot_model_progress == 'each':
            plot_progress(model, data, history, i_kappa, name=model_name, 
                savefig=save_progress_figs, project_dir=project_dir)

        if save_durations:
            save_kappa_scan_checkpoint(
                kappas[:i_kappa+1], median_durations,
                i_kappa, scan_config['target_duration'], kappas[best_i],
                med_dur_histories,
                model_names = (
                    model_names
                    if (save_models == 'all') else None),
                best_model_name = (
                    model_names[best_i]
                    if (save_models == 'best') else None),
                name = scan_name, project_dir = project_dir)
        
        if warning_convergence_tol is not None and not _check_converged(
            med_dur_history[-5:], warning_convergence_tol, window = 5):
            warnings.warn(fill(
                f"Fitting with kappa={kappa} did not converge. Increase `num_iters` "
                 "in `fit_model_kwargs`, or to silence this warning increase "
                 "`warning_convergence_tol`."))

    if ((scan_config['target_duration'] < np.min(median_durations)) or
        (scan_config['target_duration'] > np.max(median_durations))):
        warnings.warn(fill(
            "Target duratiuon did not lie within durations achieved by kappa_scan."))
    
    return (
        dict(i = best_i, kappa = kappas[best_i]),
        kappas, median_durations, med_dur_histories)

        





    
