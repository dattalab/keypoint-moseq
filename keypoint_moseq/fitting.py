import joblib
import os
import numpy as np
import tqdm
import jax
import warnings
warnings.formatwarning = lambda msg, *a: str(msg)
from textwrap import fill
from datetime import datetime

from keypoint_moseq.viz import plot_progress
from keypoint_moseq.io import save_checkpoint, format_data, save_hdf5
from keypoint_moseq.util import get_durations, get_frequencies, pad_along_axis, reindex_by_frequency
from jax_moseq.models.keypoint_slds import estimate_coordinates, resample_model, init_model
from jax_moseq.utils import check_for_nans, batch, unbatch

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


def _wrapped_resample(data, model, **resample_options):
    try: 
        resample_model(data, **model, **resample_options)
    except KeyboardInterrupt: 
        print('Early termination of fitting: user interruption')
        raise StopResampling()

    any_nans, nan_info, messages = check_for_nans(model)
    
    if any_nans:
        print('Early termination of fitting: NaNs encountered')
        for msg in messages: print('  - {}'.format(msg))
        print('\nFor additional information, see https://keypoint-moseq.readthedocs.io/en/latest/troubleshooting.html#nans-during-fitting')
        raise StopResampling()
    
    return model


def fit_model(model,
              data,
              labels,
              start_iter=0,
              history=None,
              verbose=True,
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
        defined by ``model['states']['x']`` (see 
        :py:func:`jax_moseq.models.arhmm.resample_model`).
        Otherwise fit a full keypoint-SLDS model
        (see :py:func:`jax_moseq.models.keypoint_slds.resample_model`)

    name : str, default=None
        Name of the model. If None, rhe model is named using the current
        date and time.

    project_dir : str, default=None
        Project directory; required if ``save_every_n_iters>0`` or
        ``save_progress_figs=True``.

    save_data : bool, default=True
        If True, include the data in the checkpoint.

    save_states : bool, default=True
        If True, include the model's states in the checkpoint.

    save_history : bool, default=True
        If True, include the model's history in the checkpoint.

    save_every_n_iters : int, default=10
        Save a checkpoint every ``save_every_n_iters``

    history_every_n_iters : int, default=10
        Update the model's history every ``history_every_n_iters``. E.g.,
        if ``history_every_n_iters=10``, the history will contain snapshots 
        from iterations 0, 10, 20, etc. If ``history_every_n_iters=0``, no
        history is saved.

    states_in_history : bool, default=True
        If True, include the model's states in the history.

    plot_every_n_iters : int, default=10
        Plot the model's progress every ``plot_every_n_iters``. If 
        ``plot_every_n_iters=0``, no plots are generated.

    save_progress_figs : bool, default=True
        If True, save the progress plots to disk.
        
    Returns
    -------
    model : dict
        Model dictionary containing states, parameters, hyperparameters,
        noise prior, and random seed.

    history : dict
        Snapshots of the model over the course of fitting, represented as
        a dictionary with keys corresponding to the iteration number and
        values corresponding to copies of the ``model`` dictionary.

    name : str
        Name of the model.
    """
    if save_every_n_iters>0 or save_progress_figs:
        assert project_dir, fill(
            'To save checkpoints or progress plots during fitting, provide '
            'a ``project_dir``. Otherwise set ``save_every_n_iters=0`` and '
            '``save_progress_figs=False``')
        if name is None: 
            name = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        savedir = os.path.join(project_dir,name)
        if not os.path.exists(savedir): os.makedirs(savedir)
        print(fill(f'Outputs will be saved to {savedir}'))

    if history is None: history = {}

    for iteration in tqdm.trange(start_iter, num_iters+1):
        if history_every_n_iters>0 and (iteration%history_every_n_iters)==0:
            history = _update_history(history, iteration, model, 
                                     include_states=states_in_history)
            
        if plot_every_n_iters>0 and (iteration%plot_every_n_iters)==0:
            plot_progress(model, data, history, iteration, name=name, 
                          savefig=save_progress_figs, project_dir=project_dir)

        if save_every_n_iters>0 and (iteration%save_every_n_iters)==0:
            save_checkpoint(model, data, history, labels, iteration, name=name,
                            project_dir=project_dir,save_history=save_history, 
                            save_states=save_states, save_data=save_data)
            
        try: model = _wrapped_resample(
            data, model, ar_only=ar_only, verbose=verbose)
        except StopResampling: break


    return model, history, name
    
    
def resume_fitting(*, params, hypparams, labels, iteration, mask,
                   Y, conf, seed, noise_prior=None, states=None, **kwargs):
    """Resume fitting a model from a checkpoint."""
    
    data = jax.device_put({'Y':Y, 'mask':mask, 'conf':conf})    
    model = init_model(data, states, params, hypparams,
                       noise_prior, seed, **kwargs)

    return fit_model(model, data, labels, start_iter=iteration+1, **kwargs)


def apply_model(*, params, coordinates, confidences=None, num_iters=5, 
                use_saved_states=True, states=None, mask=None, labels=None, 
                noise_prior=None, ar_only=False, save_results=True, verbose=False,
                project_dir=None, name=None, results_path=None, **kwargs): 
    """
    Apply a model to data.

    There are two scenarios for applying this function:
        - The model is being applied to the same data it was fit to.
            This would useful if the the data was chunked into segments
            to allow parallelization during fitting. In this case,
            set ``use_saved_states=True`` and provide ``states``.
        - The model is being applied to new data. In this case,
            set ``use_saved_states=False``.

    Model outputs are saved to disk as a .h5 file, either at ``results_path``
    if it is specified, or at ``{project_dir}/{name}/results.h5`` if it is not.
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

    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays
        of shape (n_frames, n_bodyparts, 2)

    confidences: dict, default=None
        Dictionary mapping filenames to keypoint confidences as ndarrays
        of shape (n_frames, n_bodyparts)

    num_iters : int, default=5
        Number of iterations to run the model. We recommend 5 iterations
        for data the model has already been fit to, and a higher number
        (e.g. 10-20) for new data.

    use_saved_states : bool, default=True
        If True, and ``states`` is provided, the model will be initialized
        with the saved states. 

    states : dict, default=None
        Dictionary of saved states. 

    mask: ndarray, default=None
        Binary mask indicating areas of padding in ``states``
        (see :py:func:`keypoint_moseq.util.batch`).

    labels: list
        Row labels ``states``
        (see :py:func:`keypoint_moseq.util.batch`).

    noise_prior : ndarray, default=None
        Prior on the noise for each observation. Should be the same shape
        as ``states['s']``.

    ar_only : bool, default=False
        See :py:func:`keypoint_moseq.fitting.fit_model`.

    save_results : bool, default=True
        If True, the model outputs will be saved to disk.

    verbose : bool, default=False
        Whether to print progress updates.

    project_dir : str, default=None
        Path to the project directory. Required if ``save_results=True``
        and ``results_path=None``.

    name : str, default=None
        Name of the model. Required if ``save_results=True``
        and ``results_path=None``.

    results_path : str, default=None
        Optional path for saving model outputs.

    Returns
    -------
    results_dict : dict
        Dictionary of model outputs with the same structure as the
        results ``.h5`` file.
    """
    
    kwargs['seg_length'] = None # dont separate the data into segments
    
    data, new_labels = format_data(
        coordinates, confidences=confidences, **kwargs)
    session_names = [key for key,start,end in new_labels]

    if save_results:
        if results_path is None: 
            assert project_dir is not None and name is not None, fill(
                'The ``save_results`` option requires either a ``results_path`` '
                'or the ``project_dir`` and ``name`` arguments')
            results_path = os.path.join(project_dir,name,'results.h5')
     
    if use_saved_states:
        assert not (states is None or mask is None or labels is None), fill(
            'The ``use_saved_states`` option requires the additional '
            'arguments ``states``, ``mask`` and ``labels``')   
        
        if noise_prior is not None:
            noise_prior = batch(unbatch(noise_prior, labels), keys=session_names)[0]
            noise_prior = np.where(data['mask'][...,None], noise_prior, 1) 
            
        new_states = {}
        for k,v in jax.device_get(states).items():
            padding = mask.shape[1] - v.shape[1]
            v = pad_along_axis(v, (padding, 0), axis=1, value=1)
            v = batch(unbatch(v, labels), keys=session_names)[0]
            new_states[k] = v[:,padding:]
        states = new_states
        
    else: 
        states = None
        noise_prior = None
    
    model = init_model(data, states, params, noise_prior=noise_prior, verbose=verbose, **kwargs)
    
    if num_iters>0:
        for iteration in tqdm.trange(num_iters, desc='Applying model'):
            try: model = _wrapped_resample(
                    data, model, ar_only=ar_only, states_only=True, verbose=verbose)
            except StopResampling: break


    nlags = model['hypparams']['ar_hypparams']['nlags']
    states = jax.device_get(model['states'])                     
    estimated_coords = jax.device_get(estimate_coordinates(
        **model['states'], **model['params'], **data))
    z_reindexed = reindex_by_frequency(states['z'], np.array(data['mask']))

    
    results_dict = {
        session_name : {
            'syllables' : np.pad(states['z'][i], (nlags,0), mode='edge')[m>0],
            'syllables_reindexed' : np.pad(z_reindexed[i], (nlags,0), mode='edge')[m>0],
            'estimated_coordinates' : estimated_coords[i][m>0],
            'latent_state' : states['x'][i][m>0],
            'centroid' : states['v'][i][m>0],
            'heading' : states['h'][i][m>0],
        } for i,(m,session_name) in enumerate(zip(data['mask'],session_names))}
    
    if save_results: 
        save_hdf5(results_path, results_dict)
        print(fill(f'Saved results to {results_path}'))
        
    return results_dict


def revert(checkpoint, iteration):
    """
    Revert a checkpoint to an earlier iteration.

    The checkpoint will revert to latest iteration stored in the 
    fitting history that is less than or equal to ``iteration``.
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
    ``hypparams`` key of the model dictionary. This function
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