from jax.tree_util import tree_map
import jax.numpy as jnp
import jax
import numpy as np
import warnings
import h5py
import joblib
import tqdm
import yaml
import os
import cv2
import re
import pandas as pd
from datetime import datetime
from textwrap import fill
from vidio.read import OpenCVReader
import matplotlib.pyplot as plt
import tabulate


from jax_moseq.utils import batch
from keypoint_moseq.util import (
    reindex_by_bodyparts, 
    list_files_with_exts, 
    interpolate_keypoints,
)


def _build_yaml(sections, comments):
    text_blocks = []
    for title,data in sections:
        centered_title = f' {title} '.center(50, '=')
        text_blocks.append(f"\n\n{'#'}{centered_title}{'#'}")
        for key,value in data.items():
            text = yaml.dump({key:value}).strip('\n')
            if key in comments: text = f"\n{'#'} {comments[key]}\n{text}"
            text_blocks.append(text)
    return '\n'.join(text_blocks)
        

def generate_config(project_dir, **kwargs):
    """
    Generate a `config.yml` file with project settings. Default 
    settings will be used unless overriden by a keyword argument.
    
    Parameters
    ----------
    project_dir: str 
        A file `config.yml` will be generated in this directory.
    
    kwargs
        Custom project settings.  
    """
    
    def _update_dict(new, original):
        return {k:new[k] if k in new else v for k,v in original.items()} 
    
    hypperams = _update_dict(kwargs, {
        'error_estimator': {'slope':-.5, 'intercept':.25},
        'obs_hypparams': {'sigmasq_0':0.1, 'sigmasq_C':.1, 'nu_sigma':1e5, 'nu_s':5},
        'ar_hypparams': {'latent_dim': 10, 'nlags': 3, 'S_0_scale': 0.01, 'K_0_scale': 10.0},
        'trans_hypparams': {'num_states': 100, 'gamma': 1e3, 'alpha': 5.7, 'kappa': 1e6},
        'cen_hypparams': {'sigmasq_loc': 0.5}})
    
    hypperams = {k : _update_dict(kwargs, v) for k,v in hypperams.items()}

    anatomy = _update_dict(kwargs, {
        'bodyparts': ['BODYPART1','BODYPART2','BODYPART3'],
        'use_bodyparts': ['BODYPART1','BODYPART2','BODYPART3'],
        'skeleton': [['BODYPART1','BODYPART2'], ['BODYPART2','BODYPART3']],
        'anterior_bodyparts': ['BODYPART1'],
        'posterior_bodyparts': ['BODYPART3']})
        
    other = _update_dict(kwargs, {
        'session_name_suffix': '',
        'verbose':False,
        'conf_pseudocount': 1e-3,
        'video_dir': '',
        'keypoint_colormap': 'autumn',
        'whiten': True,
        'fix_heading': False,
        'seg_length': 10000 })
       
    fitting = _update_dict(kwargs, {
        'added_noise_level': 0.1,
        'PCA_fitting_num_frames': 1000000,
        'conf_threshold': 0.5,
#         'kappa_scan_target_duration': 12,
#         'kappa_scan_min': 1e2,
#         'kappa_scan_max': 1e12,
#         'num_arhmm_scan_iters': 50,
#         'num_arhmm_final_iters': 200,
#         'num_kpslds_scan_iters': 50,
#         'num_kpslds_final_iters': 500
    })
    
    comments = {
        'verbose': 'whether to print progress messages during fitting',
        'keypoint_colormap': 'colormap used for visualization; see `matplotlib.cm.get_cmap` for options',
        'added_noise_level': 'upper bound of uniform noise added to the data during initial AR-HMM fitting; this is used to regularize the model',
        'PCA_fitting_num_frames': 'number of frames used to fit the PCA model during initialization',
        'video_dir': 'directory with videos from which keypoints were derived (used for crowd movies)',
        'session_name_suffix': 'suffix used to match videos to session names; this can usually be left empty (see `util.find_matching_videos` for details)',
        'bodyparts': 'used to access columns in the keypoint data',
        'skeleton': 'used for visualization only',
        'use_bodyparts': 'determines the subset of bodyparts to use for modeling and the order in which they are represented',
        'anterior_bodyparts': 'used to initialize heading',
        'posterior_bodyparts': 'used to initialize heading',
        'seg_length': 'data are broken up into segments to parallelize fitting',
        'trans_hypparams': 'transition hyperparameters',
        'ar_hypparams': 'autoregressive hyperparameters',
        'obs_hypparams': 'keypoint observation hyperparameters',
        'cen_hypparams': 'centroid movement hyperparameters',
        'error_estimator': 'parameters to convert neural net likelihoods to error size priors',
        'save_every_n_iters': 'frequency for saving model snapshots during fitting; if 0 only final state is saved', 
        'kappa_scan_target_duration': 'target median syllable duration (in frames) for choosing kappa',
        'whiten': 'whether to whiten principal components; used to initialize the latent pose trajectory `x`',
        'conf_threshold': 'used to define outliers for interpolation when the model is initialized',
        'conf_pseudocount': 'pseudocount used regularize neural network confidences',
        'fix_heading': 'whether to keep the heading angle fixed; this should only be True if the pose is constrained to a narrow range of angles, e.g. a headfixed mouse.',
    }

    sections = [
        ('ANATOMY', anatomy),
        ('FITTING', fitting),
        ('HYPER PARAMS',hypperams),
        ('OTHER', other)
    ]

    with open(os.path.join(project_dir,'config.yml'),'w') as f: 
        f.write(_build_yaml(sections, comments))
                          
        
def check_config_validity(config):
    """
    Check if the config is valid.

    To be valid, the config must satisfy the following criteria:
        - All the elements of `config["use_bodyparts"]` are 
          also in `config["bodyparts"]` 
        - All the elements of `config["anterior_bodyparts"]` are
          also in `config["bodyparts"]` 
        - All the elements of `config["anterior_bodyparts"]` are
          also in `config["bodyparts"]` 
        - For each pair in `config["skeleton"]`, both elements 
          also in `config["bodyparts"]` 

    Parameters
    ----------
    config: dict 

    Returns
    -------
    validity: bool
    """
    error_messages = []
    
    # check anatomy
    for bodypart in config['use_bodyparts']:
        if not bodypart in config['bodyparts']:
            error_messages.append(           
                f'ACTION REQUIRED: `use_bodyparts` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')
            
    for bodypart in sum(config['skeleton'],[]):
        if not bodypart in config['bodyparts']:
            error_messages.append(
                f'ACTION REQUIRED: `skeleton` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')
            
    for bodypart in config['anterior_bodyparts']:
        if not bodypart in config['bodyparts']:
            error_messages.append(
                f'ACTION REQUIRED: `anterior_bodyparts` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')
            
    for bodypart in config['posterior_bodyparts']:
        if not bodypart in config['bodyparts']:
            error_messages.append(     
                f'ACTION REQUIRED: `posterior_bodyparts` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')

    if len(error_messages)==0: 
        return True
    for msg in error_messages: 
        print(fill(msg, width=70, subsequent_indent='  '), end='\n\n')
    return False
            

def load_config(project_dir, check_if_valid=True, build_indexes=True):
    """
    Load a project config file.
    
    Parameters
    ----------
    project_dir: str
        Directory containing the config file
        
    check_if_valid: bool, default=True
        Check if the config is valid using 
        :py:func:`keypoint_moseq.io.check_config_validity`
        
    build_indexes: bool, default=True
        Add keys `"anterior_idxs"` and `"posterior_idxs"` to the 
        config. Each maps to a jax array indexing the elements of 
        `config["anterior_bodyparts"]` and 
        `config["posterior_bodyparts"]` by their order in 
        `config["use_bodyparts"]`

    Returns
    -------
    config: dict
    """
    config_path = os.path.join(project_dir,'config.yml')
    
    with open(config_path, 'r') as stream:  
        config = yaml.safe_load(stream)

    if check_if_valid: 
        check_config_validity(config)
        
    if build_indexes:
        config['anterior_idxs'] = jnp.array(
            [config['use_bodyparts'].index(bp) for bp in config['anterior_bodyparts']])
        config['posterior_idxs'] = jnp.array(
            [config['use_bodyparts'].index(bp) for bp in config['posterior_bodyparts']])
    
    if not 'skeleton' in config or config['skeleton'] is None:
        config['skeleton'] = []
        
    return config

def update_config(project_dir, **kwargs):
    """
    Update the config file stored at `project_dir/config.yml`.
     
    Use keyword arguments to update key/value pairs in the config.
    To update model hyperparameters, just use the name of the 
    hyperparameter as the keyword argument. 

    Examples
    --------
    To update `video_dir` to `/path/to/videos`::

      >>> update_config(project_dir, video_dir='/path/to/videos')
      >>> print(load_config(project_dir)['video_dir'])
      /path/to/videos

    To update `trans_hypparams['kappa']` to `100`::

      >>> update_config(project_dir, kappa=100)
      >>> print(load_config(project_dir)['trans_hypparams']['kappa'])
      100
    """
    config = load_config(project_dir, check_if_valid=False, build_indexes=False)
    config.update(kwargs)
    generate_config(project_dir, **config)
    
        
def setup_project(project_dir, deeplabcut_config=None, sleap_file=None,
                  overwrite=False, **options):
    """
    Setup a project directory with the following structure::

        project_dir
        └── config.yml
    
    Parameters
    ----------
    project_dir: str 
        Path to the project directory (relative or absolute)
        
    deeplabcut_config: str, default=None
        Path to a deeplabcut config file. Relevant settings, including
        `'bodyparts'`, `'skeleton'`, `'use_bodyparts'`, and 
        `'video_dir'` will be imported from the deeplabcut config and 
        used to initialize the keypoint MoSeq config. (overrided by kwargs). 

    sleap_file: str, default=None
        Path to a sleap hdf5 file for one of the recordings to be modeled. 
        Relevant settings, including `'bodyparts'`, `'skeleton'`, 
        and `'use_bodyparts'` will be imported from the sleap file and used 
        to initialize the keypoint MoSeq config. (overrided by kwargs). 
        
    overwrite: bool, default=False
        Overwrite any config.yml that already exists at the path
        `{project_dir}/config.yml`
        
    options
        Used to initialize config file. Overrides default settings
    """

    if os.path.exists(project_dir) and not overwrite:
        print(fill(
            f'The directory `{project_dir}` already exists. Use '
            '`overwrite=True` or pick a different name'))
        return
        
    if deeplabcut_config is not None: 
        dlc_options = {}
        with open(deeplabcut_config, 'r') as stream:           
            dlc_config = yaml.safe_load(stream)
            
            if dlc_config is None:
                raise RuntimeError(
                    f'{deeplabcut_config} does not exists or is not a'
                    ' valid yaml file')
                
            if 'multianimalproject' in dlc_config and dlc_config['multianimalproject']:
                dlc_options['bodyparts'] = dlc_config['multianimalbodyparts']
                dlc_options['use_bodyparts'] = dlc_config['multianimalbodyparts']
            else:
                dlc_options['bodyparts'] = dlc_config['bodyparts']
                dlc_options['use_bodyparts'] = dlc_config['bodyparts']

            dlc_options['skeleton'] = dlc_config['skeleton']
            dlc_options['video_dir'] = os.path.join(dlc_config['project_path'],'videos')

        options = {**dlc_options, **options}

    elif sleap_file is not None:
        sleap_options = {}
        with h5py.File(sleap_file, 'r') as f:

            node_names = [n.decode('utf-8') for n in f['node_names']]
            edge_names = [[n.decode('utf-8') for n in edge] for edge in f['edge_names']]
            
            sleap_options['bodyparts'] = node_names
            sleap_options['use_bodyparts'] = node_names
            sleap_options['skeleton'] = edge_names
            
        options = {**sleap_options, **options}
    
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    generate_config(project_dir, **options)
            
    
def format_data(coordinates, confidences=None, keys=None, 
                seg_length=None, bodyparts=None, use_bodyparts=None,
                conf_pseudocount=1e-3, added_noise_level=0.1, **kwargs):
    """
    Format keypoint coordinates and confidences for inference.

    Data are transformed as follows:
        1. Coordinates and confidences are each merged into a single 
           array using :py:func:`keypoint_moseq.util.batch`. 
        2. The keypoints axis is reindexed according to the order
           of elements in `use_bodyparts` with respect to their 
           initial orer in `bodyparts`.
        3. Uniform noise proportional to `added_noise_level` is
           added to the keypoint coordinates to prevent degenerate
           solutions during fitting. 
        4. Keypoint confidences are augmented by `conf_pseudocount`.
        5. Wherever NaNs occur in the coordinates, they are replaced
           by values imputed using linear interpolation, and the
           corresponding confidences are set to `conf_pseudocount`.
    
    Parameters
    ----------
    coordinates: dict
        Keypoint coordinates for a collection of sessions. Values
        must be numpy arrays of shape (T,K,D) where K is the number
        of keypoints and D={2 or 3}. 
        
    confidences: dict, default=None
        Nonnegative confidence values for the keypoints in 
        `coordinates` as numpy arrays of shape (T,K).
        
    keys: list of str, default=None
        (See :py:func:`keypoint_moseq.util.batch`)
        
    bodyparts: list, default=None
        Label for each keypoint represented in `coordinates`. Required
        to reindex coordinates and confidences according to `use_bodyparts`.

    use_bodyparts: list, default=None
        Ordered subset of keypoint labels to be used for modeling.
        If `use_bodyparts=None`, then all keypoints are used.

    conf_pseudocount: float, default=1e-3
        Pseudocount used to augment keypoint confidences.
    
    seg_length: int, default=None
        Length of each segment. If `seg_length=None`, a length is 
        chosen so that no time-series are broken into multiple segments.
        If all time-series are shorter than `seg_length`, then
        `seg_length` is set to the length of the shortest time-series.
        
    Returns
    -------
    data: dict with the following items
    
        Y: jax array with shape (n_segs, seg_length, K, D)
            Keypoint coordinates from all sessions broken into 
            fixed-length segments.
            
        conf: jax array with shape (n_segs, seg_length, K)
            Confidences from all sessions broken into fixed-length 
            segments. If no input is provided for `confidences`, 
            then `data["conf"]=None`.
        
        mask: jax array with shape (n_segs, seg_length)
            Binary array where 0 indicates areas of padding 
            (see :py:func:`keypoint_moseq.util.batch`).
            
    labels: list of tuples (object, int, int)
        Label for each row of `Y` and `conf` 
        (see :py:func:`keypoint_moseq.util.batch`).
    """    
    if keys is None: 
        keys = sorted(coordinates.keys()) 
    else: 
        bad_keys = set(keys) - set(coordinates.keys())
        assert len(bad_keys) == 0, fill(
            f'Keys {bad_keys} not found in coordinates')

    assert len(keys) > 0, 'No sessions found'

    num_keypoints = [coordinates[key].shape[-2] for key in keys]
    assert len(set(num_keypoints)) == 1, fill(
        f'All sessions must have the same number of keypoints, but '
        f'found {set(num_keypoints)} keypoints across sessions.')
    
    if bodyparts is not None:
        assert len(bodyparts) == num_keypoints[0], fill(
            f'The number of keypoints in `coordinates` ({num_keypoints[0]}) '
            f'does not match the number of labels in `bodyparts` '
            f'({len(bodyparts)})')

    if any(['/' in key for key in keys]): 
        warnings.warn(fill(
            'WARNING: Session names should not contain "/", this will cause '
            'problems with saving/loading hdf5 files.'))
        
    if confidences is None:
        confidences = {key: np.ones_like(coordinates[key][...,0]) for key in keys}

    if bodyparts is not None and use_bodyparts is not None:
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)
        confidences = reindex_by_bodyparts(confidences, bodyparts, use_bodyparts)

    for key in keys:
        outliers = np.isnan(coordinates[key]).any(-1)
        coordinates[key] = interpolate_keypoints(coordinates[key], outliers)
        confidences[key] = np.where(outliers, 0, np.nan_to_num(confidences[key]))
    
    if seg_length is not None:
        max_session_length = max([coordinates[key].shape[0] for key in keys])
        seg_length = min(seg_length, max_session_length)

    Y,mask,labels = batch(coordinates, seg_length=seg_length, keys=keys)
    Y = Y.astype(float)

    conf = batch(confidences, seg_length=seg_length, keys=keys)[0]
    if np.min(conf) < 0: 
        conf = np.maximum(conf,0) 
        warnings.warn(fill(
            'Negative confidence values are not allowed and will be set to 0.'))
    conf = conf + conf_pseudocount
  
    if added_noise_level>0: 
        Y += np.random.uniform(-added_noise_level,added_noise_level,Y.shape)
        
    return jax.device_put({'mask':mask, 'Y':Y, 'conf':conf}), labels


def save_pca(pca, project_dir, pca_path=None):
    """
    Save a PCA model to disk.

    The model is saved to `pca_path` or else to 
    `{project_dir}/pca.p`.
    
    Parameters
    ----------
    pca: :py:class:`sklearn.decomposition.PCA`
    project_dir: str
    pca_path: str, default=None
    """
    if pca_path is None: 
        pca_path = os.path.join(project_dir,'pca.p')
    joblib.dump(pca, pca_path)
    
def load_pca(project_dir, pca_path=None):
    """
    Load a PCA model from disk.

    The model is loaded from `pca_path` or else from 
    `{project_dir}/pca.p`.

    Parameters
    ----------
    project_dir: str
    pca_path: str, default=None

    Returns
    -------
    pca: :py:class:`sklearn.decomposition.PCA`
    """ 
    if pca_path is None:
        pca_path = os.path.join(project_dir,'pca.p')
        assert os.path.exists(pca_path), fill(
            f'No PCA model found at {pca_path}')
    return joblib.load(pca_path)


def load_last_checkpoint(project_dir):
    """
    Load checkpoint for the most recent model.

    This method assumes the following directory structure for saved
    model checkpoints::

        project_dir
        ├──YYYY_MM_DD-HH_MM_SS
        │  └checkpoint.p
        ⋮
        └──YYYY_MM_DD-HH_MM_SS
           └checkpoint.p

    Parameters
    ----------
    project_dir: str

    Returns
    -------
    checkpoint: dict
        (See :py:func:`keypoint_moseq.io.load_checkpoint`)
    """ 
    pattern = re.compile(r'(\d{4}_\d{1,2}_\d{1,2}-\d{2}_\d{2}_\d{2})')
    paths = list(filter(lambda p: pattern.search(p), os.listdir(project_dir)))
    assert len(paths)>0, fill(
        f'There are no directories in {project_dir} that contain'
        ' a date string with format %Y_%m_%d-%H_%M_%S')
    
    name = sorted(
        paths, key=lambda p: datetime.strptime(
        pattern.search(p).group(), '%Y_%m_%d-%H_%M_%S')
    )[-1]
    
    path = os.path.join(project_dir,name,'checkpoint.p')
    return load_checkpoint(path=path), name


def load_checkpoint(project_dir=None, name=None, path=None):
    """
    Load model fitting checkpoint.

    The checkpoint path can be specified directly via `path`.
    Othewise is assumed to be `{project_dir}/<name>/checkpoint.p`.

    Parameters
    ----------
    project_dir: str, default=None
    name: str, default=None
    path: str, default=None

    Returns
    -------
    checkpoint: dict
        See :py:func:`keypoint_moseq.io.save_checkpoint`
    """
    if path is None: 
        assert project_dir is not None and name is not None, fill(
            '`name` and `project_dir` are required if no `path` is given.')
        path = os.path.join(project_dir,name,'checkpoint.p')
    return joblib.load(path)


def save_checkpoint(model, data, history, labels, iteration, 
                    path=None, name=None, project_dir=None,
                    save_history=True, save_states=True, save_data=True):
    """
    Save a checkpoint during model fitting.

    A single checkpoint file contains model snapshots from the full history
    of model fitting. To restart fitting from an iteration earlier than the
    last iteration, use :py:func:`keypoint_moseq.fitting.revert`.

    The checkpoint path can be specified directly via `path`.
    Otherwise is assumed to be `{project_dir}/<name>/checkpoint.p`. See
    :py:func:`keypoint_moseq.fitting.fit_model` for a more detailed
    description of the checkpoint contents.

    Parameters
    ----------
    model: dict, history: dict
        See :py:func:`keypoint_moseq.fitting.fit_model`

    data: dict, labels: list of tuples
        See :py:func:`keypoint_moseq.io.format_data`

    iteration: int
        Current iteration of model fitting

    save_history: bool, default=True
        Whether to include `history` in the checkpoint

    save_states: bool, default=True
        Whether to include `states` in the checkpoint

    save_data: bool, default=True
        Whether to include `Y`, `conf`, and `mask` in the checkpoint
    
    project_dir: str, default=None
        Project directory; used in conjunction with `name` to determine
        the checkpoint path if `path` is not specified.

    name: str, default=None
        Model name; used in conjunction with `project_dir` to determine
        the checkpoint path if `path` is not specified.

    path: str, default=None
        Checkpoint path; if not specified, the checkpoint path is determined
        from `project_dir` and `name`.


    Returns
    -------
    checkpoint: dict
        Dictionary containing `history`, `labels` and `name` as 
        well as the key/value pairs from `model` and `data`.
    """
    
    if path is None: 
        assert project_dir is not None and name is not None, fill(
            '`name` and `project_dir` are required if no `path` is given.')
        path = os.path.join(project_dir,name,'checkpoint.p')

    dirname = os.path.dirname(path)
    if not os.path.exists(dirname): 
        print(fill(f'Creating the directory {dirname}'))
        os.makedirs(dirname)
    
    save_dict = {
        'labels': labels,
        'iteration' : iteration,
        'hypparams' : jax.device_get(model['hypparams']),
        'params'    : jax.device_get(model['params']), 
        'seed'      : np.array(model['seed']),
        'name'      : name}

    if save_data: 
        save_dict.update(jax.device_get(data))
        
    if save_states or save_data: 
        save_dict['mask'] = np.array(data['mask'])
        
    if save_states: 
        save_dict['states'] = jax.device_get(model['states'])
        save_dict['noise_prior'] = jax.device_get(model['noise_prior'])
        
    if save_history:
        save_dict['history'] = history
        
    joblib.dump(save_dict, path)
    return save_dict

    
def load_results(project_dir=None, name=None, path=None):
    """
    Load the results from a modeled dataset.

    The results path can be specified directly via `path`. Otherwise
    it is assumed to be `{project_dir}/<name>/results.h5`.
    
    Parameters
    ----------
    project_dir: str, default=None
    name: str, default=None
    path: str, default=None

    Returns
    -------
    results: dict
        See :py:func:`keypoint_moseq.fitting.apply_model`
    """
    if path is None: 
        assert project_dir is not None and name is not None, fill(
            '`name` and `project_dir` are required if no `path` is given.')
        path = os.path.join(project_dir,name,'results.h5')
    return load_hdf5(path)


def save_results_as_csv(project_dir=None, name=None, h5_path=None, 
                        save_dir=None, use_bodyparts=None,
                        path_sep='-', **kwargs):
    """
    Convert modeling results from h5 to csv format.

    The input h5 file is assumed to contain modeling outputs for one
    or more recordings. This function creates a directory and then
    saves a separate csv file for each.

    The path to the input h5 file can be specified directly via
    `results_path`. Otherwise it is assumed to be
    `{project_dir}/{name}/results.h5`. The path to the output
    directory can be specified directly via `save_dir`. Otherwise
    it will be set to `{project_dir}/{name}/results`. Any files
    already in the output directory will be overwritten.
    
    Parameters
    ----------
    project_dir: str, default=None
        Project directory; required if `h5_path` or `save_dir` is not provided.

    name: str, default=None
        Name of the model; required if `h5_path` or `save_dir` is not provided.

    h5_path: str, default=None
        Path to the h5 file containing modeling results.

    save_dir: str, default=None
        Path to the directory where the csv files will be saved.

    use_bodyparts: list, default=None
        List of bodyparts that were used for modeling. If provided,
        will be used for the csv column names corresponding to 
        `estimated_coordinates`. Otherwise, the bodyparts will be
        named `bodypart0`, `bodypart1` etc.

    path_sep: str, default='-'
        If a path separator ("/" or "\") is present in the recording name, 
        it will be replaced with `path_sep` when saving the csv file.
    """

    if h5_path is None:
        assert project_dir is not None and name is not None, fill(
            'Provide either a `h5_path` or a `project_dir` and `name`')
        h5_path = os.path.join(project_dir, name, 'results.h5')

    if save_dir is None:
        assert project_dir is not None and name is not None, fill(
            'Provide either a `save_dir` or a `project_dir` and `name`')
        save_dir = os.path.join(project_dir, name, 'results')

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    with h5py.File(h5_path, 'r') as results:
        for key in tqdm.tqdm(results.keys(), desc='Saving to csv'):
            column_names, data = [], []

            if 'syllables_reindexed' in results[key].keys():
                column_names.append(['syllables reindexed'])
                data.append(results[key]['syllables_reindexed'][()].reshape(-1,1))

            if 'syllables' in results[key].keys():
                column_names.append(['syllables non-reindexed'])
                data.append(results[key]['syllables'][()].reshape(-1,1))

            if 'centroid' in results[key].keys():
                d = results[key]['centroid'].shape[1]
                column_names.append(['centroid x', 'centroid y', 'centroid z'][:d])
                data.append(results[key]['centroid'][()])

            if 'heading' in results[key].keys():
                column_names.append(['heading'])
                data.append(results[key]['heading'][()].reshape(-1,1))

            if 'estimated_coordinates' in results[key].keys():
                k,d = results[key]['estimated_coordinates'].shape[1:]
                if use_bodyparts is None:
                    use_bodyparts = [f'bodypart{i}' for i in range(k)]
                for i, bp in enumerate(use_bodyparts):
                    column_names.append([f'estimated {bp} x', f'estimated {bp} y', f'{bp} z'][:d])
                    data.append(results[key]['estimated_coordinates'][:,i,:])

            if 'latent_state' in results[key].keys():
                latent_dim = results[key]['latent_state'].shape[1]
                column_names.append([f'latent_state {i}' for i in range(latent_dim)])
                data.append(results[key]['latent_state'][()])

            dfs = [pd.DataFrame(arr, columns=cols) for arr,cols in zip(data,column_names)]
            df = pd.concat(dfs, axis=1)

            for col in df.select_dtypes(include=[np.floating]).columns:
                df[col] = df[col].astype(float).round(4)

            save_name = key.replace(os.path.sep, path_sep)
            save_path = os.path.join(save_dir, save_name)
            df.to_csv(f'{save_path}.csv', index=False)




def _name_from_path(filepath, path_in_name, path_sep, remove_extension):
    """
    Create a name from a filepath. Either return the name of the file
    (with the extension removed) or return the full filepath, where the
    path separators are replaced with `path_sep`.
    """
    if remove_extension:
        filepath = os.path.splitext(filepath)[0]
    if path_in_name:
        return filepath.replace(os.path.sep, path_sep)
    else:
        return os.path.basename(filepath)


def _print_colored_table(row_labels, col_labels, values):
    try:
        from IPython.display import display
        display_available = True
    except ImportError:
        display_available = False

    title = 'Proportion of NaNs'
    df = pd.DataFrame(values, index=row_labels, columns=col_labels)
    
    if display_available:
        def colorize(val):
            color = plt.get_cmap("Reds")(val*0.8)
            return f'background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})'  
        colored_df = df.style.applymap(colorize).set_caption('Proportion of NaNs')
        display(colored_df)
        return colored_df
    else:
        print(title)
        print(tabulate(df, headers='keys', tablefmt="simple_grid", showindex=True))


def check_nan_proportions(coordinates, bodyparts, 
                          warning_threshold=0.5,
                          breakdown=False, **kwargs):
    """
    Check if any bodyparts have a high proportion of NaNs.
    
    Parameters
    ----------
    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays
        of shape (n_frames, n_bodyparts, 2)

    bodyparts: list of str
        Name of each bodypart. The order of the names should match the
        order of the bodyparts in `coordinates`.

    warning_threshold: float, default=0.5
        If the proportion of NaNs for a bodypart is greater than
        `warning_threshold`, then a warning is printed.
        
    breakdown: bool, default=False
        Whether to print a table detailing the proportion of NaNs
        for each bodyparts in each array of `coordinates`.
    """
    if breakdown:
        keys = sorted(coordinates.keys())
        nan_props = [np.isnan(coordinates[k]).any(-1).mean(0) for k in keys]
        _print_colored_table(keys, bodyparts, nan_props)
    else:
        all_coords = np.concatenate(list(coordinates.values()))
        nan_props = np.isnan(all_coords).any(-1).mean(0)
        if np.any(nan_props > warning_threshold):
            bps = [bp for bp,p in zip(bodyparts,nan_props) if p > warning_threshold]
            warnings.warn(
                 '\nCoordinates for the following bodyparts are missing (set to NaN) in at least '
                 '{}% of frames:\n - {}\n\n'.format(warning_threshold*100, '\n - '.join(bps)))
            warnings.warn(
                'This may cause problems during modeling. See '
                'https://keypoint-moseq.readthedocs.io/en/latest/FAQs.html#high-proportion-of-nans'
                ' for additional information.')



def load_deeplabcut_results(filepath_pattern, recursive=True, path_sep='-',
                            path_in_name=False, remove_extension=True):
    """
    Load tracking results from deeplabcut csv or hdf5 files.

    For single-animal tracking, each file yields a single key/value 
    pair in the returned `coordinates` and `confidences` dictionaries. For
    multi-animal tracking, a key/vaue pair will be generated for each
    tracked individual. For example a single file called `two_mice.h5`
    with individuals "mouseA" and "mouseB" will yield the pair of
    keys `'two_mice_mouseA', 'two_mice_mouseB'`. 

    Parameters
    ----------
    filepath_pattern: str or list of str
        Filepath pattern for a set of deeplabcut csv or hdf5 files, 
        or a list of such patterns. Filepath patterns can be:

            - single file (e.g. `/path/to/file.csv`) 
            - single directory (e.g. `/path/to/dir/`)
            - set of files (e.g. `/path/to/fileprefix*`)
            - set of directories (e.g. `/path/to/dirprefix*`)

    recursive: bool, default=True
        Whether to search recursively for deeplabcut csv or hdf5 files.

    path_in_name: bool, default=False
        Whether to name the tracking results from each file by the path
        to the file (True) or just the filename (False). If True, the
        `path_sep` argument is used to separate the path components.
        
    path_sep: str, default='-'
        Separator to use when `path_in_name` is True. For example,
        if `path_sep` is `'-'`, then the tracking results from the
        file `/path/to/file.csv` will be named `path-to-file`. Using
        `'/'` as the separator is discouraged, as it will cause problems
        saving/loading the modeling results to/from hdf5 files.

    remove_extension: bool, default=True
        Whether to remove the file extension from the name of the tracking
        results.

    Returns
    -------
    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays
        of shape (n_frames, n_bodyparts, 2)

    confidences: dict
        Dictionary mapping filenames to `likelihood` scores as ndarrays
        of shape (n_frames, n_bodyparts)

    bodyparts: list of str
        List of bodypart names. The order of the names matches the order
        of the bodyparts in `coordinates` and `confidences`.
    """
    filepaths = list_files_with_exts(
        filepath_pattern, ['.csv','.h5','.hdf5'], recursive=recursive)
    assert len(filepaths)>0, fill(
        f'No deeplabcut csv or hdf5 files found for {filepath_pattern}')

    coordinates,confidences = {},{}
    for filepath in tqdm.tqdm(filepaths, desc='Loading from deeplabcut'):

        ext = os.path.splitext(filepath)[1]
        if ext=='.h5': df = pd.read_hdf(filepath)
        if ext=='.csv': df = pd.read_csv(filepath, header=[0,1,2], index_col=0) 
        
        name = _name_from_path(filepath, path_in_name, path_sep, remove_extension)
        bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()   
        
        if 'individuals' in df.columns.names:
            for ind in df.columns.get_level_values('individuals').unique():
                ind_df = df.xs(ind, axis=1, level='individuals')
                arr = ind_df.to_numpy().reshape(len(ind_df), -1, 3)
                coordinates[f'{name}_{ind}'] = arr[:,:,:-1]
                confidences[f'{name}_{ind}'] = arr[:,:,-1]
        else:
            arr = df.to_numpy().reshape(len(df), -1, 3)
            coordinates[name] = arr[:,:,:-1]
            confidences[name] = arr[:,:,-1]

        # except Exception as e: 
        #     print(fill(f'Error loading {filepath}: {e}'))
    
    if len(coordinates) > 0:
        check_nan_proportions(coordinates, bodyparts)

    return coordinates,confidences,bodyparts



def load_sleap_results(filepath_pattern, recursive=True, path_sep='-',
                       path_in_name=False, remove_extension=True):
    """
    Load keypoints from sleap hdf5 files. 

    For single-animal tracking, each file yields a single key/value 
    pair in the returned `coordinates` and `confidences` dictionaries. For
    multi-animal tracking, a key/vaue pair will be generated for each track.
    For example a single file called `two_mice.h5` will yield the pair of
    keys `'two_mice_track0', 'two_mice_track1'`.

    Parameters
    ----------
    filepath_pattern: str, default=None
        Filepath pattern for a set of sleap hdf5 files, or a list of 
        such patterns. Filepath patterns can be:

            - single file (e.g. `/path/to/file.csv`) 
            - single directory (e.g. `/path/to/dir/`)
            - set of files (e.g. `/path/to/fileprefix*`)
            - set of directories (e.g. `/path/to/dirprefix*`)

    recursive: bool, default=True
        Whether to search recursively for sleap hdf5 files.

    path_in_name: bool, default=False
        Whether to name the tracking results from each file by the path
        to the file (True) or just the filename (False). If True, the
        `path_sep` argument is used to separate the path components.
        
    path_sep: str, default='-'
        Separator to use when `path_in_name` is True. For example,
        if `path_sep` is `'-'`, then the tracking results from the
        file `/path/to/file.csv` will be named `path-to-file`. Using
        `'/'` as the separator is discouraged, as it will cause problems
        saving/loading the modeling results to/from hdf5 files.

    remove_extension: bool, default=True
        Whether to remove the file extension from the name of the tracking
        results.

    Returns
    -------
    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays
        of shape (n_frames, n_bodyparts, 2)

    confidences: dict
        Dictionary mapping filenames to `likelihood` scores as ndarrays
        of shape (n_frames, n_bodyparts)

    bodyparts: list of str
        List of bodypart names. The order of the names matches the order
        of the bodyparts in `coordinates` and `confidences`.
    """
    filepaths = list_files_with_exts(
        filepath_pattern, ['.h5','.hdf5'], recursive=recursive)
    assert len(filepaths)>0, fill(
        f'No sleap hdf5 files found for {filepath_pattern}.')

    coordinates,confidences = {},{}
    for filepath in tqdm.tqdm(filepaths, desc='Loading from sleap'):
        try: 
            name = _name_from_path(filepath, path_in_name, path_sep, remove_extension)
            with h5py.File(filepath, 'r') as f:
                coords = f['tracks'][()]
                confs = f['point_scores'][()]
                bodyparts = [name.decode('utf-8') for name in f['node_names']]
                if coords.shape[0] == 1: 
                    coordinates[name] = coords[0].T
                    confidences[name] = confs[0].T
                else:
                    for i in range(coords.shape[0]):
                        coordinates[f'{name}_track{i}'] = coords[i].T
                        confidences[f'{name}_track{i}'] = confs[i].T
        except Exception as e: 
            print(fill(f'Error loading {filepath}: {e}'))

    if len(coordinates) > 0:
        check_nan_proportions(coordinates, bodyparts)

    return coordinates,confidences,bodyparts

# hdf5 save/load routines modified from
# https://gist.github.com/nirum/b119bbbd32d22facee3071210e08ecdf

def save_hdf5(filepath, save_dict):
    """
    Save a dict of pytrees to an hdf5 file.
    
    Parameters
    ----------
    filepath: str
        Path of the hdf5 file to create.

    save_dict: dict
        Dictionary where the values are pytrees, i.e. recursive 
        collections of tuples, lists, dicts, and numpy arrays.
    """
    with h5py.File(filepath, 'a') as f:
        for k,tree in save_dict.items():
            _savetree_hdf5(jax.device_get(tree), f, k)

def load_hdf5(filepath):
    """
    Load a dict of pytrees from an hdf5 file.

    Parameters
    ----------
    filepath: str
        Path of the hdf5 file to load.
            
    Returns
    -------
    save_dict: dict
        Dictionary where the values are pytrees, i.e. recursive
        collections of tuples, lists, dicts, and numpy arrays.
    """
    with h5py.File(filepath, 'r') as f:
        return {k:_loadtree_hdf5(f[k]) for k in f}

def _savetree_hdf5(tree, group, name):
    """Recursively save a pytree to an h5 file group."""
    if name in group: del group[name]
    if isinstance(tree, np.ndarray):
        group.create_dataset(name, data=tree)
    else:
        subgroup = group.create_group(name)
        subgroup.attrs['type'] = type(tree).__name__
        if isinstance(tree, tuple) or isinstance(tree, list):
            for k, subtree in enumerate(tree):
                _savetree_hdf5(subtree, subgroup, f'arr{k}')
        elif isinstance(tree, dict):
            for k, subtree in tree.items():
                _savetree_hdf5(subtree, subgroup, k)
        else: raise ValueError(f'Unrecognized type {type(tree)}')

def _loadtree_hdf5(leaf):
    """Recursively load a pytree from an h5 file group."""
    if isinstance(leaf, h5py.Dataset):
        return np.array(leaf)
    else:
        leaf_type = leaf.attrs['type']
        values = map(_loadtree_hdf5, leaf.values())
        if leaf_type == 'dict': return dict(zip(leaf.keys(), values))
        elif leaf_type == 'list': return list(values)
        elif leaf_type == 'tuple': return tuple(values)
        else: raise ValueError(f'Unrecognized type {leaf_type}')

