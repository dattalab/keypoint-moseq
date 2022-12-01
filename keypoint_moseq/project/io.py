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
import tqdm
import re
import pandas as pd
from datetime import datetime
from textwrap import fill
from vidio.read import OpenCVReader
from keypoint_moseq.util import batch, reindex_by_bodyparts
warnings.formatwarning = lambda msg, *a: str(msg)

def build_yaml(sections, comments):
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
    Generate a config.yml file with project settings.
    Default settings will be used unless overriden by 
    a keywork argument.
    
    Parameters
    ----------
    project_dir: str 
        A file ``config.yml`` will be generated in this directory.
    
    **kwargs
        Custom project settings.
        
    """
    
    def update_dict(new, original):
        return {k:new[k] if k in new else v for k,v in original.items()} 
    
    hypperams = {k: update_dict(kwargs,v) for k,v in {
        'error_estimator': {'slope':1, 'intercept':1},
        'obs_hypparams': {'sigmasq_0':0.1, 'sigmasq_C':.1, 'nu_sigma':1e5, 'nu_s':5},
        'ar_hypparams': {'nlags': 3, 'S_0_scale': 0.01, 'K_0_scale': 10.0},
        'trans_hypparams': {'num_states': 100, 'gamma': 1e3, 'alpha': 5.7, 'kappa': 1e6},
        'cen_hypparams': {'sigmasq_loc': 0.5}
    }.items()}

    anatomy = update_dict(kwargs, {
        'bodyparts': ['BODYPART1','BODYPART2','BODYPART3'],
        'use_bodyparts': ['BODYPART1','BODYPART2','BODYPART3'],
        'skeleton': [['BODYPART1','BODYPART2'], ['BODYPART2','BODYPART3']],
        'anterior_bodyparts': ['BODYPART1'],
        'posterior_bodyparts': ['BODYPART3']})
        
    other = update_dict(kwargs, {
        'verbose':True,
        'conf_pseudocount': 1e-3,
        'video_dir': '',
        'keypoint_colormap': 'autumn',
        'latent_dimension': 10,
        'whiten': True,
        'seg_length': 10000 })
       
    fitting = update_dict(kwargs, {
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
        'video_dir': 'directory with videos from which keypoints were derived (used for crowd movies)',
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
    }
    
    sections = [
        ('ANATOMY', anatomy),
        ('FITTING', fitting),
        ('HYPER PARAMS',hypperams),
        ('OTHER', other)
    ]

    with open(os.path.join(project_dir,'config.yml'),'w') as f: 
        f.write(build_yaml(sections, comments))
                          
        
def check_config_validity(config):
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

    if len(error_messages)>0: 
        print('')
    for msg in error_messages: 
        print(fill(msg, width=70, subsequent_indent='  '), end='\n\n')
            
def load_config(project_dir, check_if_valid=True):
    """
    Load config.yml from ``project_dir`` and return
    the resulting dict. Optionally check if the config is valid. 
    """
    config_path = os.path.join(project_dir,'config.yml')
    with open(config_path, 'r') as stream:  config = yaml.safe_load(stream)
    if check_if_valid: check_config_validity(config)
    return config

def update_config(project_dir, **kwargs):
    """
    Update config.yml from ``project_dir`` to include
    all the key/value pairs in **kwargs.
    """
    config = load_config(project_dir, check_if_valid=False)
    config.update(kwargs)
    generate_config(project_dir, **config)
    
        
def setup_project(project_dir, deeplabcut_config=None, 
                  overwrite=False, **options):
    """
    Setup a project directory with the following structure
    ```
        project_dir
        └── config.yml
    ```
    
    Parameters
    ----------
    project_dir: str 
        Path to the project directory (relative or absolute)
        
    deeplabcut_config: str, default=None
        Path to a deeplabcut config file. Relevant settings will be
        imported and used to initialize the keypoint MoSeq config.
        (overrided by **kwargs)
        
    overwrite: bool, default=False
        Overwrite any config.yml that already exists at the path
        ``[project_dir]/config.yml``
        
    **options
        Used to initialize config file
    """

    if os.path.exists(project_dir) and not overwrite:
        print(fill(f'The directory `{project_dir}` already exists. Use `overwrite=True` or pick a different name for the project directory'))
        return
        
    if deeplabcut_config is not None: 
        dlc_options = {}
        with open(deeplabcut_config, 'r') as stream:           
            dlc_config = yaml.safe_load(stream)
            
            if dlc_config is None:
                raise RuntimeError(
                    f'{deeplabcut_config} does not exists or is not a valid yaml file')
                
            if 'multianimalproject' in dlc_config and dlc_config['multianimalproject']:
                raise NotImplementedError(
                    'Config initialization from multi-animal deeplabcut '
                    'projects is not yet supported')
                
            dlc_options['bodyparts'] = dlc_config['bodyparts']
            dlc_options['use_bodyparts'] = dlc_config['bodyparts']
            dlc_options['skeleton'] = dlc_config['skeleton']
            dlc_options['video_dir'] = os.path.join(dlc_config['project_path'],'videos')
                
        options = {**dlc_options, **options}
    
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    generate_config(project_dir, **options)
            
    
def format_data(coordinates, *, confidences=None, keys=None, 
                seg_length, bodyparts, use_bodyparts,
                conf_pseudocount=1e-3, added_noise_level=0.1, **kwargs):
    """
    Stacks variable-length time-series of keypoint coordinates into a
    single array for batch processing, optionally breaking each one into
    segments of fixed length. The is done for keypoint confidences if
    they are provided. Keypoints are also subsetted/reordered based on 
    ``use_bodyparts``. 0-padding ensures that the resulting arrays are
    not ragged. 
    
    Parameters
    ----------
    coordinates: dict
        Keypoint coordinates for a collection of sessions. Values
        must be numpy arrays of shape (T,K,D) where K is the number
        of keypoints and D={2 or 3}. Keys can be any unique str,
        but must the name of a videofile to enable downstream analyses
        such as crowd movies. 
        
    confidences: dict, default=None
        Neural network confidences for a collection of sessions. Values
        must be numpy arrays of shape (T,K) that match the corresponding 
        arrays in ``coordinates``. 
        
    bodyparts: list of str
        Name of each keypoint. Should have length K corresponding to
        the shape of arrays in ``coordinates``.
        
    use_bodyparts: list of str
        Names of keypoints to use for modeling. Should be a subset of 
        ``bodyparts``.
        
    keys: list, default=None
        Specifies a subset of sessions to include and the order in which
        to stack them. If ``keys=None``, all sessions will be used and 
        ordered using ``sorted``.
        
    conf_pseudocount: float, default=1e-3
        Pseudocount neural network confidences.
    
    seg_length: int, default=None
        Length of each segment. If ``seg_length=None``, a length is 
        chosen so that no time-series are broken into multiple segments.
        
    Returns
    -------
    data: dict with the following items
    
        Y: numpy array with shape (n_segs, seg_length, K, D)
            Keypoint coordinates from all sessions broken into segments.
            
        conf: numpy array with shape (n_segs, seg_length, K)
            Confidences from all sessions broken into segments. If no 
            input is provided for ``confidences``, ``conf`` will be set
            to ``None``. Note that confidences are increased by 
            ``conf_pseudocount``.
        
        mask: numpy array with shape (n_segs, seg_length)
            Binary array where 0 indicates areas of padding.
            
        labels: list of tuples (object, int, int)
            The location in ``data_dict`` that each segment came from
            in the form of tuples (key, start, end).
    """    
    
    if keys is None: keys = sorted(coordinates.keys()) 
    coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)
    confidences = reindex_by_bodyparts(confidences, bodyparts, use_bodyparts)
    
    Y,mask,labels = batch(coordinates, seg_length=seg_length, keys=keys)
    
    if confidences is not None:
        conf = batch(confidences, seg_length=seg_length, keys=keys)[0]
        if conf.min() < 0: 
            conf = np.maximum(conf,0) 
            warnings.warn(fill(
                'Negative confidence values are not allowed and will be set to 0.'))
        conf = conf + conf_pseudocount
  
    if added_noise_level>0: 
        Y += np.random.uniform(-added_noise_level,added_noise_level,Y.shape)
        
    return jax.device_put({'mask':mask, 'Y':Y, 'conf':conf}), labels


def save_pca(pca, project_dir, pca_path=None):
    if pca_path is None: 
        pca_path = os.path.join(project_dir,'pca.p')
    joblib.dump(pca, pca_path)
    
def load_pca(project_dir, pca_path=None):
    if pca_path is None:
        pca_path = os.path.join(project_dir,'pca.p')
        assert os.path.exists(pca_path), fill(
            f'No PCA model found at {pca_path}')
    return joblib.load(pca_path)


def load_last_checkpoint(project_dir):
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
    if path is None: 
        assert project_dir is not None and name is not None, fill(
            '``name`` and ``project_dir`` are required if no ``path`` is given.')
        path = os.path.join(project_dir,name,'checkpoint.p')
    return joblib.load(path)


def save_checkpoint(model, data, history, labels, iteration, 
                    path=None, name=None, project_dir=None,
                    save_history=True, save_states=True, save_data=True):
    
    if path is None: 
        assert project_dir is not None and name is not None, fill(
            '``name`` and ``project_dir`` are required if no ``path`` is given.')
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

    
def load_results(project_dir=None, name=None, path=None):
    if path is None: 
        assert project_dir is not None and name is not None, fill(
            '``name`` and ``project_dir`` are required if no ``path`` is given.')
        path = os.path.join(project_dir,name,'results.h5')
    return load_hdf5(path)


def load_keypoints_from_deeplabcut_file(filepath, *, bodyparts, **kwargs):
    ext = os.path.splitext(filepath)[1]
    assert ext in ['.csv','.h5']
    if ext=='.h5': df = pd.read_hdf(filepath)
    if ext=='.csv': df = pd.read_csv(filepath, header=[0,1,2], index_col=0)
        
    dlc_bodyparts = list(zip(*df.columns.to_list()))[1][::3]
    assert dlc_bodyparts==tuple(bodyparts), fill(
        f'{os.path.basename(filepath)} contains bodyparts'
        f'\n\n{dlc_bodyparts}\n\nbut expected\n\n{bodyparts}')
    
    arr = df.to_numpy().reshape(-1, len(bodyparts), 3)
    coordinates,confidences = arr[:,:,:-1],arr[:,:,-1]
    return coordinates,confidences


def load_keypoints_from_deeplabcut_list(paths, **kwargs): 
    coordinates,confidences = {},{}
    for filepath in tqdm.tqdm(paths, desc='Loading from deeplabcut'):
        filename = os.path.basename(filepath)
        coordinates[filename],confidences[filename] = \
            load_keypoints_from_deeplabcut_file(filepath, **kwargs)
    return coordinates,confidences
        
    
def load_keypoints_from_deeplabcut(*, video_dir, directory=None, **kwargs):
    if directory is None:
        directory = video_dir
        print(fill(f'Searching in {directory}. Use the ``directory`` '
              'argument to specify another search location'))
    filepaths = [
        os.path.join(directory,f) 
        for f in os.listdir(directory)
        if os.path.splitext(f)[1] in ['.csv','.h5']]
    return load_keypoints_from_deeplabcut_list(filepaths, **kwargs)


# hdf5 save/load routines modified from
# https://gist.github.com/nirum/b119bbbd32d22facee3071210e08ecdf

def save_hdf5(filepath, save_dict):
    """Saves a pytree with a dict at the root to an hdf5 file
    Args:
        filepath: str, Path of the hdf5 file to create.
        tree: pytree, Recursive collection of tuples, lists, dicts, 
        numpy arrays to store. The root is assumed to be a dict. """
    with h5py.File(filepath, 'a') as f:
        for k,tree in save_dict.items():
            _savetree_hdf5(jax.device_get(tree), f, k)

def load_hdf5(filepath):
    """Loads a pytree with a dict at the root from an hdf5 file.
    Args:
        filepath: str, Path of the hdf5 file to load."""
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

