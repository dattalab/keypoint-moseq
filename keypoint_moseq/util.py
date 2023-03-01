import numpy as np
import os
import glob
import tqdm
from textwrap import fill
from jax.config import config
config.update('jax_enable_x64', True)
import jax, jax.numpy as jnp, jax.random as jr
from jax.tree_util import tree_map
from itertools import groupby
from functools import partial
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from jaxlib.xla_extension import DeviceArray as jax_array
from jax_moseq.models.keypoint_slds import inverse_rigid_transform
na = jnp.newaxis


def np_io(fn): 
    """
    Converts a function involving jax arrays to one that inputs and
    outputs numpy arrays.
    """
    return lambda *args, **kwargs: jax.device_get(
        fn(*jax.device_put(args), **jax.device_put(kwargs)))


def stateseq_stats(stateseqs, mask):
    """
    Get durations and frequencies for a batch of state sequences

    Parameters
    ----------
    stateseqs: ndarray
        Batch of state sequences where the last dim indexes time 

    mask: ndarray
        Binary indicator for which elements of ``stateseqs`` are valid,
        e.g. when state sequences of different lengths have been padded
        and stacked together in ``stateseqs``

    Returns
    -------
    frequency: ndarray, shape (max(stateseqs)+1,)
        The frequency of each syllable (not run-length encoded)

    durations: ndarray
        The duration of each each syllable (across all state sequences)

    """
    s = np.array(stateseqs.flatten()[mask[...,-stateseqs.shape[-1]:].flatten()>0])
    durations = np.array([sum(1 for i in g) for k,g in groupby(s)])
    frequency = np.bincount(s)
    return frequency, durations


def print_dims_to_explain_variance(pca, f):
    """
    Print the number of principal components requred to explain a given
    fraction of variance.

    Parameters
    ----------  
    pca: sklearn.decomposition._pca.PCA, A fit PCA model
    f: float, Target variance fraction
    """
    cs = np.cumsum(pca.explained_variance_ratio_)
    if cs[-1] < f: print(f'All components together only explain {cs[-1]*100}% of variance.')
    else: print(f'>={f*100}% of variance exlained by {(cs>f).nonzero()[0].min()+1} components.')


def concatenate_stateseqs(stateseqs, mask=None):
    """
    Concatenate state sequences, optionally applying a mask.

    Parameters
    ----------
    stateseqs: dict or ndarray, shape (..., t)
        Dictionary mapping names to 1d arrays, or a single
        multi-dimensional array representing a batch of state sequences
        where the last dim indexes time

    mask: ndarray, shape (..., >=t), default=None
        Binary indicator for which elements of ``stateseqs`` are valid,
        e.g. when state sequences of different lengths have been padded.
        If ``mask`` contains more time-points than ``stateseqs``, the
        initial extra time-points will be ignored.

    Returns
    -------
    stateseqs_flat: ndarray
        1d array containing all state sequences 
    """
    if isinstance(stateseqs, dict):
        stateseq_flat = np.hstack(list(stateseqs.values()))
    elif mask is not None:
        stateseq_flat = stateseqs[mask[:,-stateseqs.shape[1]:]>0]
    else: stateseq_flat = stateseqs.flatten()
    return stateseq_flat


def get_durations(stateseqs, mask=None):
    """
    Get durations for a batch of state sequences. For a more detailed 
    description of the function parameters, see 
    :py:func:`keypoint_moseq.util.concatenate_stateseqs`

    Parameters
    ----------
    stateseqs: dict or ndarray of shape (..., t)
    mask: ndarray of shape (..., >=t), default=None

    Returns
    -------
    durations: 1d array
        The duration of each each state (across all state sequences)

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])
    }
    >>> get_durations(stateseqs)
    array([2, 3, 1, 3, 1])

    """
    stateseq_flat = concatenate_stateseqs(stateseqs, mask=mask).astype(int)
    changepoints = np.insert(np.diff(stateseq_flat).nonzero()[0]+1,0,0)
    return changepoints[1:]-changepoints[:-1]


def get_frequencies(stateseqs, mask=None, num_states=None):
    """
    Get state frequencies for a batch of state sequences. Each frame is
    counted separately. For a more detailed  description of the function 
    parameters, see :py:func:`keypoint_moseq.util.concatenate_stateseqs`

    Parameters
    ----------
    stateseqs: dict or ndarray of shape (..., t)
    mask: ndarray of shape (..., >=t), default=None
    num_states: int, default=None
        Number of different states. If None, the number of states will
        be set to ``max(stateseqs)+1``.

    Returns
    -------
    frequencies: 1d array
        Proportion of frames in each state across all state sequences

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])
    }
    >>> get_frequencies(stateseqs)
    array([0.3, 0.3, 0.3, 0.1])

    """    
    stateseq_flat = concatenate_stateseqs(stateseqs, mask=mask).astype(int)
    return np.bincount(stateseq_flat, minlength=num_states)/len(stateseq_flat)

def reindex_by_frequency(stateseqs, mask=None):
    """
    Reindex a sequence of syllables by frequency. The most frequent
    syllable will be assigned 0, the second most frequent 1, etc.
    For a more detailed  description of the function parameters, 
    see :py:func:`keypoint_moseq.util.concatenate_stateseqs`

    Parameters
    ----------
    stateseqs: dict or ndarray of shape (..., t)
    mask: ndarray of shape (..., >=t), default=None

    Returns
    -------
    stateseqs_reindexed: ndarray
        The reindexed state sequences in the same format as ``stateseqs``
    """
    frequency = get_frequencies(stateseqs, mask=mask)
    o = np.argsort(np.argsort(frequency)[::-1])
    if isinstance(stateseqs, dict):
        stateseqs_reindexed = {k: o[seq] for k,seq in stateseqs.items()}
    else: stateseqs_reindexed = o[stateseqs]
    return stateseqs_reindexed


def list_files_with_exts(dir_path, ext_list, recursive=True):
    """
    This function lists all the files in a directory with one of the
    specified extensions.

    Parameters
    ----------
    dir_path : str
        The directory path to search.

    ext_list : list of str
        A list of file extensions to search for, including the dot 
        (e.g., '.txt').

    recursive : bool, default=True
        Whether to search for files recursively.

    Returns
    -------
    list
        A list of file paths.
    """
    ext_list += [ext.upper() for ext in ext_list]
    if recursive: return [
        os.path.join(root, name)
        for root, dirs, files in os.walk(dir_path)
        for name in files if os.path.splitext(name)[1] in ext_list]
    else: return [
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if os.path.splitext(name)[1] in ext_list]

def find_matching_videos(keys, video_dir, as_dict=False, 
                         video_extension=None, recursive=True):
    """
    Find video files for a set of session names. The filename of each
    video is assumed to be a prefix within the session name. For example
    given the following video directory::

        video_dir
        ├─ videoname1.avi
        └─ videoname2.avi
 
    the videos would be matched to session names as follows::

        >>> keys = ['videoname1blahblah','videoname2yadayada']
        >>> find_matching_videos(keys, video_dir, as_dict=True)

        {'videoname1blahblah': 'video_dir/videoname1.avi',
         'videoname2blahblah': 'video_dir/videoname2.avi'}
 
    Parameters
    -------
    keys: iterable
        Session names (as strings)

    video_dir: str
        Path to the video directory. 
        
    video_extension: str, default=None
        Extension of the video files. If None, videos are assumed to 
        have the one of the following extensions: "mp4", "avi", "mov"

    recursive: bool, default=True
        If True, search recursively for videos in subdirectories of
        `video_dir`.

    as_dict: bool, default=False
        Determines whether to return a dict mapping session names to 
        video paths, or a list of paths in the same order as `keys`.

    Returns
    -------
    video_paths: list or dict (depending on `as_dict`)
    """  

    if video_extension is None:
        extensions = ['.mp4','.avi','.mov']
    else: 
        if video_extension[0] != '.': 
            video_extension = '.'+video_extension
        extensions = [video_extension]

    videos = list_files_with_exts(video_dir, extensions, recursive=recursive)
    videos_to_paths = {os.path.splitext(os.path.basename(f))[0]:f for f in videos}

    video_paths = []
    for key in keys:
        matches = [path for video,path in videos_to_paths.items() 
                   if os.path.basename(key).startswith(video)]
        assert len(matches)>0, fill(f'No matching videos found for {key}')
        assert len(matches)<2, fill(f'More than one video matches {key} ({matches})')
        video_paths.append(matches[0])
    if as_dict: return dict(zip(sorted(keys),video_paths))
    else: return video_paths


def pad_along_axis(arr, pad_widths, axis=0, value=0):
    """
    Pad an array along a single axis

    Parameters
    -------
    arr: ndarray, Array to be padded
    pad_widths: tuple (int,int), Amount of padding on either end
    axis: int, Axis along which to add padding
    value: float, Value of padded array elements

    Returns
    _______
    padded_arr: ndarray

    """
    pad_widths_full = [(0,0)]*len(arr.shape)
    pad_widths_full[axis] = pad_widths
    padded_arr = np.pad(arr, pad_widths_full, constant_values=value)
    return padded_arr


def unbatch(data, labels): 
    """
    Invert :py:func:`keypoint_moseq.util.batch`
 
    Parameters
    -------
    data: ndarray, shape (num_segs, seg_length, ...)
        Stack of segmented time-series

    labels: tuples (str,int,int)
        Labels for the rows of ``data`` as tuples with the form
        (name,start,end)

    Returns
    -------
    data_dict: dict
        Dictionary mapping names to reconstructed time-series
    """     
    data_dict = {}
    keys = sorted(set([key for key,start,end in labels]))    
    for key in keys:
        length = np.max([e for k,s,e in labels if k==key])
        seq = np.zeros((int(length),*data.shape[2:]), dtype=data.dtype)
        for (k,s,e),d in zip(labels,data):
            if k==key: seq[s:e] = d[:e-s]
        data_dict[key] = seq
    return data_dict


def batch(data_dict, keys=None, seg_length=None, seg_overlap=30):
    """
    Stack time-series data of different lengths into a single array for
    batch processing, optionally breaking up the data into fixed length 
    segments. Data is 0-padded so that the stacked array isn't ragged.

    Parameters
    -------
    data_dict: dict {str : ndarray}
        Dictionary mapping names to ndarrays, where the first dim
        represents time. All data arrays must have the same shape except
        for the first dim. 

    keys: list of str, default=None
        Optional list of names specifying which datasets to include in 
        the output and what order to put them in. Each name must be a 
        key in ``data_dict``. If ``keys=None``, names will be sorted 
        alphabetically.

    seg_length: int, default=None
        Break each time-series into segments of this length. If 
        ``seg_length=None``, the final stacked array will be as long
        as the longest time-series. 

    seg_overlap: int, default=30
        Amount of overlap between segments. For example, setting
        ``seg_length=N`` and ``seg_overlap=M`` will result in segments
        with start/end times (0, N+M), (N, 2*N+M), (2*N, 3*N+M),...

    Returns
    -------
    data: ndarray, shape (N, seg_length, ...)
        Stacked data array

    mask: ndarray, shape (N, seg_length)
        Binary indicator specifying which elements of ``data`` are not
        padding (``mask==0`` in padded locations)

    keys: list of tuples (str,int), length N
        Row labels for ``data`` consisting (name, segment_num) pairs

    """
    if keys is None: keys = sorted(data_dict.keys())
    Ns = [len(data_dict[key]) for key in keys]
    if seg_length is None: seg_length = np.max(Ns)
        
    stack,mask,labels = [],[],[]
    for key,N in zip(keys,Ns):
        for start in range(0,N,seg_length):
            arr = data_dict[key]
            end = min(start+seg_length+seg_overlap, N)
            pad_length = seg_length+seg_overlap-(end-start)
            padding = np.zeros((pad_length,*arr.shape[1:]), dtype=arr.dtype)
            mask.append(np.hstack([np.ones(end-start),np.zeros(pad_length)]))
            stack.append(np.concatenate([arr[start:end],padding],axis=0))
            labels.append((key,start,end))

    stack = np.stack(stack)
    mask = np.stack(mask)
    return stack,mask,labels


    
def filter_angle(angles, size=9, axis=0):
    """
    Perform median filtering on time-series of angles by transforming to
    a (cos,sin) representation, filtering in R^2, and then transforming 
    back into angle space. 

    Parameters
    -------
    angles: ndarray, Array of angles (in radians)
    size: int, default=9, Size of the filtering kernel
    axis: int, default=0, Axis along which to filter

    Returns
    -------
    filtered_angles: ndarray
    """
    kernel = np.where(np.arange(len(angles.shape))==axis, size, 1)
    return np.arctan2(median_filter(np.sin(angles), kernel),
                      median_filter(np.cos(angles), kernel))


def filter_centroids_headings(centroids, headings, filter_size=9):
    """
    Perform median filtering on centroids and headings

    Parameters
    -------
    centroids: dict {str : ndarray, shape (t,2)}
        Centroids stored as a dictionary mapping session names to 
        ndarrays, where the first dim represents time

    headings: dict {str : 1d array }
        Headings stored as a dictionary mapping session names to
        1d arrays representing an angle in radians

    filter_size: int, default=9
        Kernel size for median filtering

    Returns
    -------
    filtered_centroids: dict
    filtered_headings: dict
    """    
    centroids = {k:median_filter(v,(filter_size,1)) for k,v in centroids.items()}
    headings = {k:filter_angle(v, size=filter_size) for k,v in headings.items()}  
    return centroids, headings


def get_syllable_instances(stateseqs, min_duration=3, pre=30, post=60,
                           min_frequency=0, min_instances=0):
    """
    Map each syllable to a list of instances when it occured. Only 
    include instances that meet the criteria specified by ``pre``, 
    ``post``, and ``min_duration``. Only include syllables that meet the
    criteria specified by ``min_frequency`` and ``min_instances``. 

    Parameters
    -------
    stateseqs: dict {str : 1d array}
        Dictionary mapping names to syllable sequences

    min_duration: int, default=3
        Mininum duration for inclusion of a syllable instance

    pre: int, default=30
        Syllable instances that start before this location in the state
        sequence will be excluded

    post: int, default=60
        Syllable instances that end after this location in the state
        sequence will be excluded

    min_frequency: int, default=0
        Minimum allowed frequency (across all state sequences) for
        inclusion of a syllable

    min_instances: int, default=0
        Minimum number of instances (across all state sequences) for
        inclusion of a syllable

    Returns
    -------
    syllable_instances: dict
        Dictionary mapping each syllable to a list of instances. Each
        instance is a tuple (name,start,end) representing subsequence
        ``stateseqs[name][start:end]``.
    """
    num_syllables = int(max(map(max,stateseqs.values()))+1)
    syllable_instances = [[] for syllable in range(num_syllables)]
    
    for key,stateseq in stateseqs.items():
        transitions = np.nonzero(stateseq[1:] != stateseq[:-1])[0]+1
        starts = np.insert(transitions,0,0)
        ends = np.append(transitions,len(stateseq))
        for (s,e,syllable) in zip(starts,ends,stateseq[starts]):
            if (e-s >= min_duration and s>=pre and s<len(stateseq)-post): 
                syllable_instances[syllable].append((key,s,e))
                
    frequencies_filter = get_frequencies(stateseqs) >= min_frequency
    counts_filter = np.array(list(map(len,syllable_instances))) >= min_instances
    use_syllables = np.all([frequencies_filter, counts_filter],axis=0).nonzero()[0]
    return {syllable : syllable_instances[syllable] for syllable in use_syllables} 


def get_edges(use_bodyparts, skeleton):
    """
    Represent the skeleton as a list of index-pairs.

    Parameters
    -------
    use_bodyparts: list
        Bodypart names

    skeleton: list
        Pairs of bodypart names as tuples (bodypart1,bodypart2)

    Returns
    -------
    edges: list
        Pairs of indexes representing the enties of ``skeleton``
    """
    edges = []
    for bp1,bp2 in skeleton:
        if bp1 in use_bodyparts and bp2 in use_bodyparts:
            edges.append([use_bodyparts.index(bp1),use_bodyparts.index(bp2)])
    return edges

        
def reindex_by_bodyparts(data, bodyparts, use_bodyparts, axis=1):
    """
    Use an ordered list of bodyparts to reindex keypoint coordinates

    Parameters
    -------
    data: dict or ndarray
        A single array of keypoint coordinates or a dict mapping from 
        names to arrays of keypoint coordinates

    bodyparts: list
        Label for each keypoint represented in ``data``

    use_bodyparts: list
        Ordered subset of keypoint labels

    axis: int, default=1
        The axis in ``data`` that represents keypoints. It is required
        that ``data.shape[axis]==len(bodyparts)``. 

    Returns
    -------
    reindexed_data: ndarray or dict
        Keypoint coordinates in the same form as ``data`` with
        reindexing applied
    """
    ix = np.array([bodyparts.index(bp) for bp in use_bodyparts])
    if isinstance(data, np.ndarray): return np.take(data, ix, axis)
    else: return {k: np.take(v, ix, axis) for k,v in data.items()}


def get_trajectories(syllable_instances, coordinates, pre=0, post=None, 
                     centroids=None, headings=None, filter_size=9):
    """
    Extract keypoint trajectories for a given collection of syllable
    instances. If centroids and headings are provided, each trajectory
    is transformed into the ego-centric reference frame from the moment 
    of syllable onset. When ``post`` is not None, trajectories will 
    all terminate a fixed number of frames after syllable onset. 

    Parameters
    -------
    syllable_instances: dict
        Mapping from each syllable to a list of instances, where each
        instance is a tuple of the form (name,start,end)

    coordinates: dict
        Dictionary mapping names to coordinates, formatted as ndarrays
        with shape (num_frames, num_keypoints, d)

    pre: int, default=0
        Number of frames to include before syllable onset

    post: int, defualt=None
        Determines the length of the trajectory. When ``post=None``,
        the trajectory terminates at the end of the syllable instance.
        Otherwise the trajectory terminates at a fixed number of frames
        after syllable (where the number is determined by ``post``).

    centroids: dict, default=None
        Dictionary with the same keys as ``coordinates`` mapping each
        name to an ndarray with shape (num_frames, d)

    headings: dict, default=None
        Dictionary with the same keys as ``coordinates`` mapping each
        name to a 1d array of heading angles in radians

    filter_size: int, default=9
        Size of median filter applied to ``centroids`` and ``headings``

    Returns
    -------
    trajectories: dict
        Dictionary similar to ``syllable_instances``, but now mapping
        each syllable to a list or array of trajectories (a list is used
        when ``post=None``, else an array)
    """
    if centroids is not None and headings is not None:
        centroids,headings = filter_centroids_headings(
            centroids, headings, filter_size=filter_size)
        
    trajectories = {}
    for syllable,instances in syllable_instances.items():
        if post is None:
            X = [coordinates[key][s-pre:e] for key,s,e in instances]
            if centroids is not None and headings is not None:
                X = [np_io(inverse_rigid_transform)(
                        x,centroids[key][s],headings[key][s]
                     ) for x,(key,s,e) in zip(X,instances)]
        else:
            X = np.array([coordinates[key][s-pre:s+post] for key,s,e in instances])
            if centroids is not None and headings is not None:
                c = np.array([centroids[key][s] for key,s,e in instances])[:,None]
                h = np.array([headings[key][s] for key,s,e in instances])[:,None]
                X = np_io(inverse_rigid_transform)(X,c,h)
        trajectories[syllable] = X

    return trajectories



def sample_instances(syllable_instances, num_samples, mode='random', 
                     pca_samples=50000, pca_dim=4, n_neighbors=50,
                     coordinates=None, pre=5, post=15, centroids=None, 
                     headings=None, filter_size=9):
    """
    Sample a fixed number of instances for each syllable.

    Parameters
    ----------
    syllable_instances: dict
        Mapping from each syllable to a list of instances, where each
        instance is a tuple of the form (name,start,end)

    num_samples: int
        Number of samples return for each syllable

    mode: str, {'random', 'density'}, default='random'
        Sampling method to use. Options are:
        
        - 'random': Instances are chosen randomly (without replacement)
        - 'density': For each syllable, a syllable-specific density function is
          computed in trajectory space and compared to the overall
          density across all syllables. An exemplar instance that 
          maximizes this ratio is chosen for each syllable, and
          its nearest neighbors are randomly sampled. 

    pca_samples: int, default=50000
        Number of trajectories to sample when fitting a PCA model for 
        density estimation (used when ``mode='density'``)

    pca_dim: int, default=4
        Number of principal components to use for density estimation
        (used when ``mode='density'``)

    n_neighbors: int, defualt=50
        Number of neighbors to use for density estimation and for 
        sampling the neighbors of the examplar syllable instance
        (used when ``mode='density'``)

    coordinates, pre, pos, centroids, heading, filter_size
        Passed to :py:func:`keypoint_moseq.util.get_trajectories`

    Returns
    -------
    sampled_instances: dict
        Dictionary in the same format as ``syllable_instances`` mapping
        each syllable to a list of sampled instances.
    """
    assert mode in ['random','density']
    assert all([len(v)>=num_samples for v in syllable_instances.values()])
    assert n_neighbors>=num_samples

    if mode=='random':
        sampled_instances = {syllable: [instances[i] for i in np.random.choice(
            len(instances), num_samples, replace=False
        )] for syllable,instances in syllable_instances.items()}
        return sampled_instances
    
    elif mode=='density':
        assert not (coordinates is None or headings is None or centroids is None), fill(
            '``coordinates``, ``headings`` and ``centroids`` are required when '
            '``mode == "density"``')

        trajectories = get_trajectories(
            syllable_instances, coordinates, pre=pre, post=post, 
            centroids=centroids, headings=headings, filter_size=filter_size)
        X = np.vstack(list(trajectories.values()))

        if X.shape[0]>pca_samples: 
            X = X[np.random.choice(X.shape[0], pca_samples, replace=False)]

        pca = PCA(n_components=pca_dim).fit(X.reshape(X.shape[0],-1))
        Xpca = pca.transform(X.reshape(X.shape[0],-1))
        all_nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(Xpca)
        
        sampled_instances = {} 

        for syllable,X in trajectories.items():
            
            Xpca = pca.transform(X.reshape(X.shape[0],-1))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(Xpca)
            distances, indices = nbrs.kneighbors(Xpca)
            local_density = 1/distances.mean(1)
            
            distances, _ = all_nbrs.kneighbors(Xpca)
            global_density = 1/distances.mean(1)
            exemplar = np.argmax(local_density/global_density)  
            samples = np.random.choice(indices[exemplar], num_samples, replace=False)      
            sampled_instances[syllable] = [syllable_instances[syllable][i] for i in samples]
         
        return sampled_instances

    else:
        raise ValueError('Invalid mode: {}'.format(mode))


def interpolate_keypoints(coordinates, outliers):
    """
    Use linear interpolation to impute the coordinates of outliers.
    
    Parameters
    ----------
    coordinates : ndarray of shape (num_frames, num_keypoints, dim)
        Keypoint observations.
    outliers : ndarray of shape (num_frames, num_keypoints)
        Binary indicator whose true entries are outlier points.
        
    Returns
    -------
    interpolated_coordinates : ndarray with same shape as ``coordinates``
        Keypoint observations with outliers imputed.
    """  

    interpolated_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[1]):
        x = np.arange(coordinates.shape[0])
        xp = x[~outliers[:,i]]
        for j in range(coordinates.shape[2]):
            fp = coordinates[:,i,j][~outliers[:,i]]
            interpolated_coordinates[:,i,j] = np.interp(x,xp,fp)

    return interpolated_coordinates


