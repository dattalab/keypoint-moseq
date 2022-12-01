import numpy as np
import os
import tqdm
from textwrap import fill
from numba import njit, prange
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
na = jnp.newaxis

def jax_io(fn): 
    """
    Decorator that converts arguments from jax arrays into numpy arrays,
    applies the function, then converts the outputs back to jax arrays.

    """
    return lambda *args, **kwargs: jax.device_put(
        fn(*jax.device_get(args), **jax.device_get(kwargs)))

def np_io(fn): 
    """
    Decorator that converts arguments from numpy arrays into jax arrays,
    applies the function, then converts the outputs back to numpy arrays.

    """
    return lambda *args, **kwargs: jax.device_get(
        fn(*jax.device_put(args), **jax.device_put(kwargs)))


@njit
def count_transitions(num_states, stateseqs, mask):
    """
    Count all transitions in `stateseqs` where the start and end
    states both have `mask>0`. The last dim of `stateseqs` indexes time. 

    Parameters
    ----------
    num_states: int
        Total number of states: must be at least ``max(stateseqs)+1``

    stateseqs: ndarray
        Batch of state sequences where the last dim indexes time 

    mask: ndarray
        Binary indicator for which elements of ``stateseqs`` are valid,
        e.g. when state sequences of different lengths have been padded

    Returns
    -------
    counts: ndarray, shape (num_states,num_states)
        The number of transitions between every pair of states

    """    
    counts = np.zeros((num_states,num_states))
    for i in prange(mask.shape[0]):
        for j in prange(mask.shape[1]-1):
            if not (   
               mask[i,j]==0 or mask[i,j+1]==0 
               or np.isnan(stateseqs[i,j]) 
               or np.isnan(stateseqs[i,j+1])
            ): counts[stateseqs[i,j],stateseqs[i,j+1]] += 1
    return counts

    
def estimate_coordinates(*, v, h, x, Cd, **kwargs):
    """
    Estimate keypoint coordinates obtained from projecting the 
    latent state ``x`` into keypoint-space (via ``Cd``) and then
    rotating and translating by ``h`` and `v`` respectively

    Parameters
    ----------
    v: jax array, shape (...,t,d), Centroid locations
    h: jax array, shape (...,t), Heading
    x: jax array, shape (...,t,D), Continuous latent state
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation

    Returns
    -------
    Yest: jax array, shape (...,t,k,d), Estimated coordinates
        
    """    
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    Yest = affine_transform(Ybar,v,h)
    return Yest


@jax_io
def interpolate(keypoints, outliers, axis=1):
    """
    Use linear interpolation to impute the coordinates of outliers

    Parameters
    ----------
    keypoints: jax array, shape (...,t,k,d)
        Keypoint coordinates 

    outliers: jax array, shape (...,t,k)
        Binary indicator array where 1 implies that the corresponding 
        keypoint is an outlier.

    Returns
    -------
    interpolated_keypoints: jax array, shape (...,t,k,d)
        Copy of ``keypoints`` where outliers have been replaced by
        linearly interpolated values.

    """   
    keypoints = np.moveaxis(keypoints, axis, 0)
    init_shape = keypoints.shape
    keypoints = keypoints.reshape(init_shape[0],-1)
    
    outliers = np.moveaxis(outliers, axis, 0)
    outliers = np.repeat(outliers[...,None],init_shape[-1],axis=-1)
    outliers = outliers.reshape(init_shape[0],-1)
    
    interp = lambda x,xp,fp: (
        np.ones_like(x)*x.mean() if len(xp)==0 else np.interp(x,xp,fp))
    
    keypoints = np.stack([interp(
        np.arange(init_shape[0]), 
        np.nonzero(~outliers[:,i])[0],
        keypoints[:,i][~outliers[:,i]]
    ) for i in range(keypoints.shape[1])], axis=1)     
    return np.moveaxis(keypoints.reshape(init_shape),0,axis)


def center_embedding(k):
    """
    Generates a matrix ``Gamma`` that maps from a (k-1)-dimensional 
    vector space  to the space of k-tuples with zero mean

    Parameters
    ----------
    k: int, Number of keypoints

    Returns
    -------
    Gamma: jax array, shape (k, k-1)

    """  
    # using numpy.linalg.svd because jax version crashes on windows
    return jnp.array(np.linalg.svd(np.eye(k) - np.ones((k,k))/k)[0][:,:-1])


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


def ensure_symmetric(X):
    """
    Ensure that a batch of matrices are symmetric by taking the average
    of each with its transpose. 

    """
    XT = jnp.swapaxes(X,-1,-2)
    return (X+XT)/2


def vector_to_angle(V):
    """Convert 2D vectors to angles in [-pi, pi]. The vector (1,0)
    corresponds to angle of 0. If V is n-dinmensional, the first
    n-1 dimensions are treated as batch dims.     
    """
    return jnp.arctan2(V[...,1],V[...,0])

    
def angle_to_rotation_matrix(h, d=3):
    """Create rotation matrices from an array of angles. If ``d > 2`` 
    then rotation is performed in the first two dims.

    Parameters
    ----------
    h: jax array, shape (*dims)
        Angles (in radians)

    d: int, default=3
        Dimension of each rotation matrix

    Returns
    -------
    m: ndarray, shape (*dims, d, d)
        Stacked rotation matrices 
        
    """
    m = jnp.tile(jnp.eye(d), (*h.shape,1,1))
    m = m.at[...,0,0].set(jnp.cos(h))
    m = m.at[...,1,1].set(jnp.cos(h))
    m = m.at[...,0,1].set(-jnp.sin(h))
    m = m.at[...,1,0].set(jnp.sin(h))
    return m


@jax.jit
def affine_transform(Y, v, h):
    """
    Apply the following affine transform
    
    .. math::
        Y \mapsto R(h) @ Y + v

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    v: jax array, shape (*dims, d), Translations
    h: jax array, shape (*dims), Heading angles
          
    Returns
    -------
    Ytransformed: jax array, shape (*dims, k, d)
        
    """
    rot_matrix = angle_to_rotation_matrix(h, d=Y.shape[-1])
    Ytransformed = (Y[...,na,:]*rot_matrix[...,na,:,:]).sum(-1) + v[...,na,:]
    return Ytransformed

@jax.jit
def inverse_affine_transform(Y, v, h):
    """
    Apply the following affine transform
    
    .. math::
        Y \mapsto R(-h) @ (Y - v)

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    v: jax array, shape (*dims, d), Translations
    h: jax array, shape (*dims), Heading angles
          
    Returns
    -------
    Ytransformed: jax array, shape (*dims, k, d)
        
    """
    rot_matrix = angle_to_rotation_matrix(-h, d=Y.shape[-1])
    Y_transformed = ((Y-v[...,na,:])[...,na,:]*rot_matrix[...,na,:,:]).sum(-1)
    return Y_transformed

def transform_data_for_pca(Y, **kwargs):
    """
    Prepare keypoint coordinates for PCA by performing egocentric 
    alignment, changing basis using ``center_embedding(k)``, and 
    reshaping to a single flat vector per frame. 

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d), Keypoint coordinates
    v: jax array, shape (*dims, d), Translations
    h: jax array, shape (*dims), Heading angles

    kwargs: must include
        ``use_bodyparts``
        ``anterior_bodyparts``
        ``posterior_bodyparts``
          
    Returns
    -------
    Y_flat: jax array, shape (*dims, (k-1)*d)
        
    """
    k,d = Y.shape[-2:]
    Y_aligned = align_egocentric(Y, **kwargs)[0]
    Y_embedded = center_embedding(k).T @ Y_aligned
    Y_flat = Y_embedded.reshape(*Y.shape[:-2],(k-1)*d)
    return Y_flat


def fit_pca(*, Y, conf, mask, conf_threshold=0.5, verbose=False,
            PCA_fitting_num_frames=1000000, **kwargs):
    """
    Fit a PCA model to transformed keypoint coordinates. If ``conf`` is
    not None, perform linear interpolation over outliers defined by
    ``conf < conf_threshold``.

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d) where d >= 2
        Keypoint coordinates

    conf: None or jax array, shape (*dims,k)
        Confidence value for each keypoint

    mask: jax array, shape (*dims, k)
        Binary indicator for which elements of ``Y`` are valid

    conf_threshold: float, default=0.5
        Confidence threshold below which keypoints will be interpolated

    PCA_fitting_num_frames: int, default=1000000
        Maximum number of frames to use for PCA. Frames will be sampled
        randomly if the input data exceed this size. 

    kwargs: must include
        ``use_bodyparts``
        ``anterior_bodyparts``
        ``posterior_bodyparts``

    Returns
    -------
    pca, sklearn.decomposition._pca.PCA
        A fit sklearn PCA model

    """
    if conf is not None: 
        if verbose: print('PCA: Interpolating low-confidence detections')
        Y = interpolate(Y, conf<conf_threshold)
       
    if verbose: print('PCA: Performing egocentric alignment')
    Y_flat = transform_data_for_pca(Y, **kwargs)[mask>0]
    PCA_fitting_num_frames = min(PCA_fitting_num_frames, Y_flat.shape[0])
    Y_sample = np.array(Y_flat)[np.random.choice(
        Y_flat.shape[0],PCA_fitting_num_frames,replace=False)]
    if verbose: print(f'PCA: Fitting PCA model on {Y_sample.shape[0]} sample poses')
    return PCA(n_components=Y_flat.shape[-1]).fit(Y_sample)


def align_egocentric(Y, *, use_bodyparts, anterior_bodyparts, 
                     posterior_bodyparts, **kwargs):
    """
    Perform egocentric alignment of keypoints by translating the 
    centroid to the origin and rotatating so that the vector pointing
    from the posterior bodyparts toward the anterior bodyparts is 
    proportional to (1,0) 

    Parameters
    ----------   
    Y: jax array, shape (*dims, k, d) where d >= 2
        Keypoint coordinates

    use_bodyparts: list of str, length k
        List of bodypart names corresponding to the second-to-last 
        dimension of ``Y``

    anterior_bodyparts: list of str, subset of ``use_bodyparts``
        List of bodyparts defining the anterior part of the animal

    posterior_bodyparts: list of str, subset of ``use_bodyparts``
        List of bodyparts defining the posterior part of the animal

    Returns
    -------
    Y_aligned: jax array, shape (*dims, k, d)
        Aligned keypoint coordinates

    v: jax array, shape (*dims, d)
        Centroid positions that were used for alignment

    h: jax array, shape (*dims)
        Heading angles that were used for alignment

    """
    anterior_keypoints = jnp.array([use_bodyparts.index(bp) for bp in anterior_bodyparts])
    posterior_keypoints = jnp.array([use_bodyparts.index(bp) for bp in posterior_bodyparts])
    
    posterior_loc = Y[..., posterior_keypoints,:2].mean(-2) 
    anterior_loc = Y[..., anterior_keypoints,:2].mean(-2) 
    
    h = vector_to_angle(anterior_loc-posterior_loc)
    v = Y.mean(-2).at[...,2:].set(0)
    Y_aligned = inverse_affine_transform(Y,v,h)
    return Y_aligned,v,h


def pad_affine(x):
    """
    Pad ``x`` with 1's so that it can be affine transformed with matrix
    multiplication. 
    """
    padding = jnp.ones((*x.shape[:-1],1))
    xpadded = jnp.concatenate((x,padding),axis=-1)
    return xpadded



def get_lags(x, nlags):
    """
    Get lags of a multivariate time series. Lags are concatenated along
    the last dim in time-order. Writing the last two dims of ``x`` as

    .. math::
        \begin{bmatrix} 
            x_0    \\
            x_1    \\
            \vdots \\
            x_{t}  \\
        \end{bmatrix}

    the output of this function with ``nlags=3`` would be

    .. math::
        \begin{bmatrix} 
            x_0     & x_1     & x_2    \\
            x_1     & x_2     & x_3    \\
            \vdots  & \vdots  & \vdots \\
            x_{t-3} & x_{t-2} & x_{t-1}
            \vdots
        \end{bmatrix}  

    
    Parameters
    ----------  
    nlags: int
        Number of lags
        
    x: jax array, shape (*dims, t, d)
        Batch of d-dimensional time series 
    
    Returns
    -------
    x_lagged: jax array, shape (*dims, t-nlags, d*nlags)

    """
    lags = [jnp.roll(x,t,axis=-2) for t in range(1,nlags+1)]
    return jnp.concatenate(lags[::-1],axis=-1)[...,nlags:,:]


def ar_to_lds(As, bs, Qs, Cs):
    """
    Reformat a linear dynamical system with L'th-order autoregressive 
    (AR) dynamics in R^D as a system with 1st-order dynamics in R^(D*L)
    
    Parameters
    ----------  
    As: jax array, shape (*dims, D, D*L),   AR transformation
    bs: jax array, shape (*dims, D),        AR affine term
    Qs: jax array, shape (*dims, D, D),     AR noise covariance
    Cs: jax array, shape (*dims, D_obs, D), obs transformation
    
    Returns
    -------
    As_: jax array, shape (*dims, D*L, D*L)
    bs_: jax array, shape (*dims, D*L)    
    Qs_: jax array, shape (*dims, D*L, D*L)  
    Cs_: jax array, shape (*dims, D_obs, D*L)

    """    
    D,L = As.shape[-2],As.shape[-1]//As.shape[-2]

    As_ = jnp.zeros((*As.shape[:-2],D*L,D*L))
    As_ = As_.at[...,:-D,D:].set(jnp.eye(D*(L-1)))
    As_ = As_.at[...,-D:,:].set(As)
    
    Qs_ = jnp.zeros((*Qs.shape[:-2],D*L,D*L))
    Qs_ = Qs_.at[...,:-D,:-D].set(jnp.eye(D*(L-1))*1e-2)
    Qs_ = Qs_.at[...,-D:,-D:].set(Qs)
    
    bs_ = jnp.zeros((*bs.shape[:-1],D*L))
    bs_ = bs_.at[...,-D:].set(bs)
    
    Cs_ = jnp.zeros((*Cs.shape[:-1],D*L))
    Cs_ = Cs_.at[...,-D:].set(Cs)
    return As_, bs_, Qs_, Cs_


def gaussian_log_prob(x, mu, sigma_inv):
    """
    Log probability Gaussian distrubution at a point
    
    Parameters
    ----------  
    x: jax array, shape (*dims,d)
        Point at which to evaluate the log probability

    mu: jax array, shape (*dims,d)
        Mean of the Gaussian distribution

    sigma_inv: jax array, shape (*dims,d,d) 
        Inverse covariance of the Gaussian distribution

    Returns
    -------
    log_probability: jax array, shape (*dims)

    """     
    return (-((mu-x)[...,na,:]*sigma_inv*(mu-x)[...,:,na]).sum((-1,-2))/2
            +jnp.log(jnp.linalg.det(sigma_inv))/2)


def latent_log_prob(*, x, z, Ab, Q, **kwargs):
    """
    Calculate the log probability of the trajectory ``x`` at each time 
    step, given switching autoregressive (AR) parameters

    Parameters
    ----------  
    x: jax array, shape (*dims,t,D)
        Continuous latent trajectories in R^D of length t

    z: jax array, shape (*dims,t)
        Discrete state sequences of length t

    Ab: jax array, shape (N,D*L+1) 
        AR transforms (including affine term) for each of N discrete
        states, where D is the dimension of the latent states and 
        L is the the order of the AR process

    Q: jax array, shape (N,D,D) 
        AR noise covariance for each of N discrete states

    Returns
    -------
    log_probability: jax array, shape (*dims,t-L)

    """

    Qinv = jnp.linalg.inv(Q)
    Qdet = jnp.linalg.det(Q)
    
    L = Ab.shape[-1]//Ab.shape[-2]
    x_lagged = get_lags(x, N)
    x_pred = (Ab[z] @ pad_affine(x_lagged)[...,na])[...,0]
    
    d = x_pred - x[:,nlags:]
    return (-(d[...,na,:]*Qinv[z]*d[...,:,na]).sum((2,3))/2
            -jnp.log(Qdet[z])/2  -jnp.log(2*jnp.pi)*Q.shape[-1]/2)


def stateseq_log_prob(*, z, pi, **kwargs):
    """
    Calculate the log probability of a discrete state sequence at each
    time-step given a matrix of transition probabilities

    Parameters
    ----------  
    z: jax array, shape (*dims,t)
        Discrete state sequences of length t

    pi: jax array, shape (N,N)
        Transition probabilities

    Returns
    -------
    log_probability: jax array, shape (*dims,t-1)

    """
    return jnp.log(pi[z[...,:-1],z[...,1:]])


def scale_log_prob(*, s, s_0, nu_s, **kwargs):
    """
    Calculate the log probability of the noise scale for each keypoint
    given the noise prior, which is a scaled inverse chi-square 

    Parameters
    ----------  
    s: jax array, shape (*dims)
        Noise scale for each keypoint at each time-step

    s_0: float or jax array, shape (*dims)
        Prior on noise scale - either a single universal value or a 
        separate prior for each keypoint at each time-step

    nu_s: int
        Degrees of freedom

    Returns
    -------
    log_probability: jax array, shape (*dims)

    """
    return -nu_s*s_0 / s / 2 - (1+nu_s/2)*jnp.log(s)

    
def location_log_prob(*, v, sigmasq_loc):
    """
    Calculate the log probability of the centroid location at each 
    time-step, given the prior on centroid movement

    Parameters
    ----------  
    v: jax array, shape (*dims,t,d)
        Location trajectories in R^d of length t

    sigmasq_loc: float
        Assumed variance in centroid displacements

    Returns
    -------
    log_probability: jax array, shape (*dims,t-1)

    """
    d = v[:,:-1]-v[:,1:]
    return (-(d**2).sum(-1)/sigmasq_loc/2 
            -v.shape[-1]/2*jnp.log(sigmasq_loc*2*jnp.pi))


def obs_log_prob(*, Y, x, s, v, h, Cd, sigmasq, **kwargs):
    """
    Calculate the log probability of keypoint coordinates at each
    time-step, given continuous latent trajectories, centroids, heading
    angles, noise scales, and observation parameters

    Parameters
    ----------  
    Y: jax array, shape (*dims,k,d), Keypoint coordinates
    x: jax array, shape (*dims,D), Latent trajectories
    s: jax array, shape (*dims,k), Noise scales
    v: jax array, shape (*dims,d), Centroids
    h: jax array, shape (*dims), Heading angles
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    sigmasq: jax array, shape (k,), Unscaled noise for each keypoint

    Returns
    -------
    log_probability: jax array, shape (*dims,k)

    """
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    sqerr = ((Y - affine_transform(Ybar,v,h))**2).sum(-1)
    return (-1/2 * sqerr/s/sigmasq - d/2 * jnp.log(2*s*sigmasq*jnp.pi))


@jax.jit
def log_joint_likelihood(*, Y, mask, x, s, v, h, z, pi, Ab, Q, Cd, sigmasq, sigmasq_loc, s_0, nu_s, **kwargs):
    """
    Calculate the total log probability for each latent state

    Parameters
    ----------  
    Y: jax array, shape (*dims,k,d), Keypoint coordinates
    mask: jax array, shape (*dims), Binary indicator for valid frames
    x: jax array, shape (*dims,D), Latent trajectories
    s: jax array, shape (*dims,k), Noise scales
    v: jax array, shape (*dims,d), Centroids
    h: jax array, shape (*dims), Heading angles
    z: jax array, shape (*dims), Discrete state sequences
    pi: jax array, shape (N,N), Transition probabilities
    Ab: jax array, shape (N,D*L+1), Autoregressive transforms
    Q: jax array, shape (D,D), Autoregressive noise covariances
    Cd: jax array, shape ((k-1)*d, D-1), Observation transformation
    sigmasq: jax array, shape (k,), Unscaled noise for each keypoint
    sigmasq_loc: float, Assumed variance in centroid displacements
    s_0: float or jax array, shape (*dims,k), Prior on noise scale
    nu_s: int, Degrees of freedom in noise prior

    Returns
    -------
    log_probabilities: dict
        Dictionary mapping the name of each latent state variables to
        its total log probability

    """
    nlags = Ab.shape[-1]//Ab.shape[-2]
    return {
        'Y': (obs_log_prob(Y=Y, x=x, s=s, v=v, h=h, Cd=Cd, sigmasq=sigmasq)*mask[...,na]).sum(),
        'x': (latent_log_prob(x=x, z=z, Ab=Ab, Q=Q)*mask[...,nlags:]).sum(),
        'z': (stateseq_log_prob(z=z, pi=pi)*mask[...,nlags+1:]).sum(),
        'v': (location_log_prob(v=v, sigmasq_loc=sigmasq_loc)*mask[...,1:]).sum(),
        's': (scale_log_prob(s=s, nu_s=nu_s, s_0=s_0)*mask[...,na]).sum()}


def merge_data(data_dict, keys=None, seg_length=None):
    """
    Stack time-series data of different lengths into a single array,
    optionally breaking up the data into fixed length segments. Data is
    0-padded so that the stacked array isn't ragged. 

    Parameters
    ----------   
    data_dict: dict {str : ndarray}
        Dictionary mapping names to ndarrays, where the first dim
        represents. All data arrays must have the same shape except
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

    Returns
    -------
    data: ndarray, shape (N, seg_length, *dims)
        Stacked data array

    mask: ndarray, shape (N, seg_length)
        Binary indicator specifying which elements of ``data`` are not
        padding (i.e. padding is where ``mask==0``)

    keys: list of tuples (str,int), length N
        Row labels for ``data`` consisting (name, segment_num) pairs

    """

    if keys is None: keys = sorted(data_dict.keys())
    max_length = np.max([data_dict[k].shape[0] for k in keys])
    if seg_length is None: seg_length = max_length
        
    def reshape(x):
        padding = (-x.shape[0])%seg_length
        x = np.concatenate([x, np.zeros((padding,*x.shape[1:]))],axis=0)
        return x.reshape(-1, seg_length, *x.shape[1:])
    
    data = np.concatenate([reshape(data_dict[k]) for k in keys],axis=0)
    mask = np.concatenate([reshape(np.ones(data_dict[k].shape[0])) for k in keys],axis=0)
    labels = [(k,i) for k in keys for i in range(int(np.ceil(len(data_dict[k])/seg_length)))]
    return data, mask, labels


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
    stateseqs: dict or ndarray, shape (*dims, t)
        Dictionary mapping names to 1d arrays, or a single
        multi-dimensional array representing a batch of state sequences
        where the last dim indexes time

    mask: ndarray, shape (*dims, >=t), default=None
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
    Get durations for a batch of state sequences. For a description of 
    the inputs, see :py:func:`keypoint_moseq.util.concatenate_stateseqs`

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


def get_frequencies(stateseqs, mask=None):
    """
    Get state frequencies for a batch of state sequences. Each frame is
    counted separately. For a description of the inputs, see 
    :py:func:`keypoint_moseq.util.concatenate_stateseqs`

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
    return np.bincount(stateseq_flat)/len(stateseq_flat)


def find_matching_videos(keys, video_dir, as_dict=False):
    """
    Find video files for a set of session names. The filename of each
    video is assumed to be a prefix within the session name. For example
    given the following video directory::

        videodirectory
        ├─ videoname1.avi
        └─ videoname2.avi
 
    the videos would be matched to session names as follows::

        >>> keys = ['videoname1blahblah','videoname2yadayada']
        >>> find_matching_videos(keys, video_dir, as_dict=True)

        {'videoname1blahblah': 'videodirectory/videoname1.avi',
         'videoname2blahblah': 'videodirectory/videoname2.avi'}
 
    Parameters
    -------
    keys: iterable
        Session names (as strings)

    video_dir: str
        Path to the video directory. Videos are assumed to have the 
        one of the following extensions: "mp4", "avi", "mov"

    as_dict: bool, default=False
        Determines whether to return a dict mapping session names to 
        video paths, or a list of paths in the same order as `keys`.

    Returns
    -------
    video_paths: list or dict (depending on `as_dict`)

    """  
    video_to_path = {
        name : os.path.join(video_dir,name+ext) 
        for name,ext in map(os.path.splitext,os.listdir(video_dir)) 
        if ext in ['.mp4','.avi','.mov']}
    video_paths = []
    for key in keys:
        matches = [path for video,path in video_to_path.items() 
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
    data: ndarray, shape (num_segs, seg_length, *dims)
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
    data: ndarray, shape (N, seg_length, *dims)
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
            if (e-s > min_duration and s>=pre and s<len(stateseq)-post): 
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
                X = [np_io(inverse_affine_transform)(
                        x,centroids[key][s],headings[key][s]
                     ) for x,(key,s,e) in zip(X,instances)]
        else:
            X = np.array([coordinates[key][s-pre:s+post] for key,s,e in instances])
            if centroids is not None and headings is not None:
                c = np.array([centroids[key][s] for key,s,e in instances])[:,None]
                h = np.array([headings[key][s] for key,s,e in instances])[:,None]
                X = np_io(inverse_affine_transform)(X,c,h)
        trajectories[syllable] = X

    return trajectories



def sample_instances(syllable_instances, num_samples, mode='random', 
                     pca_samples=50000, pca_dim=4, n_neighbors=50,
                     coordinates=None, pre=5, post=15, centroids=None, 
                     headings=None, filter_size=9):
    """
    Sample a fixed number of instances for each syllable.

    Parameters
    -------
    syllable_instances: dict
        Mapping from each syllable to a list of instances, where each
        instance is a tuple of the form (name,start,end)

    num_samples: int
        Number of samples return for each syllable

    mode: str
        One of the following sampling methods

        'random'
            Instances are chosen randomly (without replacement)

        'density'
            For each syllable, a syllable-specific density function is
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

    The remaining parameters are passed to :py:func:`keypoint_moseq.util.get_trajectories`

    Returns
    -------
    sampled_instances: dict
        Dictionary in the same format as ``syllable_instances`` mapping
        each syllable to a list of sampled instances.

    """
    assert mode in ['random','density']
    assert all([len(v)>=num_samples for v in syllable_instances.values()])
    assert n_neighbors>=num_samples
    
    if mode=='density': 
        assert not (coordinates is None or headings is None or centroids is None), fill(
            '``coordinates``, ``headings`` and ``centroids`` are required when '
            '``mode == "density"``')
    
    if mode=='random':
        sampled_instances = {syllable: [instances[i] for i in np.random.choice(
            len(instances), num_samples, replace=False
        )] for syllable,instances in syllable_instances.items()}
        return sampled_instances
    
    
    else:
        trajectories = get_trajectories(
            syllable_instances, coordinates, pre=pre, post=post, 
            centroids=centroids, headings=headings, filter_size=filter_size)
            
        X = np.vstack(list(trajectories.values()))
        if n>pca_samples: X = X[np.random.choice(n, pca_samples, replace=False)]
        pca = PCA(n_components=pca_dim).fit(X.reshape(X.shape[0],-1))
        Xpca = pca.transform(flatten(X))
        all_nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(Xpca)
        
        sampled_instances = {}
        sampled_trajectories = {}
        
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
            sampled_trajectories[syllable] = [syllable_trajectories[syllable][i] for i in samples]
        
        return sampled_instances

