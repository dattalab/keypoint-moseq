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
    return lambda *args, **kwargs: jax.device_put(
        fn(*jax.device_get(args), **jax.device_get(kwargs)))

def np_io(fn): 
    return lambda *args, **kwargs: jax.device_get(
        fn(*jax.device_put(args), **jax.device_put(kwargs)))

@njit
def count_transitions(num_states, mask, stateseqs):
    """
    Count all transitions in `stateseqs` where the start and end
    states both have `mask>0`. The last dim of `stateseqs` indexes time. 
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

    
def estimate_coordinates(*, Y, v, h, x, Cd, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    Yexp = affine_transform(Ybar,v,h)
    return Yexp

@jax_io
def interpolate(keypoints, outliers, axis=1):
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
    # using numpy.linalg.svd because jax version crashes on windows
    return jnp.array(np.linalg.svd(np.eye(k) - np.ones((k,k))/k)[0][:,:-1])

def stateseq_stats(stateseqs, mask):
    s = np.array(stateseqs.flatten()[mask[...,-stateseqs.shape[-1]:].flatten()>0])
    durations = [sum(1 for i in g) for k,g in groupby(s)]
    usage = np.bincount(s)
    return usage, durations


def ensure_symmetric(X):
    XT = jnp.swapaxes(X,-1,-2)
    return (X+XT)/2

def vector_to_angle(V):
    """Convert 2D vectors to angles in [-pi, pi]. The vector (1,0)
    corresponds to angle of 0. If V is n-dinmensional, the first
    n-1 dimensions are treated as batch dims. 

    Parameters
    ----------
    V: ndarray, shape (*dims, d) where d >= 2

    Returns
    -------
    angles: ndarray, shape (*dims)
    
    """
    y,x = V[...,1],V[...,0]+1e-10
    angles = (jnp.arctan(y/x)+(x>0)*jnp.pi)%(2*jnp.pi)-jnp.pi
    return angles

    
def angle_to_rotation_matrix(h, keypoint_dim=3):
    """Create rotation matrices from an array of angles. If
    `keypoint_dim > 2` then rotation is performed in the 
    first two dims and the remaing axes are fixed.

    Parameters
    ----------
    h: ndarray, shape (*dims)
        Angles (in radians)

    Returns
    -------
    m: ndarray, shape (*dims, keypoint_dim, keypoint_dim)
        Stacked rotation matrices 
        
    keypoint_dim: int, default=3
        Dimension of each rotation matrix
    
    """
    m = jnp.tile(jnp.eye(keypoint_dim), (*h.shape,1,1))
    m = m.at[...,0,0].set(jnp.cos(h))
    m = m.at[...,1,1].set(jnp.cos(h))
    m = m.at[...,0,1].set(-jnp.sin(h))
    m = m.at[...,1,0].set(jnp.sin(h))
    return m


@jax.jit
def affine_transform(Y, v, h):
    """
    Apply the following affine transform, batching over keypoints.
    
    .. math::
        Y \mapsto R(h) @ Y + v

    Parameters
    ----------   
    Y: ndarray, shape (*dims, k, d) where d >= 2
        Batches of k keypoints in d-dim embedding space
        
    v: ndarray, shape (*dims, d)
        Translations
        
    h: ndarray, shape (*dims)
        Angles in radians
          
    Returns
    -------
    Ytransformed: ndarray, shape (*dims, k, d)
        
    """
    rot_matrix = angle_to_rotation_matrix(h, keypoint_dim=Y.shape[-1])
    Ytransformed = (Y[...,na,:]*rot_matrix[...,na,:,:]).sum(-1) + v[...,na,:]
    return Ytransformed

@jax.jit
def inverse_affine_transform(Y, v, h):
    """
    Apply the following affine transform, batching over keypoints.
    
    .. math::
        Y \mapsto R(-h) @ (Y - v)

    Parameters
    ----------   
    Y: ndarray, shape (*dims, k, d) where d >= 2
        Batches of k keypoints in d-dim embedding space
        
    v: ndarray, shape (*dims, d)
        Translations
        
    h: ndarray, shape (*dims)
        Angles in radians
          
    Returns
    -------
    Ytransformed: ndarray, shape (*dims, k, d)
        
    """
    rot_matrix = angle_to_rotation_matrix(-h, keypoint_dim=Y.shape[-1])
    Ytransformed = ((Y-v[...,na,:])[...,na,:]*rot_matrix[...,na,:,:]).sum(-1)
    return Ytransformed

def transform_data_for_pca(Y, **kwargs):
    n,t,k,d = Y.shape
    Y_aligned = align_egocentric(Y, **kwargs)[0]
    Y_flat = (center_embedding(k).T @ Y_aligned).reshape(n,t,(k-1)*d)
    return Y_flat

def fit_pca(*, Y, conf, mask, conf_threshold=0.5, verbose=False,
            PCA_fitting_num_frames=100000, **kwargs):
    
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


def align_egocentric(Y, *, use_bodyparts, anterior_bodyparts, posterior_bodyparts, **kwargs):
    
    anterior_keypoints = jnp.array([use_bodyparts.index(bp) for bp in anterior_bodyparts])
    posterior_keypoints = jnp.array([use_bodyparts.index(bp) for bp in posterior_bodyparts])
    
    posterior_loc = Y[..., posterior_keypoints,:2].mean(-2) 
    anterior_loc = Y[..., anterior_keypoints,:2].mean(-2) 
    
    h = vector_to_angle(anterior_loc-posterior_loc)
    v = Y.mean(-2).at[...,2:].set(0)
    return inverse_affine_transform(Y,v,h),v,h


def pad_affine(x):
    """
    Pad with 1's to enable affine transform with matrix multiplication.
    Padding is appended and the end of the last dimension.
    """
    padding = jnp.ones((*x.shape[:-1],1))
    xpadded = jnp.concatenate((x,padding),axis=-1)
    return xpadded



def get_lags(x, nlags):
    """
    Get lags of a multivariate time series.
    
    Parameters
    ----------  
    nlags: int
        Number of lags
        
    x: ndarray, shape (*dims, t, d)
        Batch of multivariate time series
    
    Returns
    -------
    xlagged: ndarray, shape [*dims, t-nlags, d*nlags]
        Array formed by concatenating lags of x along the last dim.
        The last row of `xlagged` is `x[...,-nlags,:],...,x[...,-2,:]`
    """
    lags = [jnp.roll(x,t,axis=-2) for t in range(1,nlags+1)]
    return jnp.concatenate(lags[::-1],axis=-1)[...,nlags:,:]


def ar_to_lds(As, bs, Qs, Cs):
    k,l = As.shape[1],As.shape[2]//As.shape[1]

    A_ = jnp.zeros((As.shape[0],k*l,k*l))
    A_ = A_.at[:,:-k,k:].set(jnp.eye(k*(l-1)))
    A_ = A_.at[:,-k:].set(As)
    
    Q_ = jnp.zeros((Qs.shape[0],k*l,k*l))
    Q_ = Q_.at[:,:-k,:-k].set(jnp.eye(k*(l-1))*1e-2)
    Q_ = Q_.at[:,-k:,-k:].set(Qs)
    
    b_ = jnp.zeros((bs.shape[0],k*l))
    b_ = b_.at[:,-k:].set(bs)
    
    C_ = jnp.zeros((Cs.shape[0],k*l))
    C_ = C_.at[:,-k:].set(Cs)
    return A_, b_, Q_, C_

def gaussian_log_prob(x, mu, sigma_inv):
    return (-((mu-x)[...,na,:]*sigma_inv*(mu-x)[...,:,na]).sum((-1,-2))/2
            +jnp.log(jnp.linalg.det(sigma_inv))/2)


def latent_log_prob(*, x, z, Ab, Q, **kwargs):
    Qinv = jnp.linalg.inv(Q)
    Qdet = jnp.linalg.det(Q)
    
    nlags = Ab.shape[2]//Ab.shape[1]
    x_lagged = get_lags(x, nlags)
    x_pred = (Ab[z] @ pad_affine(x_lagged)[...,na])[...,0]
    
    d = x_pred - x[:,nlags:]
    return (-(d[...,na,:]*Qinv[z]*d[...,:,na]).sum((2,3))/2
            -jnp.log(Qdet[z])/2  -jnp.log(2*jnp.pi)*Q.shape[-1]/2)

def stateseq_log_prob(*, z, pi, **kwargs):
    return jnp.log(pi[z[:,:-1],z[:,1:]])

def scale_log_prob(*, s, nu_s, s_0, **kwargs):
    return -nu_s*s_0 / s / 2 - (1+nu_s/2)*jnp.log(s)
    
def location_log_prob(*, v, sigmasq_loc):
    d = v[:,:-1]-v[:,1:]
    return (-(d**2).sum(-1)/sigmasq_loc/2 
            -v.shape[-1]/2*jnp.log(sigmasq_loc*2*jnp.pi))

def obs_log_prob(*, Y, x, s, v, h, Cd, sigmasq, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    sqerr = ((Y - affine_transform(Ybar,v,h))**2).sum(-1)
    return (-1/2 * sqerr/s/sigmasq - d/2 * jnp.log(2*s*sigmasq*jnp.pi))

@jax.jit
def log_joint_likelihood(*, Y, mask, x, s, v, h, z, pi, Ab, Q, Cd, sigmasq, sigmasq_loc, nu_s, s_0, **kwargs):
    nlags = Ab.shape[2]//Ab.shape[1]
    return {
        'Y': (obs_log_prob(Y=Y, x=x, s=s, v=v, h=h, Cd=Cd, sigmasq=sigmasq)*mask[:,:,na]).sum(),
        'x': (latent_log_prob(x=x, z=z, Ab=Ab, Q=Q)*mask[:,nlags:]).sum(),
        'z': (stateseq_log_prob(z=z, pi=pi)*mask[:,nlags+1:]).sum(),
        'v': (location_log_prob(v=v, sigmasq_loc=sigmasq_loc)*mask[:,1:]).sum(),
        's': (scale_log_prob(s=s, nu_s=nu_s, s_0=s_0)*mask[:,:,na]).sum()}

def merge_data(data_dict, keys=None, batch_length=None):
    if keys is None: keys = sorted(data_dict.keys())
    max_length = np.max([data_dict[k].shape[0] for k in keys])
    if batch_length is None: batch_length = max_length
        
    def reshape(x):
        padding = (-x.shape[0])%batch_length
        x = np.concatenate([x, np.zeros((padding,*x.shape[1:]))],axis=0)
        return x.reshape(-1, batch_length, *x.shape[1:])
    
    data = np.concatenate([reshape(data_dict[k]) for k in keys],axis=0)
    mask = np.concatenate([reshape(np.ones(data_dict[k].shape[0])) for k in keys],axis=0)
    keys = [(k,i) for k in keys for i in range(int(np.ceil(len(data_dict[k])/batch_length)))]
    return data, mask, keys

def print_dims_to_explain_variance(pca, f):
    cs = np.cumsum(pca.explained_variance_ratio_)
    if cs[-1] < f: print(f'All components together only explain {cs[-1]*100}% of variance.')
    else: print(f'>={f*100}% of variance exlained by {(cs>f).nonzero()[0].min()+1} components.')

def get_durations(z, mask):
    stateseq_flat = z[mask[:,-z.shape[1]:]>0]
    changepoints = np.insert(np.diff(stateseq_flat).nonzero()[0]+1,0,0)
    return changepoints[1:]-changepoints[:-1]

def get_usages(syllables, mask=None):
    if isinstance(syllables, dict):
        stateseq_flat = np.hstack(list(syllables.values()))
    elif mask is not None:
        stateseq_flat = syllables[mask[:,-syllables.shape[1]:]>0]
    else: stateseq_flat = syllables.flatten()
    return np.bincount(stateseq_flat)/len(stateseq_flat)

def find_matching_videos(keys, video_dir, as_dict=False):
    video_to_path = {
        name : os.path.join(video_dir,name+ext) 
        for name,ext in map(os.path.splitext,os.listdir(video_dir)) 
        if ext in ['.mp4','.avi','.mov']}
    video_paths = []
    for key in sorted(keys):
        matches = [path for video,path in video_to_path.items() 
                   if os.path.basename(key).startswith(video)]
        assert len(matches)>0, fill(f'No matching videos found for {key}')
        assert len(matches)<2, fill(f'More than one video matches {key} ({matches})')
        video_paths.append(matches[0])
    if as_dict: return dict(zip(sorted(keys),video_paths))
    else: return video_paths

def unbatch(batches, mask, batch_info): 
    nlags = mask.shape[1]-batches.shape[1]
    keys = sorted(set([key for key,start,end in batch_info]))
    
    unbatched = {}
    for key in keys:
        length = np.max([e for k,s,e in batch_info if k==key])
        seq = np.zeros((int(length),*batches.shape[2:]), dtype=batches.dtype)
        for (k,s,e),m,b in zip(batch_info,mask,batches):
            if k==key: seq[s+nlags:e] = b[m[nlags:]>0]
        unbatched[key] = seq[nlags:]
    return unbatched


def batch(data_dict, batch_length=None, keys=None, batch_overlap=30):
    if keys is None: keys = sorted(data_dict.keys())
    Ns = [len(data_dict[key]) for key in keys]
    if batch_length is None: batch_length = np.max(Ns)
        
    batches,mask,batch_info = [],[],[]
    for key,N in zip(keys,Ns):
        for start in range(0,N,batch_length):
            arr = data_dict[key]
            end = min(start+batch_length+batch_overlap, N)
            pad_length = batch_length+batch_overlap-(end-start)
            padding = np.zeros((pad_length,*arr.shape[1:]), dtype=arr.dtype)
            mask.append(np.hstack([np.ones(end-start),np.zeros(pad_length)]))
            batches.append(np.concatenate([arr[start:end],padding],axis=0))
            batch_info.append((key,start,end))

    batches = np.stack(batches)
    mask = np.stack(mask)
    return batches,mask,batch_info
    
    
def filter_angle(angles, size=9, axis=0):
    kernel = np.where(np.arange(len(angles.shape))==axis, size, 1)
    return np.arctan2(median_filter(np.sin(angles), kernel),
                      median_filter(np.cos(angles), kernel))

def filter_centroids_headings(centroids, headings, filter_size=9):
    centroids = {k:median_filter(v,(filter_size,1)) for k,v in centroids.items()}
    headings = {k:filter_angle(v, size=filter_size) for k,v in headings.items()}  
    return centroids, headings


def get_syllable_instances(syllables, min_duration=3, pre=30, post=60,
                           min_usage=0, min_instances=0):
    
    num_syllables = int(max(map(max,syllables.values()))+1)
    syllable_instances = [[] for syllable in range(num_syllables)]
    
    for key,stateseq in syllables.items():
        transitions = np.nonzero(stateseq[1:] != stateseq[:-1])[0]+1
        starts = np.insert(transitions,0,0)
        ends = np.append(transitions,len(stateseq))
        for (s,e,syllable) in zip(starts,ends,stateseq[starts]):
            if (e-s > min_duration and s>=pre and s<len(stateseq)-post): 
                syllable_instances[syllable].append((key,s,e))
                
    usages_filter = get_usages(syllables) >= min_usage
    counts_filter = np.array(list(map(len,syllable_instances))) >= min_instances
    use_syllables = np.all([usages_filter, counts_filter],axis=0).nonzero()[0]
    return {syllable : syllable_instances[syllable] for syllable in use_syllables} 

def get_edges(use_bodyparts, skeleton):
    edges = []
    for bp1,bp2 in skeleton:
        if bp1 in use_bodyparts and bp2 in use_bodyparts:
            edges.append([use_bodyparts.index(bp1),use_bodyparts.index(bp2)])
    return edges
        
def reindex_by_bodyparts(data, bodyparts, use_bodyparts, axis=1):
    ix = np.array([bodyparts.index(bp) for bp in use_bodyparts])
    if isinstance(data, np.ndarray): return np.take(data, ix, axis)
    else: return {k: np.take(v, ix, axis) for k,v in data.items()}
    
    
    

def get_trajectories(syllable_instances, coordinates, pre=5, post=15, 
                     centroids=None, headings=None, filter_size=9):
    
    centroids,headings = filter_centroids_headings(
        centroids, headings, filter_size=filter_size)
        
    trajectories = {}
    for syllable,instances in syllable_instances.items():
        X = np.array([coordinates[key][s-pre:s+post] for key,s,e in instances])
        c = np.array([centroids[key][s] for key,s,e in instances])[:,None]
        h = np.array([headings[key][s] for key,s,e in instances])[:,None]
        trajectories[syllable] = np_io(inverse_affine_transform)(X,c,h)
   
    return trajectories





def sample_instances(syllable_instances, num_samples, pre=5, post=15,
                     mode='random', bodyparts=None, use_bodyparts=None,
                     centroids=None, headings=None, coordinates=None, 
                     pca_samples=50000, pca_dim=4, n_neighbors=50,
                     filter_size=9, return_trajectories=False):
    
    assert mode in ['random','density']
    assert all([len(v)>=num_samples for v in syllable_instances.values()])
    assert n_neighbors>=num_samples
    
    if return_trajectories or mode=='density': 
        assert not (coordinates is None or headings is None or centroids is None), fill(
            '``coordinates``, ``headings`` and ``centroids`` are required when '
            '``mode == "density"`` or ``return_trajectories==True``')
    
    if mode=='random':
        sampled_instances = {syllable: [instances[i] for i in np.random.choice(
            len(instances), num_samples, replace=False
        )] for syllable,instances in syllable_instances.items()}
        
        if return_trajectories: 
            sampled_trajectories = get_trajectories(
                sampled_instances, coordinates, pre=pre, post=post, 
                centroids=centroids, headings=headings, 
                filter_size=filter_size)
            return sampled_instances, sampled_trajectories
        else: return sampled_instances
    
    
    else:
        trajectories = get_trajectories(
            syllable_instances, coordinates, pre=pre, post=post, 
            centroids=centroids, headings=headings, 
            filter_size=filter_size)
            
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
        
        if not return_trajectories: return sampled_instances
        else: return sampled_instances, sampled_trajectories
