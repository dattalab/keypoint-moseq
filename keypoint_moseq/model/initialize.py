import jax, jax.numpy as jnp, jax.random as jr
from jax.tree_util import tree_map
import numpy as np
from keypoint_moseq.util import transform_data_for_pca, align_egocentric, interpolate, jax_io
from keypoint_moseq.model.transitions import sample_hdp_transitions, sample_transitions
from keypoint_moseq.model.gibbs import resample_scales, resample_stateseqs
from keypoint_moseq.model.distributions import sample_mniw
from textwrap import fill
na = jnp.newaxis


def initialize_ar_params(seed, *, num_states, nu_0, S_0, M_0, K_0, **kwargs):
    Ab,Q = jax.vmap(sample_mniw, in_axes=(0,na,na,na,na))(
        jr.split(seed, num_states),nu_0,S_0,M_0,K_0)
    return Ab,Q

def initialize_hdp_transitions(seed, *, num_states, alpha, kappa, gamma):
    seeds = jr.split(seed)
    counts = jnp.zeros((num_states,num_states))
    betas_init = jr.dirichlet(seeds[0], jnp.ones(num_states)*gamma/num_states)   
    betas, pi = sample_hdp_transitions(seeds[1], counts, betas_init, alpha, kappa, gamma)
    return betas, pi

def initialize_transitions(seed, *, num_states, alpha, kappa):
    seeds = jr.split(seed)
    counts = jnp.zeros((num_states,num_states))
    pi = sample_transitions(seeds[1], counts, alpha, kappa)
    return pi


def initialize_obs_params(pca, *, Y, mask, latent_dimension, whiten, **kwargs):
    if whiten:
        Y_flat = transform_data_for_pca(Y, **kwargs)[mask>0]
        latents_flat = jax_io(pca.transform)(Y_flat)[:,:latent_dimension]
        cov = jnp.cov(latents_flat.T)
        W = jnp.linalg.cholesky(cov)
    else:
        W = jnp.eye(latent_dimension)
        
    Cd = jnp.array(jnp.hstack([
        pca.components_[:latent_dimension].T @ W, 
        pca.mean_[:,na]]))
    return Cd
                                 

def initialize_latents(pca, *, Y, Cd, **kwargs):
    Y_flat = transform_data_for_pca(Y, **kwargs)
    # using numpy.linalg.pinv because jax version crashes on windows
    return (Y_flat - Cd[:,-1]) @ jnp.array(np.linalg.pinv(Cd[:,:-1]).T)


def initialize_states(seed, pca, Y, mask, conf, conf_threshold, 
                      params, noise_prior, *, obs_hypparams, 
                      verbose=False, **kwargs):

    if verbose: print('Initializing states')
    
    if conf is None: 
        Y_interp = Y
    else: 
        Y_interp = interpolate(Y, conf<conf_threshold)
    
    states = {}
    
    states['v'],states['h'] = align_egocentric(
        Y_interp, **kwargs)[1:]
    
    states['x'] = initialize_latents(
        pca, Y=Y_interp, **params, **kwargs)
    
    states['s'] = resample_scales(
        seed, Y=Y, conf=conf, **states, **params, 
        s_0=noise_prior, **obs_hypparams)
    
    states['z'] = resample_stateseqs(
        seed, Y=Y, mask=mask, **states, **params)[0]
    
    return states  

    
def initialize_params(seed, pca, Y, mask, conf, conf_threshold, *,
                      ar_hypparams, trans_hypparams, verbose=False, 
                      **kwargs):
    
    if verbose: print('Initializing parameters')
    
    if conf is None: Y_interp = Y
    else: Y_interp = interpolate(Y, conf<conf_threshold)
        
    params = {'sigmasq':jnp.ones(Y.shape[-2])}
    params['Ab'],params['Q'] = initialize_ar_params(seed, **ar_hypparams)
    params['betas'],params['pi'] = initialize_hdp_transitions(seed, **trans_hypparams)
    params['Cd'] = initialize_obs_params(pca, Y=Y_interp, mask=mask, **kwargs)
    return params


def initialize_hyperparams(*, conf, error_estimator, latent_dimension, 
                           trans_hypparams, ar_hypparams, obs_hypparams, 
                           cen_hypparams, use_bodyparts, verbose=False, 
                           **kwargs):
    
    if verbose: print('Initializing hyper-parameters')
    
    trans_hypparams = trans_hypparams.copy()
    obs_hypparams = obs_hypparams.copy()
    cen_hypparams = cen_hypparams.copy()
    ar_hypparams = ar_hypparams.copy()
    
    d,nlags = latent_dimension, ar_hypparams['nlags']
    ar_hypparams['S_0'] = ar_hypparams['S_0_scale']*jnp.eye(d)
    ar_hypparams['K_0'] = ar_hypparams['K_0_scale']*jnp.eye(d*nlags+1)
    ar_hypparams['M_0'] = jnp.pad(jnp.eye(d), ((0,0),((nlags-1)*d,1)))
    ar_hypparams['num_states'] = trans_hypparams['num_states']
    ar_hypparams['nu_0'] = d+2
    
    return {
        'ar_hypparams':ar_hypparams, 
        'obs_hypparams':obs_hypparams, 
        'cen_hypparams':cen_hypparams,
        'trans_hypparams':trans_hypparams}


def estimate_error(conf, *, slope, intercept):
    return (10**(jnp.log10(conf+1e-6)*slope+intercept))**2

def get_noise_prior(conf, *, error_estimator, use_bodyparts, **kwargs):
    if conf is None: return jnp.ones(len(use_bodyparts))
    else: return estimate_error(conf, **error_estimator)

def initialize_model(states=None, 
                     params=None, 
                     hypparams=None, 
                     noise_prior=None,
                     seed=None, 
                     pca=None, 
                     Y=None, 
                     mask=None, 
                     conf=None, 
                     conf_threshold=0.5, 
                     random_seed=0, 
                     **kwargs):
    
    if None in (states,params): assert not None in (pca,Y,mask), fill(
        'Either provide ``states`` and ``params`` or provide a pca model and data')
        
    if seed is not None: seed = jnp.array(seed, dtype='uint32')
    else: seed = jr.PRNGKey(random_seed)

    if hypparams is not None:
        as_scalar = lambda arr: arr.item() if arr.shape==() else arr
        hypparams = tree_map(as_scalar, jax.device_put(hypparams))
    else: hypparams = initialize_hyperparams(conf=conf, **kwargs)
    kwargs.update(hypparams)
    
    if noise_prior is None: noise_prior = get_noise_prior(conf, **kwargs)
        
    if params is not None: params = jax.device_put(params)
    else: params = initialize_params(
        seed, pca, Y, mask, conf, conf_threshold, **kwargs)
        
    if states is not None: states = jax.device_put(states)
    else: states = initialize_states(
        seed, pca, Y, mask, conf, conf_threshold, params, noise_prior, **kwargs)

    return {
        'seed': seed,
        'states': states, 
        'params': params, 
        'hypparams': hypparams, 
        'noise_prior': noise_prior}
