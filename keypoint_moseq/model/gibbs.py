from jax.config import config
config.update('jax_enable_x64', True)
import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.util import *
from keypoint_moseq.model.distributions import *
from keypoint_moseq.model.transitions import sample_hdp_transitions, sample_transitions
from keypoint_moseq.model.kalman import kalman_sample
na = jnp.newaxis

@jax.jit
def resample_latents(seed, *, Y, mask, v, h, z, s, Cd, sigmasq, Ab, Q, **kwargs):
    d,nlags,n = Ab.shape[1],Ab.shape[2]//Ab.shape[1],Y.shape[0]
    Gamma = center_embedding(Y.shape[-2])
    Cd = jnp.kron(Gamma, jnp.eye(Y.shape[-1])) @ Cd
    ys = inverse_affine_transform(Y,v,h).reshape(*Y.shape[:-2],-1)
    A, B, Q, C, D = *ar_to_lds(Ab[...,:-1],Ab[...,-1],Q,Cd[...,:-1]),Cd[...,-1]
    R = jnp.repeat(s*sigmasq,Y.shape[-1],axis=-1)[:,nlags-1:]
    mu0,S0 = jnp.zeros(d*nlags),jnp.eye(d*nlags)*10
    xs = jax.vmap(kalman_sample, in_axes=(0,0,0,0,na,na,na,na,na,na,na,0))(
        jr.split(seed, ys.shape[0]), ys[:,nlags-1:], mask[:,nlags-1:-1], 
        z, mu0, S0, A, B, Q, C, D, R)
    xs = jnp.concatenate([xs[:,0,:-d].reshape(-1,nlags-1,d), xs[:,:,-d:]],axis=1)
    return xs

@jax.jit
def resample_heading(seed, *, Y, v, x, s, Cd, sigmasq, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    S = (Ybar[...,na]*(Y - v[...,na,:])[...,na,:]/(s*sigmasq)[...,na,na]).sum(-3)
    kappa_cos = S[...,0,0]+S[...,1,1]
    kappa_sin = S[...,0,1]-S[...,1,0]
    theta = vector_to_angle(jnp.stack([kappa_cos,kappa_sin],axis=-1))
    kappa = jnp.sqrt(kappa_cos**2 + kappa_sin**2)
    return sample_vonmises(seed, theta, kappa)

    
@jax.jit 
def resample_location(seed, *, mask, Y, h, x, s, Cd, sigmasq, sigmasq_loc, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    gammasq = 1/(1/(s*sigmasq)).sum(-1)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    rot_matrix = angle_to_rotation_matrix(h, d=d)
    mu = ((Y - (rot_matrix[...,na,:,:]*Ybar[...,na,:]).sum(-1)) \
          *(gammasq[...,na]/(s*sigmasq))[...,na]).sum(-2)

    m0,S0 = jnp.zeros(d), jnp.eye(d)*1e6
    A,B,Q = jnp.eye(d)[na],jnp.zeros(d)[na],jnp.eye(d)[na]*sigmasq_loc
    C,D,R = jnp.eye(d),jnp.zeros(d),gammasq[...,na]*jnp.ones(d)
    zz = jnp.zeros_like(mask[:,1:], dtype=int)

    return jax.vmap(kalman_sample, in_axes=(0,0,0,0,na,na,na,na,na,na,na,0))(
        jr.split(seed, mask.shape[0]), mu, mask[:,:-1], zz, m0, S0, A, B, Q, C, D, R)



@jax.jit
def resample_obs_params(seed, *, Y, mask, sigmasq, v, h, x, s, sigmasq_C, **kwargs):
    k,d,D = *Y.shape[-2:],x.shape[-1]
    Gamma = center_embedding(k)
    mask = mask.flatten()
    s = s.reshape(-1,k)
    x = x.reshape(-1,D)
    xt = pad_affine(x)

    Sinv = jnp.eye(k)[na,:,:]/s[:,:,na]/sigmasq[na,:,na]
    xx_flat = (xt[:,:,na]*xt[:,na,:]).reshape(xt.shape[0],-1).T
    # serialize this step because of memory constraints
    mGSG = mask[:,na,na] * Gamma.T@Sinv@Gamma
    S_xx_flat = jax.lax.map(lambda xx_ij: (xx_ij[:,na,na]*mGSG).sum(0), xx_flat)
    S_xx = S_xx_flat.reshape(D+1,D+1,k-1,k-1)
    S_xx = jnp.kron(jnp.concatenate(jnp.concatenate(S_xx,axis=-2),axis=-1),jnp.eye(d))
    Sigma_n = jnp.linalg.inv(jnp.eye(d*(D+1)*(k-1))/sigmasq_C + S_xx)

    vecY = inverse_affine_transform(Y, v, h).reshape(-1,k*d)
    S_yx = (mask[:,na,na]*vecY[:,:,na]*jnp.kron(
        jax.vmap(jnp.kron)(xt[:,na,:],Sinv@Gamma), 
        jnp.eye(d))).sum((0,1))         
    mu_n = Sigma_n@S_yx
                         
    return jr.multivariate_normal(seed, mu_n, Sigma_n).reshape(D+1,d*(k-1)).T

@jax.jit
def resample_obs_variance(seed, *, Y, mask, Cd, v, h, x, s, nu_sigma, sigmasq_0, **kwargs):
    k,d = Y.shape[-2:]
    s = s.reshape(-1,k)
    mask = mask.flatten()
    x = x.reshape(-1,x.shape[-1])
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(-1,k-1,d)
    Ytild = inverse_affine_transform(Y,v,h).reshape(-1,k,d)
    S_y = (mask[:,na]*((Ytild - Ybar)**2).sum(-1)/s).sum(0)
    variance = (nu_sigma*sigmasq_0 + S_y)/(nu_sigma+3*mask.sum())
    degs = (nu_sigma+3*mask.sum())*jnp.ones_like(variance)
    return sample_scaled_inv_chi2(seed, degs, variance)



def _ar_log_likelihood(x, params):
    Ab, Q = params
    nlags = Ab.shape[-1]//Ab.shape[-2]
    mu = pad_affine(get_lags(x, nlags))@Ab.T
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(x[...,nlags:,:])


@jax.jit
def resample_stateseqs(seed, *, x, mask, Ab, Q, pi, **kwargs):
    nlags = Ab.shape[2]//Ab.shape[1]
    log_likelihoods = jax.lax.map(partial(_ar_log_likelihood,x), (Ab, Q))
    stateseqs, log_likelihoods = jax.vmap(sample_hmm_stateseq, in_axes=(0,0,0,na))(
        jr.split(seed,mask.shape[0]),
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[:,nlags:], pi)
    return stateseqs, log_likelihoods

@jax.jit
def resample_scales(seed, *, x, v, h, Y, Cd, sigmasq, nu_s, s_0, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    Ytild = inverse_affine_transform(Y,v,h)
    variance = (((Ytild - Ybar)**2).sum(-1)/sigmasq + s_0*nu_s)/(nu_s+3)
    degs = (nu_s+3)*jnp.ones_like(variance)
    return sample_scaled_inv_chi2(seed, degs, variance)

@jax.jit
def resample_regression_params(seed, mask, x_in, x_out, nu_0, S_0, M_0, K_0):
    S_out_out = (x_out[:,:,na]*x_out[:,na,:]*mask[:,na,na]).sum(0)
    S_out_in = (x_out[:,:,na]*x_in[:,na,:]*mask[:,na,na]).sum(0)
    S_in_in = (x_in[:,:,na]*x_in[:,na,:]*mask[:,na,na]).sum(0)
    K_0_inv = jnp.linalg.inv(K_0)
    K_n = jnp.linalg.inv(K_0_inv + S_in_in)
    M_n = (M_0@K_0_inv + S_out_in)@K_n
    S_n = S_0 + S_out_out + (M_0@K_0_inv@M_0.T - M_n@jnp.linalg.inv(K_n)@M_n.T)
    return sample_mniw(seed, nu_0+mask.sum(), S_n, M_n, K_n)


@partial(jax.jit, static_argnames=('num_states','nlags'))
def resample_ar_params(seed, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs):
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1,nlags*x.shape[-1]+1)
    x_out = x[...,nlags:,:].reshape(-1,x.shape[-1])
    masks = mask[...,nlags:].reshape(1,-1)*jnp.eye(num_states)[:,z.flatten()]
    return jax.vmap(resample_regression_params, in_axes=(0,0,na,na,na,na,na,na))(
        jr.split(seed,num_states), masks, x_in, x_out, nu_0, S_0, M_0, K_0)


def resample_hdp_transitions(seed, *, z, mask, betas, alpha, kappa, gamma, num_states, **kwargs):
    counts = jax_io(count_transitions)(num_states, z, mask)
    betas, pi = sample_hdp_transitions(seed, counts, betas, alpha, kappa, gamma)
    return betas, pi

def resample_transitions(seed, *, z, mask, alpha, kappa, num_states, **kwargs):
    counts = jax_io(count_transitions)(num_states, z, mask)
    pi = sample_transitions(seed, counts, alpha, kappa)
    return pi


def resample_model(data, *, states, params, hypparams, seed, 
                   noise_prior, ar_only=False, states_only=False):
    
    seed = jr.split(seed)[1]

    if not states_only: 
        params['betas'],params['pi'] = resample_hdp_transitions(
            seed, **data, **states, **params, 
            **hypparams['trans_hypparams'])
        
    if not states_only: 
        params['Ab'],params['Q']= resample_ar_params(
            seed, **data, **states, **params, 
            **hypparams['ar_hypparams'])
    
    states['z'] = resample_stateseqs(
        seed, **data, **states, **params)[0]
    
    if not ar_only:     

        if not states_only: 
            params['sigmasq'] = resample_obs_variance(
                seed, **data, **states, **params, 
                **hypparams['obs_hypparams'])
        
        states['x'] = resample_latents(
            seed, **data, **states, **params)
        
        states['h'] = resample_heading(
            seed, **data, **states, **params)
        
        states['v'] = resample_location(
            seed, **data, **states, **params, 
            **hypparams['cen_hypparams'])
        
        states['s'] = resample_scales(
            seed, **data, **states, **params, 
            s_0=noise_prior, **hypparams['obs_hypparams'])
        
    return {
        'seed': seed,
        'states': states, 
        'params': params, 
        'hypparams': hypparams,
        'noise_prior': noise_prior}
