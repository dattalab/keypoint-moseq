from jax.config import config
import jax, jax.numpy as jnp, jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd

def sample_vonmises(seed, theta, kappa):
    return tfd.VonMises(theta, kappa).sample(seed=seed)

def sample_gamma(seed, a, b):
    return jr.gamma(seed, a) / b

def sample_inv_gamma(seed, a, b):
    return 1/sample_gamma(seed, a, b)

def sample_scaled_inv_chi2(seed, degs, variance):
    return sample_inv_gamma(seed, degs/2, degs*variance/2)

def sample_chi2(seed, degs):
    return jr.gamma(seed, degs/2)*2

def sample_discrete(seed, distn,dtype=jnp.int32):
    return jr.categorical(seed, jnp.log(distn))

def sample_mn(seed, M, U, V):
    G = jr.normal(seed,M.shape)
    G = jnp.dot(jnp.linalg.cholesky(U),G)
    G = jnp.dot(G,jnp.linalg.cholesky(V).T)
    return M + G

def sample_invwishart(seed,S,nu):
    n = S.shape[0]
    chol = jnp.linalg.cholesky(S)
    chi2_seed, norm_seed = jr.split(seed)
    x = jnp.diag(jnp.sqrt(sample_chi2(chi2_seed, nu-jnp.arange(n))))
    x = x.at[jnp.triu_indices_from(x,1)].set(jr.normal(norm_seed, (n*(n-1)//2,)))
    R = jnp.linalg.qr(x,'r')
    T = jax.scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return jnp.dot(T,T.T)

def sample_mniw(seed, nu, S, M, K):
    sigma = sample_invwishart(seed, S, nu)
    A = sample_mn(seed, M, sigma, K)
    return A, sigma



def sample_hmm_stateseq(seed, log_likelihoods, mask, pi):
    """
    Use the forward-backward algorithm to sample state-sequences in a Markov chain.
    
    """
    def _forward_message(carry, args):
        ll_t, mask_t = args
        in_potential, logtot = carry
        cmax = ll_t.max()
        alphan_t = in_potential * jnp.exp(ll_t - cmax)
        norm = alphan_t.sum() + 1e-16
        alphan_t = alphan_t / norm
        logprob = jnp.log(norm) + cmax
        in_potential = alphan_t.dot(pi)*mask_t + in_potential*(1-mask_t)
        return (in_potential, logtot + logprob*mask_t), alphan_t    

    def _sample(args):
        seed, next_potential, alphan_t = args
        seed, newseed = jr.split(seed)
        s = sample_discrete(newseed, next_potential * alphan_t)
        next_potential = pi[:,s]
        return (seed,next_potential), s

    def _backward_message(carry, args):
        seed, next_potential = carry
        alphan_t, mask_t = args
        return jax.lax.cond(
            mask_t>0, _sample, 
            lambda args: (args[:-1],0), 
            (seed, next_potential, alphan_t))
        
    init_distn = jnp.ones(pi.shape[0])/pi.shape[0]
    (_,log_likelihood), alphan = jax.lax.scan(_forward_message,  (init_distn,0.), (log_likelihoods, mask))
    
    init_potential = jnp.ones(pi.shape[0])
    _,stateseq = jax.lax.scan(_backward_message, (seed,init_potential), (alphan,mask), reverse=True)
    return stateseq, log_likelihood


