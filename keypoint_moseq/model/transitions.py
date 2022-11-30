from numba import njit, prange
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr


@njit
def sample_crp_tablecounts(concentration,customers,colweights):
    m = np.zeros_like(customers)
    tot = np.sum(customers)
    randseq = np.random.random(tot)
    tmp = np.empty_like(customers).flatten()
    tmp[0] = 0
    tmp[1:] = np.cumsum(np.ravel(customers)[:customers.size-1])
    starts = tmp.reshape(customers.shape)
    for i in prange(customers.shape[0]):
        for j in range(customers.shape[1]):
            for k in range(customers[i,j]):
                m[i,j] += randseq[starts[i,j]+k] \
                    < (concentration * colweights[j]) / (k+concentration*colweights[j])
    return m

def sample_ms(counts, betas, alpha, kappa):
    ms = sample_crp_tablecounts(alpha, np.array(counts,dtype=int), np.array(betas))
    newms = ms.copy()
    if ms.sum() > 0:
        # np.random.binomial fails when n=0, so pull out nonzero indices
        indices = np.nonzero(newms.flat[::ms.shape[0]+1])
        newms.flat[::ms.shape[0]+1][indices] = np.array(np.random.binomial(
                ms.flat[::ms.shape[0]+1][indices],
                betas[indices]*alpha/(betas[indices]*alpha + kappa)),
                dtype=np.int32)
    return jnp.array(newms)

def sample_hdp_transitions(seed, counts, betas, alpha, kappa, gamma):
    seeds,N = jr.split(seed,3),counts.shape[0]
    ms = sample_ms(counts, betas, alpha, kappa)
    betas = jr.dirichlet(seeds[1], ms.sum(0)+gamma/N)
    conc = alpha*betas[None,:] + counts + kappa*jnp.identity(N)
    return betas, jr.dirichlet(seeds[2], conc)

def sample_transitions(seed, counts, alpha, kappa):
    conc = counts + alpha + kappa*jnp.identity(counts.shape[0])
    return jr.dirichlet(seed, conc)
    