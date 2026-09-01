import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm

import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.initialize import init_states, estimate_error
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca

def batched_init_model(data, pca, config, params_subset_size=5, batch_size=20, seed=None):
    """
    Initializes a Keypoint-MoSeq model in batches to avoid GPU OOM errors on large datasets.
    
    This function acts as a wrapper around `kpms.init_model`, processing the 
    state initialization step in chunks to reduce peak memory usage.
    """
    N_sequences = data['Y'].shape[0]

    if seed is None:
        seed = jax.random.PRNGKey(0)

    # STEP 1: Initialize Global Parameters (Small subset)
    subset_slice = slice(0, min(params_subset_size, N_sequences))
    subset_data = {
        'Y': data['Y'][subset_slice],
        'mask': data['mask'][subset_slice],
        'conf': data['conf'][subset_slice] if 'conf' in data else None
    }

    print(f"[Batch Init] Fitting global parameters on subset of {subset_slice.stop} sequences...")
    model = kpms.init_model(subset_data, pca=pca, seed=seed, **config)

    # STEP 2: Initialize States (Batched)
    params = model['params']
    hypparams = model['hypparams']
    obs_hypparams = hypparams['obs_hypparams']
    
    current_seed = model.get('seed', seed)
    prior = current_seed

    num_batches = int(np.ceil(N_sequences / batch_size))
    full_states = {'x': [], 'v': [], 'h': [], 's': [], 'z': []}
    
    print(f"[Batch Init] Initializing states for {N_sequences} sequences in {num_batches} batches...")

    for i in range(num_batches):
        prior, batch_seed = jax.random.split(prior)
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N_sequences)

        Y_chunk = jnp.array(data['Y'][start_idx:end_idx])
        mask_chunk = jnp.array(data['mask'][start_idx:end_idx])
        conf_chunk = jnp.array(data['conf'][start_idx:end_idx]) if 'conf' in data else None

        Y_flat_chunk, v_chunk, h_chunk = preprocess_for_pca(
            Y_chunk,
            anterior_idxs=config.get('anterior_idxs'),
            posterior_idxs=config.get('posterior_idxs'),
            conf=conf_chunk,
            conf_threshold=config.get('conf_threshold', 0.5),
            fix_heading=config.get('fix_heading', False),
            verbose=False
        )

        if conf_chunk is not None and config.get('error_estimator', None) is not None:
            batch_noise_prior = estimate_error(conf_chunk, **config['error_estimator'])
        else:
            batch_noise_prior = 1.0

        chunk_states = init_states(
            batch_seed, Y_chunk, mask_chunk, params,
            batch_noise_prior, obs_hypparams,
            Y_flat=Y_flat_chunk, v=v_chunk, h=h_chunk, 
        )

        # Offload to CPU
        for k in ['x', 'v', 'h', 's', 'z']:
            if k in chunk_states:
                full_states[k].append(jax.device_put(chunk_states[k], jax.devices('cpu')[0]))

    # STEP 3: Stitching
    final_states = {}
    for k in ['x', 'v', 'h', 's', 'z']:
        if full_states[k]:
            final_states[k] = jnp.concatenate(full_states[k], axis=0)

    model['states'] = final_states

    # Recalculate full noise_prior if needed
    if 'conf' in data and config.get('error_estimator', None) is not None:
        with jax.default_device(jax.devices("cpu")[0]):
            model['noise_prior'] = estimate_error(jnp.array(data['conf']), **config['error_estimator'])
    else:
        model['noise_prior'] = 1.0

    model['seed'] = prior
    return model


# --- Internal Kernel for Resampling ---
def _single_sequence_resample(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

_kernel_vmap = jax.vmap(
    _single_sequence_resample, 
    in_axes=(0, 0, 0, 0, 0, None, None, None, 0) 
)

@jax.jit
def _resample_scales_kernel_batched(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    return _kernel_vmap(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0)


def batched_resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
    """
    A memory-efficient drop-in replacement for `gibbs.resample_scales`.
    Processes sequences in batches to avoid OOM during the squared error calculation.
    """
    BATCH_SIZE = 10 
    N = Y.shape[0]
    seeds = jr.split(seed, N)
    results = []

    is_s0_per_seq = (hasattr(s_0, "shape") and s_0.shape[0] == N)
    
    def pad_arr(arr, target_len):
        pad_size = target_len - arr.shape[0]
        padding = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
        return jnp.pad(arr, padding, mode='edge')

    standard_inputs = [seeds, Y, x, v, h]
    
    iterator = range(0, N, BATCH_SIZE) 
    
    for i in iterator:
        start, end = i, min(i + BATCH_SIZE, N)
        actual_len = end - start
        
        batch_args = [arr[start:end] for arr in standard_inputs]
        
        if is_s0_per_seq:
            batch_s_0 = s_0[start:end]
        elif hasattr(s_0, 'shape'):
            batch_s_0 = jnp.repeat(s_0[None, ...], actual_len, axis=0)
        else:
            batch_s_0 = jnp.full((actual_len,), s_0)

        if actual_len < BATCH_SIZE:
            batch_args = [pad_arr(a, BATCH_SIZE) for a in batch_args]
            batch_s_0  = pad_arr(batch_s_0, BATCH_SIZE)

        s_out = _resample_scales_kernel_batched(*batch_args, Cd, sigmasq, nu_s, batch_s_0)
        s_out.block_until_ready()

        results.append(s_out[:actual_len])

    return jnp.concatenate(results, axis=0)
