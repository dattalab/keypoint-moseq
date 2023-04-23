For problems that cannot be fixed using any of the steps below, please open a `github issue <https://github.com/dattalab/keypoint-moseq/issues>`_.


Dead kernel
-----------

On Windows, GPU out of memory (OOM) errors may cause silent kernel failure. To determine whether this is the likely cause, compare keypoint MoSeq's expected memory usage during model fitting (roughly 1MB per 100 frames of data) to the total memory available (VRAM) on your GPU. To check the total available VRAM, use ``nvidia-smi`` for Mac and Linux or the Task Manager in Windows. 


Out of memory
-------------

There are two main causes of GPU out of memory (OOM) errors:

1. **Multiple instances of keypoint MoSeq are running on the same GPU.** 

  This can happen if you're running multiple notebooks or scripts at the same time. Since JAX preallocates 90% of the GPU when it is first initialized (i.e. after running ``import keypoint_moseq``), there is very little memory left for the second notebook/script. To fix this, you can either shutdown the kernels of the other notebooks/scripts or use a different GPU.


2. **Large datasets.** 

  Keypoint MoSeq requires ~1MB GPU memory for each 100 frames of data during model fitting. If your GPU isn't big enough, try one of the following:

  - Use `Google colab <https://colab.research.google.com/github/dattalab/keypoint-moseq/blob/main/docs/keypoint_moseq_colab.ipynb>`_. 

    - Colab provides free access to GPUs with 16GB of VRAM.

    - Larger GPUs can be accessed using colab pro. 

  - Switch to single-precision computing by running the code below immediarely after importing keypoint MoSeq. Note that this may result in numerical instability which will cause NaN values to appear during fitting. Keypoint MoSeq will abort fitting if this occurs::

      import jax
      jax.config.update('jax_enable_x64', False)

    
  - Fit to a subset of the data, then apply the model to the rest of the data. 

    - To fit a subset of the data, specify the subset as a list of paths during data loading::

        initial_data = ['path/to/file1.h5', 'path/to/file2.h5']
        coordinates, confidences = kpms.load_deeplabcut_results(initial_data)

    - After model fitting, apply the model serially to new data as follows::

        checkpoint = kpms.load_checkpoint(project_dir=project_dir, name=name)

        new_data_batch1 = ['path/to/file3.h5', 'path/to/second/file4.h5']
        new_data_batch2 = ['path/to/file5.h5', 'path/to/second/file6.h5']

        for batch in [initial_data, new_data_batch1, new_data_batch2]:

            coordinates, confidences = kpms.load_deeplabcut_results(batch)

            results = kpms.apply_model(
                coordinates=coordinates, confidences=confidences, 
                use_saved_states=False, pca=kpms.load_pca(project_dir),
                project_dir=project_dir, **config(), **checkpoint, num_iters=5)



NaNs during fitting
-------------------

NaNs are much more likely with single-precision computing. Check the precision using::

    import jax
    jax.config.read('jax_enable_x64')

If the output is ``True`` (i.e. JAX is using double-precision), then please contact calebsw@gmail.com and include the data used for fitting as well as the most recent model checkpoint. 


Installation errors
-------------------

- ``UNKNOWN: no kernel image is available for execution on the device``

  If you're running into issues when trying to use the GPU-accelerated version, you might see this error message::

     jaxlib.xla_extension.XlaRuntimeError: UNKNOWN: no kernel image is available for execution on the device

  First, check if jax can detect your GPU::

     python -c "import jax; print(jax.default_backend())

  The result should be "gpu". If it isn't, then you might not be using the right version of ``cudatoolkit`` or ``cudnn``. If you installed these via ``conda``, you can check by doing a ``conda list | grep cud``. If you are on the right versions, try `updating your GPU driver to the latest version <https://nvidia.com/drivers>`_.
