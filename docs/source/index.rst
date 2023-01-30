Keypoint MoSeq
==============

.. image:: logo.jpg
   :align: center


Installation
------------


To use a GPU (recommended), install the appropriate driver and CUDA version. CUDA ≥11.1 and cuDNN ≥8.2 are required. `This section of the DeepLabCut docs <https://deeplabcut.github.io/DeepLabCut/docs/installation.html#gpu-support>`_ may be helpful. Next use `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_  or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to configure a new python environment called ``keypoint_moseq``:

.. code-block::

   conda create -n keypoint_moseq python=3.9
   conda activate keypoint_moseq

   # make te environment available to jupyter
   python -m ipykernel install --user --name=keypoint_moseq

   # Include the following line if installing on Windows
   # conda install -c conda-forge pytables

Install jax using one of the lines below

.. code-block::

   # MacOS or Linux (CPU)
   pip install "jax[cpu]"

   # MacOS or Linux (GPU)
   pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Windows (CPU)
   pip install jax https://whls.blob.core.windows.net/unstable/cpu/jaxlib-0.3.22-cp39-cp39-win_amd64.whl

   # Windows (GPU)
   pip install jax https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.22+cuda11.cudnn82-cp39-cp39-win_amd64.whl


Install `jax-moseq <https://github.com/dattalab/jax-moseq>`_ followed by `keypoint-moseq <https://github.com/dattalab/keypoint-moseq>`_

.. code-block::

   pip install -U git+https://github.com/dattalab/jax-moseq
   pip install -U git+https://github.com/dattalab/keypoint-moseq



.. toctree::
   :caption: Tutorial

   tutorial


.. toctree::
   :caption: API Documentation

   fitting
   viz
   io
   util
   calibration
