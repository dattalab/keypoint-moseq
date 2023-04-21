Installation
------------

- For a Mac with an M1 chip, follow `Install using conda`_.
- The first import of ``keypoint_moseq`` after installation can take a few minutes.

Install using pip
~~~~~~~~~~~~~~~~~

To use a GPU (recommended), install the appropriate driver and CUDA version. CUDA ≥11.1 and cuDNN ≥8.2 are required. `This section of the DeepLabCut docs <https://deeplabcut.github.io/DeepLabCut/docs/installation.html#gpu-support>`_ may be helpful. Next use `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_  or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to configure a new python environment called ``keypoint_moseq``::

   conda create -n keypoint_moseq python=3.9
   conda activate keypoint_moseq

   # Include the following line if installing on Windows
   # conda install -c conda-forge pytables

Install jax using one of the lines below::

   # MacOS or Linux (CPU)
   pip install "jax[cpu]"

   # MacOS or Linux (GPU)
   pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Windows (CPU)
   pip install jax==0.3.22 https://whls.blob.core.windows.net/unstable/cpu/jaxlib-0.3.22-cp39-cp39-win_amd64.whl

   # Windows (GPU)
   pip install jax==0.3.22 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.22+cuda11.cudnn82-cp39-cp39-win_amd64.whl


Install `keypoint-moseq <https://github.com/dattalab/keypoint-moseq>`_::

   pip install keypoint-moseq

Make the environment available to jupyter::

   python -m ipykernel install --user --name=keypoint_moseq

Install using conda
~~~~~~~~~~~~~~~~~~~

As an alternative, you can install directly from conda environment files. This will automatically install the appropriate GPU drivers and other dependencies. Start by cloning the repository::

   git clone https://github.com/dattalab/keypoint-moseq
   cd keypoint-moseq # (`chdir keypoint-moseq` if using Windows)

Install the appropriate conda environment for your platform::

   # Windows (CPU-only)
   conda env create -f conda_envs/environment.win64_cpu.yml

   # Windows (GPU)
   conda env create -f conda_envs/environment.win64_gpu.yml

   # Linux (CPU-only)
   conda env create -f conda_envs/environment.linux_cpu.yml

   # Linux (GPU)
   conda env create -f conda_envs/environment.linux_gpu.yml

   # Mac (CPU-only)
   conda env create -f conda_envs/environment.mac_cpu.yml

See `this issue <https://github.com/dattalab/keypoint-moseq/issues/5>`_ for updates regarding Apple Silicon (M1/M2) support. For now, you can use the Mac (CPU) version on newer Macs.

Activate the new environment::

   conda activate keypoint_moseq

Troubleshooting
---------------

- ``UNKNOWN: no kernel image is available for execution on the device``

  If you're running into issues when trying to use the GPU-accelerated version, you might see this error message::

     jaxlib.xla_extension.XlaRuntimeError: UNKNOWN: no kernel image is available for execution on the device

  First, check if jax can detect your GPU::

     python -c "import jax; print(jax.default_backend())

  The result should be "gpu". If it isn't, then you might not be using the right version of ``cudatoolkit`` or ``cudnn``. If you installed these via ``conda``, you can check by doing a ``conda list | grep cud``. If you are on the right versions, try `updating your GPU driver to the latest version <https://nvidia.com/drivers>`_.


License
-------

MoSeq is freely available for academic use under a license provided by Harvard University. Please refer to the `license file <https://github.com/dattalab/keypoint-moseq/blob/main/LICENSE.md>`_ for details. If you are interested in using MoSeq for commercial purposes please contact Bob Datta directly at srdatta@hms.harvard.edu, who will put you in touch with the Harvard Technology Transfer office.