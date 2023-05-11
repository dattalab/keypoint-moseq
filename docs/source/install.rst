Local installation
------------------

- To use keypoint MoSeq without a local installation, try `Google colab <colab>`_.
- Note that the first import of ``keypoint_moseq`` after installation can take a few minutes.
- If you are using a GPU, confirm that jax has access to it once installation is complete::

   python -c "import jax; print(jax.devices())"


Install using conda
~~~~~~~~~~~~~~~~~~~

Use conda environment files to automatically install the appropriate GPU drivers and other dependencies. Start by cloning the repository::

   git clone https://github.com/dattalab/keypoint-moseq
   cd keypoint-moseq # (`chdir keypoint-moseq` if using Windows)

Install the appropriate conda environment for your platform::

   # Windows (CPU-only)
   conda env create -f conda_envs\environment.win64_cpu.yml

   # Windows (GPU)
   conda env create -f conda_envs\environment.win64_gpu.yml

   # Linux (CPU-only)
   conda env create -f conda_envs/environment.linux_cpu.yml

   # Linux (GPU)
   conda env create -f conda_envs/environment.linux_gpu.yml

   # Mac (CPU-only)
   conda env create -f conda_envs/environment.mac_cpu.yml

[Windows users only!] update pytables::

   conda update -c conda-forge pytables

Activate the new environment::

   conda activate keypoint_moseq

To use the keypoint_moseq environment in a notebook, either launch jupyterlab (`jupyter lab`) or register the environment as a jupyter notebook kernel as follows::

   python -m ipykernel install --user --name=keypoint_moseq


Install using pip
~~~~~~~~~~~~~~~~~

..note::
   If you are using Windows with a GPU, CUDA 11.1 / cuDNN 8.2 is required. If you have a different version of CUDA installed, then follow the directions to `Install using conda`_.

Create a new conda environment::

   conda create -n keypoint_moseq python=3.9
   conda activate keypoint_moseq

Install jax using one of the lines below. 

   # MacOS or Linux (CPU)
   pip install "jax[cpu]==0.3.22"

   # MacOS or Linux (GPU)
   pip install "jax[cuda11_cudnn82]==0.3.22" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Windows (CPU)
   pip install jax==0.3.22 https://whls.blob.core.windows.net/unstable/cpu/jaxlib-0.3.22-cp39-cp39-win_amd64.whl

   # Windows (GPU)
   pip install jax==0.3.22 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.22+cuda11.cudnn82-cp39-cp39-win_amd64.whl


Install `keypoint-moseq <https://github.com/dattalab/keypoint-moseq>`_::

   pip install keypoint-moseq

[Windows users only!] update pytables::

   conda update -c conda-forge pytables

Make the environment available to jupyter::

   python -m ipykernel install --user --name=keypoint_moseq

