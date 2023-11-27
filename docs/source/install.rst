Local installation
------------------

- Total installation time is around 10 minutes.
- The first import of keypoint_moseq after installation can take a few minutes.
- If you experience any issues, reach out to us on `slack <https://join.slack.com/t/moseqworkspace/shared_invite/zt-151x0shoi-z4J0_g_5rwJDlO1IfCU34A>`_! We're happy to help.

.. note::

   If using Windows, make sure to run all the commands below from an Anaconda Prompt.


Install using conda
~~~~~~~~~~~~~~~~~~~



Use conda environment files to automatically install the appropriate GPU drivers and other dependencies. Start by cloning the repository::

   git clone https://github.com/dattalab/keypoint-moseq
   cd keypoint-moseq

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

Activate the new environment::

   conda activate keypoint_moseq


To run keypoint-moseq in jupyter, either launch jupyterlab directly from the `keypoint_moseq` environment or register a globally-accessible jupyter kernel as follows::

   python -m ipykernel install --user --name=keypoint_moseq
   
   
.. note::

   If you are using Windows with a GPU and see the error ``XlaRuntimeError: UNKNOWN: no kernel image is available for execution on the device`` try updating your GPU drivers to the latest version. 


Install using pip
~~~~~~~~~~~~~~~~~

.. note::

   If you are using Windows with a GPU and would like to pip install keypoint-moseq, then you must also have CUDA 11.1 / cuDNN 8.2 installed system-wide (i.e. through the usual Windows OS). Furthermore, if you are using Windows 11, you must use Windows Subsystem for Linux.

Create a new conda environment with python 3.9::

   conda create -n keypoint_moseq python=3.9
   conda activate keypoint_moseq

Install jax using one of the lines below::

   # MacOS or Linux (CPU)
   pip install "jax[cpu]==0.3.22" -f https://storage.googleapis.com/jax-releases/jax_releases.html

   # MacOS or Linux (GPU)
   pip install "jax[cuda11_cudnn82]==0.3.22" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Windows (CPU)
   pip install jax==0.3.22 https://whls.blob.core.windows.net/unstable/cpu/jaxlib-0.3.22-cp39-cp39-win_amd64.whl

   # Windows (GPU)
   pip install jax==0.3.22 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.22+cuda11.cudnn82-cp39-cp39-win_amd64.whl

Install `keypoint-moseq <https://github.com/dattalab/keypoint-moseq>`_::

   pip install keypoint-moseq

To run keypoint-moseq in jupyter, either launch jupyterlab directly from the ``keypoint_moseq`` environment or register a globally-accessible jupyter kernel as follows::

   python -m ipykernel install --user --name=keypoint_moseq
