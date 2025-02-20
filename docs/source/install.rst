Local installation
==================

- Total installation time is around 10 minutes.
- The first import of keypoint_moseq after installation can take a few minutes.
- If you experience any issues, reach out to us on `slack <https://join.slack.com/t/moseqworkspace/shared_invite/zt-151x0shoi-z4J0_g_5rwJDlO1IfCU34A>`_! We're happy to help.

.. note::

   If using Windows, make sure to run all the commands below from an Anaconda Prompt.

.. note::

   Keypoint moseq supports the same platforms as `jax <https://github.com/jax-ml/jax?tab=readme-ov-file#supported-platforms>`_. That is, it supports CPU and GPU installations on linux systems, and CPU installations on MacOS and Windows systems. GPU on WSL2 is considered 'experimental'.

Create a new conda environment with python 3.10::

   conda create -n keypoint_moseq python=3.10
   conda activate keypoint_moseq

Then use pip to install the version of keypoint moseq that you want::

   pip install keypoint-moseq # CPU only
   pip install keypoint-moseq[cuda] # GPU with CUDA 12

To run keypoint-moseq in jupyter, either launch jupyterlab directly from the ``keypoint_moseq`` environment or register a globally-accessible jupyter kernel as follows::

   python -m ipykernel install --user --name=keypoint_moseq
