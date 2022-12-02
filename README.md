# Keypoint MoSeq

Motion Sequencing ([MoSeq](https://www.cell.com/neuron/fulltext/S0896-6273(15)01037-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627315010375%3Fshowall%3Dtrue)) for keypoint tracking data. Check out our [colab notebook](https://github.com/dattalab/keypoint-moseq/blob/main/examples/keypointMoSeq_demo_colab.ipynb) for a tutorial. 

## Installation

1. If you plan to use a GPU (recommended), install the appropriate driver and CUDA version. CUDA ≥11.1 and cuDNN ≥8.2 are required. [This section of the DeepLabCut docs](https://deeplabcut.github.io/DeepLabCut/docs/installation.html#gpu-support) may be helpful.


2. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Create and activate an environment called `keypoint_moseq` with python 3.9:
```
conda create -n keypoint_moseq python=3.9
conda activate keypoint_moseq

# Include the following line if installing on Windows
# conda install -c conda-forge pytables
```

3. Install jax
```
# MacOS and Linux users (CPU)
pip install "jax[cpu]"

# MacOS and Linux users (GPU)
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Windows users (CPU)
pip install jax https://whls.blob.core.windows.net/unstable/cpu/jaxlib-0.3.22-cp39-cp39-win_amd64.whl

# Windows users (GPU)
pip install jax https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.22+cuda11.cudnn82-cp39-cp39-win_amd64.whl
```

4. Install [jax-moseq](https://github.com/dattalab/jax-moseq) followed by [keypoint-moseq](https://github.com/dattalab/keypoint-moseq):
```
pip install -U git+https://github.com/dattalab/jax-moseq
pip install -U git+https://github.com/dattalab/keypoint-moseq
```

5. Make the new environment accessible in jupyter 
```
python -m ipykernel install --user --name=keypoint_moseq
```
