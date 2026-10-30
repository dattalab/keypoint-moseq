# use double-precision by default
from jax import config

config.update("jax_enable_x64", True)


# Keep compiled programs between sessions in the user's cache directory,
# unless the user has chosen a cache directory. Every program is cached,
# since most of them compile in less than JAX's default threshold of 1 s.
# This runs before anything is compiled, which is when JAX decides whether
# to use the cache.
def _use_persistent_compilation_cache():
    import os
    import platformdirs

    if (
        config.jax_compilation_cache_dir is None
        and "JAX_COMPILATION_CACHE_DIR" not in os.environ
    ):
        cache_dir = os.path.join(platformdirs.user_cache_dir("keypoint_moseq"), "jax")
        config.update("jax_compilation_cache_dir", cache_dir)
        config.update("jax_persistent_cache_min_compile_time_secs", 0)


_use_persistent_compilation_cache()
del _use_persistent_compilation_cache

# simple warning formatting
import warnings

warnings.formatwarning = lambda msg, *a: str(msg)

# Suppress harmless warnings thrown by Python 3.12's stricter syntax warnings
warnings.filterwarnings('ignore', category=SyntaxWarning, module='panel.*')

from .io import *
from .viz import *
from .util import *
from .fitting import *
from .analysis import *
from .calibration import noise_calibration

from jax_moseq.models.keypoint_slds import fit_pca
from jax_moseq.utils import get_frequencies, get_durations

from . import _version

__version__ = _version.get_versions()["version"]
