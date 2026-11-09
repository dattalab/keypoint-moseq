from jax import config


# JAX compresses cache entries with zstd when compression.zstd (Python 3.14+)
# or the zstandard package is available, and with zlib otherwise, and an
# environment can't read entries compressed the other way. Returns the
# method JAX uses here, so that each method gets its own cache folder.
def _cache_compression():
    try:
        from jax._src import compilation_cache as cc

        return "zstd" if cc.zstd or cc.zstandard else "zlib"
    except (ImportError, AttributeError):
        for module in ["compression.zstd", "zstandard"]:
            try:
                __import__(module)
                return "zstd"
            except ImportError:
                pass
        return "zlib"


# Keep compiled programs between sessions in the user's cache directory,
# unless the user has chosen a cache directory. Every program is cached,
# since most of them compile in less than JAX's default threshold of 1 s,
# unless the user has set a threshold.
def _use_persistent_compilation_cache():
    import os
    import platformdirs

    if (
        config.jax_compilation_cache_dir is None
        and "JAX_COMPILATION_CACHE_DIR" not in os.environ
    ):
        cache_dir = os.path.join(
            platformdirs.user_cache_dir("keypoint_moseq"), "jax", _cache_compression()
        )
        config.update("jax_compilation_cache_dir", cache_dir)
        if (
            config.jax_persistent_cache_min_compile_time_secs == 1.0  # JAX's default
            and "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ
        ):
            config.update("jax_persistent_cache_min_compile_time_secs", 0)


_use_persistent_compilation_cache()
del _use_persistent_compilation_cache, _cache_compression

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
