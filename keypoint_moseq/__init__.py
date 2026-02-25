# -----------------------------------------------------------------------------
# JAX compatibility shim for chex 0.1.6 with newer JAX versions
# chex expects deprecated jax.interpreters.* APIs that were removed
# Modern JAX unifies all array types into jax.Array
# -----------------------------------------------------------------------------
import jax

# Shim for jax.interpreters.pxla.ShardedDeviceArray
if not hasattr(jax.interpreters.pxla, 'ShardedDeviceArray'):
    jax.interpreters.pxla.ShardedDeviceArray = jax.Array

# Shim for jax.interpreters.batching.BatchTracer
if not hasattr(jax.interpreters.batching, 'BatchTracer'):
    jax.interpreters.batching.BatchTracer = jax.Array

# Shim for jax.interpreters.xla.DeviceArray and _DeviceArray
if not hasattr(jax.interpreters.xla, 'DeviceArray'):
    jax.interpreters.xla.DeviceArray = jax.Array
if not hasattr(jax.interpreters.xla, '_DeviceArray'):
    jax.interpreters.xla._DeviceArray = jax.Array

# NumPy 2.0 compatibility shim
# np.bool8 was removed in numpy 2.0, bokeh still uses it
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
# -----------------------------------------------------------------------------

# use double-precision by default
from jax import config

config.update("jax_enable_x64", True)

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
