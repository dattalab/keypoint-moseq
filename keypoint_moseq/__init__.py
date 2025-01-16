# use double-precision by default
from jax import config

config.update("jax_enable_x64", True)

# simple warning formatting
import warnings

warnings.formatwarning = lambda msg, *a: str(msg)

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
