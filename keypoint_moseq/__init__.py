# use double-precision by default
from jax import config
config.update("jax_enable_x64", True)

# simple warning formatting
import warnings
warnings.formatwarning = lambda msg, *a: str(msg)

from .util import *

from .io import *

from .calibration import (
    noise_calibration,
)

from .fitting import (
    revert, 
    fit_model, 
    apply_model, 
    extract_results,
    resume_fitting, 
    update_hypparams,
    kappa_scan
)

from .viz import (
    plot_pcs, 
    plot_scree, 
    plot_progress, 
    plot_syllable_frequencies,
    plot_duration_distribution,
    generate_crowd_movies, 
    generate_grid_movies,
    generate_trajectory_plots,
    plot_kappa_scan
)

from .analysis import (
    compute_moseq_df, 
    compute_stats_df, 
    plotting_fingerprint,
    create_fingerprint_dataframe, 
    plot_syll_stats_with_sem,
    get_group_trans_mats,
    changepoint_analysis,
)

from jax_moseq.models.keypoint_slds import (
    fit_pca, 
    init_model
)
from . import _version
__version__ = _version.get_versions()['version']
