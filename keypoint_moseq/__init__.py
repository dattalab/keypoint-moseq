from jax import config
config.update("jax_enable_x64", True)

from .util import *

from .io import *

from .calibration import (
    noise_calibration,
)

from .fitting import (
    revert, 
    fit_model, 
    apply_model, 
    resume_fitting, 
    update_hypparams,
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