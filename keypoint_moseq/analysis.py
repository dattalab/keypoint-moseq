from math import ceil
from matplotlib.lines import Line2D
from cytoolz import sliding_window, complement
from collections import OrderedDict
from tqdm.auto import tqdm
import networkx as nx
import warnings
import os
import yaml

import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

from bokeh.io import output_notebook, show
from IPython.display import display
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations

from tqdm import tqdm
from collections import defaultdict, OrderedDict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from copy import deepcopy
from cytoolz import sliding_window
from os.path import join, exists

from keypoint_moseq.widgets import GroupSettingWidgets, SyllableLabeler

from glob import glob
from keypoint_moseq.widgets import GroupSettingWidgets, SyllableLabeler


from keypoint_moseq.util import stateseq_stats, sample_instances, get_trajectories, get_frequencies
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, dendrogram

# imports for changepoint analysis
from statsmodels.stats.multitest import fdrcorrection
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal import argrelextrema
from keypoint_moseq.util import filter_angle, filtered_derivative, permute_cyclic, get_syllable_instances
from keypoint_moseq.io import format_data, load_results, load_deeplabcut_results
from jax_moseq.models.keypoint_slds import align_egocentric
from jax_moseq.utils import unbatch
na = np.newaxis

output_notebook()


def interactive_group_setting(project_dir, model_dirname):
    """start the interactive group setting widget

    Parameters
    ----------
    project_dir : str
        the path to the project directory
    model_dirname : str
        the name of the model directory

    """

    index_filepath = os.path.join(project_dir, 'index.yaml')

    if not os.path.exists(index_filepath):
        generate_index(project_dir, model_dirname, index_filepath)

    # display the widget
    index_grid = GroupSettingWidgets(index_filepath)
    display(index_grid.clear_button, index_grid.group_set)
    display(index_grid.qgrid_widget)
    return index_filepath

# fingerprint
def robust_min(v):
    """find the 1% quantile of the input vector and return it as the robust minimum value
    Parameters
    ----------
    v : numpy.array
        the array to find robust minimum from

    Returns
    -------
    float
        the robust minimum value of the array
    """

    return v.quantile(0.01)


def robust_max(v):
    """find the 99% quantile of the input vector and return it as the robust maximum value
    Parameters
    ----------
    v : numpy.array
        the array to find robust maximum from

    Returns
    -------
    float
        the robust maximum value of the array
    """

    return v.quantile(0.99)


def _apply_to_col(df, fn, **kwargs):
    return df.apply(fn, axis=0, **kwargs)


def create_fingerprint_dataframe(moseq_df, stats_df, n_bins = 50,
                                 groupby_list=['group', 'name'], range_type='robust',
                                 scalars=['heading', 'velocity_px_s']):
    """create a summary dataframe to visualize the data as the MoSeq fingerprint (behvavoiral summary) plot

    Parameters
    ----------
    scalar_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    mean_df : pandas.DataFrame
        the summay statistics dataframe for syllable frequencies and kinematic values
    stat_type : str, optional
        the statistics to plot, by default 'mean'
    n_bins : int, optional
        the number of bins to use for the histogram, by default 100
    groupby_list : list, optional
        the list of column names to group by, by default ['group','name']
    range_type : str, optional
        the range type to use for the heatmap, by default 'robust'
    scalars : list, optional
        the list of scalars to include in the fingerprint, by default ['heading', 'velocity_px_s']

    Returns
    -------
    fingerprint_df : pandas.DataFrame
        the fingerprint dataframe to be used for plotting
    pandas.DataFrame
        the range dataframe of the values with the selcted range type
    """

    # set statistics to plot to mean
    stat_type='mean'

    # deep copy the dfs
    moseq_df = moseq_df.copy()
    stats_df = stats_df.copy()

    # pivot mean_df to be groupby x syllable
    syll_summary =stats_df.pivot_table(
        index=groupby_list, values='frequency', columns='syllable')
    syll_summary.columns = pd.MultiIndex.from_arrays(
        [['MoSeq'] * syll_summary.shape[1], syll_summary.columns])
    min_p = syll_summary.min().min()
    max_p = syll_summary.max().max()

    ranges = moseq_df.reset_index(drop=True)[scalars].agg(
        ['min', 'max', robust_min, robust_max])
    # add syllable ranges to this df
    # robust mode would exclude the extreme values
    ranges['MoSeq'] = [min_p, max_p, min_p, max_p]
    range_idx = ['min', 'max'] if range_type == 'full' else [
        'robust_min', 'robust_max']

    def bin_scalars(data: pd.Series, n_bins=50, range_type='full'):
        _range = ranges.loc[range_idx, data.name]
        bins = np.linspace(_range.iloc[0], _range.iloc[1], n_bins)

        binned_data = data.value_counts(normalize=True, sort=False, bins=bins)
        binned_data = binned_data.sort_index().reset_index(drop=True)
        binned_data.index.name = 'bin'
        return binned_data

    binned_scalars = moseq_df.groupby(groupby_list)[scalars].apply(
        _apply_to_col, fn=bin_scalars, range_type=range_type, n_bins=n_bins)

    scalar_fingerprint = binned_scalars.pivot_table(
        index=groupby_list, columns='bin', values=binned_scalars.columns)

    fingerprints = scalar_fingerprint.join(syll_summary, how='outer')

    return fingerprints, ranges.loc[range_idx]



def plot_fingerprint(summary, range_dict, save_dir=None, preprocessor_type='minmax',
                         num_level=1, level_names=['Group'], vmin=None, vmax=None,
                         figsize=(10, 6), fontsize=12, plot_columns=['heading', 'velocity_px_s', 'MoSeq'],
                         col_names=[('Heading', 'a.u.'), ('Velocity', 'px/s'), ('MoSeq', 'Syllable ID')]):
    """plot the fingerprint plot from fingerprint dataframe

    Parameters
    ----------
    summary : pandas.DataFrame
        the fingerprint dataframe to be used for plotting
    range_dict : pandas.DataFrame 
        the range dataframe of the values with the selcted range type
    preprocessor_type : str, optional
        the type of sklearn preprocessor to use to process data to plot, by default 'minmax'
    num_level : int, optional
        the number of levels to group by for plotting, by default 1
    level_names : list, optional
        the list of level names to use for plotting, by default ['Group']
    vmin : float, optional
        min value to plot, by default None, the min value for plotting will be found from the data
    vmax : float, optional
        max value to plot, by default None, the max value for plotting will be found from the data
    figsize : tuple, optional
        the size of the figure, by default (10,15)
    plot_columns : list, optional
        the columns to plot the fingerprint, by default ['heading','velocity_px_s', 'MoSeq']
    col_names : list, optional
        column names for the fingerprint plot, by default [('Heading','a.u.'),('velocity','px/s'), ('MoSeq','Syllable ID')]

    Raises
    ------
    Exception
        too many levels to unpack. num_level should be less than the number of levels in the summary dataframe
    """

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    assert preprocessor_type in ['minmax', 'standard', 'none']
    if preprocessor_type == 'minmax':
        preprocessor = MinMaxScaler()
    elif preprocessor_type == 'standard':
        preprocessor = StandardScaler()
    else:
        preprocessor = None

    name_map = dict(zip(plot_columns, col_names))

    # get level label (group)
    level = summary.index.get_level_values(0)
    level_label = LabelEncoder().fit_transform(level)
    find_mid = (np.diff(np.r_[0, np.argwhere(
            np.diff(level_label)).ravel(), len(level_label)])/2).astype('int32')
    level_ticks = np.r_[0, np.argwhere(
            np.diff(level_label)).ravel()] + find_mid        

    levels = []
    level_plot = []
    level_ticks = []

    # col_num = 1 (for group level) + column in summary
    col_num = 1 + len(plot_columns)

    # https://matplotlib.org/stable/tutorials/intermediate/gridspec.html
    fig = plt.figure(1, figsize=figsize, facecolor='white')

    gs = GridSpec(2, col_num, wspace=0.1, hspace=0.1,
                  width_ratios=[1]*num_level+[8]*(col_num-num_level), height_ratios=[10, 0.1], figure=fig)

    # plot the group level
    temp_ax = fig.add_subplot(gs[0, 0])
    temp_ax.set_title('Label', fontsize=fontsize)
    temp_ax.imshow(level_label[:, np.newaxis],
                    aspect='auto', cmap='Set3')
    plt.yticks(level_ticks, level
                [level_ticks], fontsize=fontsize)

    temp_ax.get_xaxis().set_ticks([])

    # compile data to plot while recording vmin and vmax in the data
    plot_dict = {}
    # initialize vmin and vmax
    temp_vmin = np.Inf
    temp_vmax = -np.Inf

    for col in plot_columns:
        data = summary[col].to_numpy()
        # process data with preprocessor
        if preprocessor is not None:
            data = preprocessor.fit_transform(data.T).T

        if np.min(data) < temp_vmin:
            temp_vmin = np.min(data)
        if np.max(data) > temp_vmax:
            temp_vmax = np.max(data)

        plot_dict[col] = data

    if vmin is None:
        vmin = temp_vmin
    if vmax is None:
        vmax = temp_vmax

    # plot the data
    for i, col in enumerate(plot_columns):
        name = name_map[col]
        temp_ax = fig.add_subplot(gs[0, i + num_level])
        temp_ax.set_title(name[0], fontsize=fontsize)
        data = plot_dict[col]

        # top to bottom is 0-20 for y axis
        if col == 'MoSeq':
            extent = [summary[col].columns[0],
                      summary[col].columns[-1], len(summary) - 1, 0]
        else:
            extent = [range_dict[col].iloc[0],
                      range_dict[col].iloc[1], len(summary) - 1, 0]

        pc = temp_ax.imshow(
            data, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, extent=extent)
        temp_ax.set_xlabel(name[1], fontsize=fontsize)
        temp_ax.set_xticks(np.linspace(
            np.ceil(extent[0]), np.floor(extent[1]), 6).astype(int))
        # https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
        temp_ax.set_yticks([])
        temp_ax.axis = 'tight'
        plt.xticks(fontsize=int(fontsize*.75))

    # plot colorbar
    # cb = fig.add_subplot(gs[1, -1])
    # plt.colorbar(pc, cax=cb, orientation='horizontal')

    # specify labels for feature scaling
    # if preprocessor_type == 'minmax':
    #     cb.set_xlabel('Min Max')
    # elif preprocessor_type == 'standard':
    #     cb.set_xlabel('Standardized')
    # else:
    #     cb.set_xlabel('Percentage Usage')

    # saving the figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir=os.path.join(project_dir, model_dirname, 'figures')
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(join(save_dir, 'moseq_fingerprint.pdf'))
    fig.savefig(join(save_dir, 'moseq_fingerprint.png'))


def label_syllables(project_dir, model_dirname, movie_type='grid'):
    """label syllables in the syllable grid movie

    Parameters
    ----------
    project_dir : str
        the path to the project directory
    model_dirname : str
        the name of the model directory
    movie_type : str, optional
        the type of movie to label, ('grid' or 'crowd')
    """

    output_notebook()

    grid_movies = glob(os.path.join(project_dir, model_dirname, 'grid_movies', '*.mp4'))
    crowd_movies = glob(os.path.join(project_dir, model_dirname, 'crowd_movies', '*.mp4'))

    if movie_type == 'grid':
        assert len(grid_movies) > 0, (
            'No grid movies found. Please run `generate_grid_movies` as described in the docs: '
            'https://keypoint-moseq.readthedocs.io/en/latest/tutorial.html#crowd-grid-movies')
        
    elif movie_type == 'crowd':
        assert len(crowd_movies) > 0, (
            'No crowd movies found. Please run `generate_crowd_movies` as described in the docs: '
            'https://keypoint-moseq.readthedocs.io/en/latest/tutorial.html#crowd-grid-movies')

    # construct the syllable info path
    syll_info_path = os.path.join(project_dir, model_dirname, "syll_info.yaml")

    # generate a new syll_info yaml file
    if not os.path.exists(syll_info_path):
        # parse model results
        model_results = load_results(project_dir, model_dirname)
        unique_sylls = np.unique(np.concatenate([file['syllables_reindexed'] for file in model_results.values()]))
        # construct the syllable dictionary
        syll_dict = {int(i): {'label': '', 'desc': '', 'movie_path': [], 'group_info': {}} for i in unique_sylls}
        # record the movie paths
        for movie_path in grid_movies:
            syll_index = int(os.path.splitext(os.path.basename(movie_path))[0][8:])
            syll_dict[syll_index]['movie_path'].append(movie_path)
        for movie_path in crowd_movies:
            syll_index = int(os.path.splitext(os.path.basename(movie_path))[0][8:])
            syll_dict[syll_index]['movie_path'].append(movie_path)

        # write to file
        print(syll_info_path)
        with open(syll_info_path, 'w') as file:
            yaml.safe_dump(syll_dict, file, default_flow_style=False)

    # construct the index path
    index_path = os.path.join(project_dir, "index.yaml")

    # create index.yaml if it does not exist
    if not os.path.exists(index_path):
        print('index.yaml does not exist, creating one...')
        generate_index(project_dir, model_dirname, index_path)

    labeler = SyllableLabeler(project_dir, model_dirname, index_path, syll_info_path, movie_type)
    output = widgets.interactive_output(labeler.interactive_syllable_labeler, {'syllables': labeler.syll_select})
    display(labeler.clear_button, labeler.syll_select, output)


def get_tie_correction(x, N_m):
    """assign tied rank values to the average of the ranks they would have received if they had not been tied for Kruskal-Wallis helper function.

    Parameters
    ----------
    x : pd.Series
        syllable usages for a single session.
    N_m : int
        Number of total sessions.

    Returns
    -------
    corrected_rank : float
        average of the inputted tied ranks.
    """

    vc = x.value_counts()
    tie_sum = 0
    if (vc > 1).any():
        tie_sum += np.sum(vc[vc != 1] ** 3 - vc[vc != 1])
    return tie_sum / (12.0 * (N_m - 1))


def run_manual_KW_test(df_usage, merged_usages_all, num_groups, n_per_group, cum_group_idx, n_perm=10000, seed=42):
    """ Run a manual Kruskal-Wallis test compare the results agree with the scipy.stats.kruskal function.

    Parameters
    ----------
    df_usage : pandas.DataFrame
        DataFrame with syllable usages. shape = (N_m, n_syllables)
    merged_usages_all : np.array
        numpy array format of the df_usage DataFrame.
    num_groups : int
        Number of unique groups
    n_per_group : list
        list of value counts for sessions per group. len == num_groups.
    cum_group_idx : list
        list of indices for different groups. len == num_groups + 1.
    n_perm : int, optional
        Number of permuted samples to generate, by default 10000
    seed : int, optional
        Random seed used to initialize the pseudo-random number generator, by default 42

    Returns
    -------
    h_all : np.array
        Array of H-stats computed for given n_syllables; shape = (n_perms, N_s)
    real_ranks : np.array
        Array of syllable ranks, shape = (N_m, n_syllables)
    X_ties : np.array
        1-D list of tied ranks, where if value > 0, then rank is tied. len(X_ties) = n_syllables
    """

    N_m, N_s = merged_usages_all.shape

    # create random index array n_perm times
    rnd = np.random.RandomState(seed=seed)
    perm = rnd.rand(n_perm, N_m).argsort(-1)

    # get degrees of freedom
    dof = num_groups - 1

    real_ranks = np.apply_along_axis(stats.rankdata, 0, merged_usages_all)
    X_ties = df_usage.apply(get_tie_correction, 0, N_m=N_m).values
    KW_tie_correct = np.apply_along_axis(stats.tiecorrect, 0, real_ranks)

    # rank data
    perm_ranks = real_ranks[perm]

    # get square of sums for each group
    ssbn = np.zeros((n_perm, N_s))
    for i in range(num_groups):
        ssbn += (
            perm_ranks[:, cum_group_idx[i]: cum_group_idx[i + 1]].sum(1) ** 2
            / n_per_group[i]
        )

    # h-statistic
    h_all = 12.0 / (N_m * (N_m + 1)) * ssbn - 3 * (N_m + 1)
    h_all /= KW_tie_correct
    p_vals = stats.chi2.sf(h_all, df=dof)

    # check that results agree
    p_i = np.random.randint(n_perm)
    s_i = np.random.randint(N_s)
    kr = stats.kruskal(
        *np.array_split(
            merged_usages_all[perm[p_i, :], s_i], np.cumsum(n_per_group[:-1])
        )
    )
    assert (kr.statistic == h_all[p_i, s_i]) & (
        kr.pvalue == p_vals[p_i, s_i]
    ), "manual KW is incorrect"

    return h_all, real_ranks, X_ties


def dunns_z_test_permute_within_group_pairs(df_usage, vc, real_ranks, X_ties, N_m, group_names, rnd, n_perm):
    """Run Dunn's z-test statistic on combinations of all group pairs, handling pre-computed tied ranks.

    Parameters
    ----------
    df_usage : pandas.DataFrame
        DataFrame containing only pre-computed syllable stats.
    vc : pd.Series
        value counts of sessions in each group.
    real_ranks : np.array
        Array of syllable ranks.
    X_ties : np.array
        1-D list of tied ranks, where if value > 0, then rank is tied
    N_m : int
        Number of sessions.
    group_names : pd.Index
        Index list of unique group names.
    rnd : np.random.RandomState
        Pseudo-random number generator.
    n_perm : int
        Number of permuted samples to generate.

    Returns
    -------
    null_zs_within_group : dict
        dict of group pair keys paired with vector of Dunn's z-test statistics of the null hypothesis.
    real_zs_within_group : dict
        dict of group pair keys paired with vector of Dunn's z-test statistics
    """

    null_zs_within_group = {}
    real_zs_within_group = {}

    A = N_m * (N_m + 1.0) / 12.0

    for (i_n, j_n) in combinations(group_names, 2):
        is_i = df_usage.group == i_n
        is_j = df_usage.group == j_n

        n_mice = is_i.sum() + is_j.sum()

        ranks_perm = real_ranks[(is_i | is_j)][rnd.rand(
            n_perm, n_mice).argsort(-1)]
        diff = np.abs(
            ranks_perm[:, : is_i.sum(), :].mean(1)
            - ranks_perm[:, is_i.sum():, :].mean(1)
        )
        B = 1.0 / vc.loc[i_n] + 1.0 / vc.loc[j_n]

        # also do for real data
        group_ranks = real_ranks[(is_i | is_j)]
        real_diff = np.abs(
            group_ranks[: is_i.sum(), :].mean(
                0) - group_ranks[is_i.sum():, :].mean(0)
        )

        # add to dict
        pair = (i_n, j_n)
        null_zs_within_group[pair] = diff / np.sqrt((A - X_ties) * B)
        real_zs_within_group[pair] = real_diff / np.sqrt((A - X_ties) * B)

    return null_zs_within_group, real_zs_within_group


def compute_pvalues_for_group_pairs(real_zs_within_group, null_zs, df_k_real, group_names, n_perm=10000, thresh=0.05, mc_method="fdr_bh"):
    """Adjust the p-values from Dunn's z-test statistics and computes the resulting significant syllables with the adjusted p-values.

    Parameters
    ----------
    real_zs_within_group : dict
        dict of group pair keys paired with vector of Dunn's z-test statistics
    null_zs : dict
        dict of group pair keys paired with vector of Dunn's z-test statistics of the null hypothesis.
    df_k_real : pandas.DataFrame
        DataFrame of KW test results.
    group_names : pd.Index
        Index list of unique group names.
    n_perm : int, optional
        Number of permuted samples to generate, by default 10000
    thresh : float, optional
        Alpha threshold to consider syllable significant, by default 0.05
    mc_method : str, optional
        Multiple Corrections method to use, by default "fdr_bh"
    verbose : bool, optional
        indicates whether to print out the significant syllable results, by default False

    Returns
    -------
    df_pval_corrected : pandas.DataFrame
        DataFrame containing Dunn's test results with corrected p-values.
    significant_syllables : list
        List of corrected KW significant syllables (syllables with p-values < thresh).
    """

    # do empirical p-val calculation for all group permutation

    p_vals_allperm = {}
    for pair in combinations(group_names, 2):
        p_vals_allperm[pair] = (
            (null_zs[pair] > real_zs_within_group[pair]).sum(0) + 1
        ) / n_perm

    # summarize into df
    df_pval = pd.DataFrame(p_vals_allperm)

    def correct_p(x): return multipletests(
        x, alpha=thresh, method=mc_method)[1]
    df_pval_corrected = df_pval.apply(
        correct_p, axis=1, result_type="broadcast")

    return df_pval_corrected, ((df_pval_corrected[df_k_real.is_sig] < thresh).sum(0))


def run_kruskal(stats_df, statistic='frequency', n_perm=10000, seed=42, thresh=0.05, mc_method='fdr_bh'):
    """Run Kruskal-Wallis test on syllable usage data.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        DataFrame containing syllable usage data.
    statistic : str, optional
        Statistic to use for KW test, by default 'frequency'
    n_perm : int, optional
        Number of permutations to run, by default 10000
    seed : int, optional
        Random seed, by default 42
    thresh : float, optional
        Alpha threshold to consider syllable significant, by default 0.05
    mc_method : str, optional
        Multiple Corrections method to use, by default "fdr_bh"

    Returns
    -------
    df_k_real : pandas.DataFrame
        DataFrame containing KW test results.
    df_pval_corrected : pandas.DataFrame
        DataFrame containing Dunn's test results with corrected p-values.
    significant_syllables : list
        List of corrected KW significant syllables (syllables with p-values < thresh).
    """
    rnd = np.random.RandomState(seed=seed)
    # get grouped mean data
    grouped_data = stats_df.pivot_table(
        index=["group", 'name'], columns="syllable", values=statistic).replace(np.nan, 0).reset_index()
    # compute KW constants
    vc = grouped_data.group.value_counts().loc[grouped_data.group.unique()]
    n_per_group = vc.values
    group_names = vc.index

    cum_group_idx = np.insert(np.cumsum(n_per_group), 0, 0)
    num_groups = len(group_names)

    # get all syllable usage data
    df_only_stats = grouped_data.drop(["group", 'name'], axis=1)
    syllable_data = grouped_data.drop(["group", 'name'], axis=1).values

    N_m, N_s = syllable_data.shape

    # Run KW and return H-stats
    h_all, real_ranks, X_ties = run_manual_KW_test(
        df_usage=df_only_stats,
        merged_usages_all=syllable_data,
        num_groups=num_groups,
        n_per_group=n_per_group,
        cum_group_idx=cum_group_idx,
        n_perm=n_perm,
        seed=seed,
    )

    # find the real k_real
    df_k_real = pd.DataFrame(
        [
            stats.kruskal(
                *np.array_split(syllable_data[:, s_i], np.cumsum(n_per_group[:-1]))
            )
            for s_i in range(N_s)
        ]
    )

    # multiple test correction
    df_k_real["p_adj"] = multipletests(
        ((h_all > df_k_real.statistic.values).sum(0) + 1) / n_perm,
        alpha=thresh,
        method=mc_method,
    )[1]

    # return significant syllables based on the threshold
    df_k_real["is_sig"] = df_k_real["p_adj"] <= thresh

    # Run Dunn's z-test statistics
    (
        null_zs_within_group,
        real_zs_within_group,
    ) = dunns_z_test_permute_within_group_pairs(
        grouped_data, vc, real_ranks, X_ties, N_m, group_names, rnd, n_perm
    )

    # Compute p-values from Dunn's z-score statistics
    df_pair_corrected_pvalues, _ = compute_pvalues_for_group_pairs(
        real_zs_within_group,
        null_zs_within_group,
        df_k_real,
        group_names,
        n_perm,
        thresh,
        mc_method,
    )

    # combine Dunn's test results into single DataFrame
    df_z = pd.DataFrame(real_zs_within_group)
    df_z.index = df_z.index.set_names("syllable")
    dunn_results_df = df_z.reset_index().melt(id_vars="syllable")

    # Get intersecting significant syllables between
    intersect_sig_syllables = {}
    for pair in df_pair_corrected_pvalues.columns.tolist():
        intersect_sig_syllables[pair] = np.where(
            (df_pair_corrected_pvalues[pair] < thresh) & (df_k_real.is_sig)
        )[0]

    return df_k_real, dunn_results_df, intersect_sig_syllables


# frequency plot stuff
def sort_syllables_by_stat_difference(stats_df, ctrl_group, exp_group, stat='frequency'):
    """sort syllables by the difference in the stat between the control and experimental group

    Parameters
    ----------
    stats_df : pandas.DataFrame
        the complete dataframe that contains kinematic data for each frame
    ctrl_group : str
        the name of the control group
    exp_group : str
        the name of the experimental group
    stat : str, optional
        the statistic to use for finding the syllable differences between two groups, by default 'frequency'

    Returns
    -------
    list
        ordering list of syllables based on the difference in the stat between the control and experimental group
    """

    # Prepare DataFrame
    mutation_df = stats_df.drop([col for col, dtype in stats_df.dtypes.items() if (
        dtype == 'object' and col not in ['group', 'syllable'])], axis=1).groupby(['group', 'syllable']).mean()

    # Get groups to measure mutation by
    control_df = mutation_df.loc[ctrl_group]
    exp_df = mutation_df.loc[exp_group]

    # compute mean difference at each syll frequency and reorder based on difference
    ordering = (exp_df[stat] - control_df[stat]
                ).sort_values(ascending=False).index

    return list(ordering)


def sort_syllables_by_stat(stats_df, stat='frequency'):
    """sort sylllabes by the stat and return the ordering and label mapping

    Parameters
    ----------
    stats_df : pandas.DataFrame
        the stats dataframe that contains kinematic data and the syllable label for each session and each syllable
    stat : str, optional
        the statistic to sort on, by default 'frequency'

    Returns
    -------
    ordering : list
        the list of syllables sorted by the stat
    relabel_mapping : dict
        the mapping from the syllable to the new plotting label
    """

    tmp = stats_df.drop([col for col, dtype in stats_df.dtypes.items() if dtype == 'object'], axis=1).groupby('syllable').mean(
    ).sort_values(by=stat, ascending=False).index

    # Get sorted ordering
    ordering = list(tmp)

    # Get order mapping
    relabel_mapping = {o: i for i, o in enumerate(ordering)}

    return ordering, relabel_mapping


def _validate_and_order_syll_stats_params(complete_df, stat='frequency', order='stat', groups=None, ctrl_group=None, exp_group=None, colors=None, figsize=(10, 5)):

    if not isinstance(figsize, (tuple, list)):
        print('Invalid figsize. Input a integer-tuple or list of len(figsize) = 2. Setting figsize to (10, 5)')
        figsize = (10, 5)

    unique_groups = complete_df['group'].unique()

    if groups is None or len(groups) == 0:
        groups = unique_groups
    elif isinstance(groups, str):
        groups = [groups]

    if isinstance(groups, (list, tuple, np.ndarray)):
        diff = set(groups) - set(unique_groups)
        if len(diff) > 0:
            groups = unique_groups

    if stat.lower() not in complete_df.columns:
        raise ValueError(
            f'Invalid stat entered: {stat}. Must be a column in the supplied dataframe.')

    if order == "stat":
        ordering, _ = sort_syllables_by_stat(
            complete_df, stat=stat)
    elif order == "diff":
        if ctrl_group is None or exp_group is None or not np.all(np.isin([ctrl_group, exp_group], groups)):
            raise ValueError(
                f'Attempting to sort by {stat} differences, but {ctrl_group} or {exp_group} not in {groups}.')
        ordering = sort_syllables_by_stat_difference(
            complete_df, ctrl_group, exp_group, stat=stat)
    if colors is None:
        colors = []
    if len(colors) == 0 or len(colors) != len(groups):
        colors = sns.color_palette(n_colors=len(groups))

    return ordering, groups, colors, figsize


def plot_syll_stats_with_sem(stats_df, project_dir, model_dirname, save_dir=None, plot_sig=True, thresh=0.05, 
                             stat='frequency', order='stat', groups=None, ctrl_group=None, exp_group=None, 
                             colors=None, join=False, figsize=(8, 4)):
    """plot syllable statistics with standard error of the mean

    Parameters
    ----------
    stats_df : pandas.DataFrame
        the dataframe that contains kinematic data and the syllable label
    project_dir : str
        the project directory
    model_dirname : str
        the model directory name
    save_dir : str
        the path to save the analysis plots
    plot_sig : bool, optional
        whether to plot the significant syllables, by default True
    thresh : float, optional
        the threshold for significance, by default 0.05
    stat : str, optional
        the statistic to plot, by default 'frequency'
    ordering : str, optional
        the ordering of the syllables, by default 'stat'
    groups : list, optional
        the list of groups to plot, by default None
    ctrl_group : str, optional
        the control group, by default None
    exp_group : str, optional
        the experimental group, by default None
    colors : list, optional
        the list of colors to use for each group, by default None
    join : bool, optional
        whether to join the points with a line, by default False
    figsize : tuple, optional
        the figure size, by default (8, 4)

    Returns
    -------
    fig : matplotlib.figure.Figure
        the figure object
    legend : matplotlib.legend.Legend
        the legend object
    """

    # get syllable info
    syll_info = None
    syll_info_path = os.path.join(project_dir, model_dirname, "syll_info.yaml")
    if syll_info_path is not None:
        if os.path.exists(syll_info_path):
            with open(syll_info_path, 'r') as f:
                syll_info = yaml.safe_load(f)

    # get significant syllables
    sig_sylls = None

    if plot_sig and len(stats_df['group'].unique()) > 1:
        # run kruskal wallis and dunn's test
        _, _, sig_pairs = run_kruskal(stats_df, statistic=stat, thresh=thresh)
        # plot significant syllables for control and experimental group
        if ctrl_group is not None and exp_group is not None:
            # check if the group pair is in the sig pairs dict
            if (ctrl_group, exp_group) in sig_pairs.keys():
                sig_sylls = sig_pairs.get((ctrl_group, exp_group))
            # flip the order of the groups
            else:
                sig_sylls = sig_pairs.get((exp_group, ctrl_group))
        else:
            print(
                'No control or experimental group specified. Not plotting significant syllables.')

    xlabel = f'Syllables sorted by {stat}'
    if order == 'diff':
        xlabel += ' difference'
    ordering, groups, colors, figsize = _validate_and_order_syll_stats_params(stats_df,
                                                                              stat=stat,
                                                                              order=order,
                                                                              groups=groups,
                                                                              ctrl_group=ctrl_group,
                                                                              exp_group=exp_group,
                                                                              colors=colors,
                                                                              figsize=figsize)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot each group's stat data separately, computes groupwise SEM, and orders data based on the stat/ordering parameters
    hue = 'group' if groups is not None else None
    ax = sns.pointplot(data=stats_df, x='syllable', y=stat, hue=hue, order=ordering,
                       join=join, dodge=True, errorbar=('ci', 68), ax=ax, hue_order=groups,
                       palette=colors)

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # add syllable labels if they exist
    if syll_info is not None:
        mean_xlabels = []
        for o in (ordering):
            mean_xlabels.append(f'{syll_info[o]["label"]} - {o}')

        plt.xticks(range(len(mean_xlabels)), mean_xlabels, rotation=90)

    # if a list of significant syllables is given, mark the syllables above the x-axis
    if sig_sylls is not None:
        markings = []
        for s in sig_sylls:
            markings.append(ordering.index(s))
        plt.scatter(markings, [-.005] * len(markings), color='r', marker='*')

        # manually define a new patch
        patch = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                              markersize=9, label='Significant Syllable')
        handles.append(patch)

    # add legend and axis labels
    legend = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1, 1))
    plt.xlabel(xlabel, fontsize=12)
    sns.despine()

    # save the figure
    # saving the figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir=os.path.join(project_dir, model_dirname, 'figures')
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{stat}_{order}_stats.pdf'))
    fig.savefig(os.path.join(save_dir, f'{stat}_{order}_stats.png'))
    return fig, legend


# transition matrix
def get_transitions(label_sequence):
    """get the syllable transitions and their locations

    Parameters
    ----------
    label_sequence : np.ndarray
        the sequence of syllable labels for a session

    Returns
    -------
    transitions : np.ndarray
        the sequence of syllable transitions
    locs : np.ndarray
        the locations of the syllable transitions
    """

    arr = deepcopy(label_sequence)

    # get syllable transition locations
    locs = np.where(arr[1:] != arr[:-1])[0] + 1
    transitions = arr[locs]

    return transitions, locs


def n_gram_transition_matrix(labels, n=2, max_label=99):
    """the transition matrix for n-grams
    Parameters
    ----------
    labels : list or np.ndarray
        session state lists
    n : int, optional
        the number of successive states in the sequence, by default 2
    max_label : int, optional
        the maximum number of the syllable labels to include, by default 99

    Returns
    -------
    trans_mat : np.ndarray
        the transition matrices for the n-grams
    """
    trans_mat = np.zeros((max_label, ) * n, dtype='float')
    for loc in sliding_window(n, labels):
        if any(l >= max_label for l in loc):
            continue
        trans_mat[loc] += 1
    return trans_mat


def normalize_transition_matrix(init_matrix, normalize):
    """normalize the transition matrices

    Parameters
    ----------
    init_matrix : numpy.ndarray
        the initial transition matrix to be normalized 
    normalize : str
        the method to normalize the transition matrix

    Returns
    -------
    init_matrix : numpy.ndarray
        the trnasition matrix normalized by the method specified
    """
    if normalize is None or normalize not in ('bigram', 'rows', 'columns'):
        return init_matrix
    if normalize == 'bigram':
        init_matrix /= init_matrix.sum()
    elif normalize == 'rows':
        init_matrix /= init_matrix.sum(axis=1, keepdims=True)
    elif normalize == 'columns':
        init_matrix /= init_matrix.sum(axis=0, keepdims=True)

    return init_matrix


def get_transition_matrix(labels, max_syllable=100, normalize='bigram',
                          smoothing=0.0, combine=False, disable_output=False) -> list:
    """compute the transition matrix for the syllable labels

    Parameters
    ----------
    labels : list or np.ndarray
        syllable labels per session
    max_syllable : int, optional
        the maximum number of syllables to include, by default 100
    normalize : str, optional
        the method to normalize the transition matrix, by default 'bigram'
    smoothing : float, optional
        the smoothing value (pseudo count) to add to the transition matrix, by default 0.0
    combine : bool, optional
        whether to combine the transition matrices for all the sessions, by default False
    disable_output : bool, optional
        whether to disable the progress bar, by default False

    Returns
    -------
    all_mats : list
        the list of transition matrices for each session
    """
    if not isinstance(labels[0], (list, np.ndarray, pd.Series)):
        labels = [labels]

    # Compute a singular transition matrix
    if combine:
        init_matrix = []

        for v in labels:
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = n_gram_transition_matrix(
                transitions, n=2, max_label=max_syllable)
            init_matrix.append(trans_mat)

        init_matrix = np.sum(init_matrix, axis=0) + smoothing
        all_mats = normalize_transition_matrix(init_matrix, normalize)
    else:
        # Compute a transition matrix for each session label list
        all_mats = []
        for v in labels:
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = n_gram_transition_matrix(
                transitions, n=2, max_label=max_syllable) + smoothing

            # Normalize matrix
            init_matrix = normalize_transition_matrix(trans_mat, normalize)
            all_mats.append(init_matrix)

    return all_mats



def get_group_trans_mats(labels, label_group, group, syll_include, normalize='bigram'):
    """get the transition matrices for each group

    Parameters
    ----------
    labels : list or np.ndarray
        session state lists
    label_group : list or np.ndarray
        the group labels for each session
    group : list or np.ndarray
        the groups in the project
    max_sylls : int
        the maximum number of syllables to include
    normalize : str, optional
        the method to normalize the transition matrix, by default 'bigram'

    Returns
    -------
    trans_mats : list
        the list of transition matrices for each group
    frequencies : list
        the list of syllable frequencies for each group
    """
    trans_mats = []
    frequencies = []

    # Computing transition matrices for each given group
    for plt_group in group:
        # list of syll labels in sessions in the group
        use_labels = [lbl for lbl, grp in zip(
            labels, label_group) if grp == plt_group]
        # find stack np array shape
        row_num = len(use_labels)
        max_len = max([len(lbl) for lbl in use_labels])
        # Get sessions to include in trans_mat
        # subset only syllable included
        trans_mats.append(get_transition_matrix(use_labels,
                                                normalize=normalize,
                                                combine=True)[syll_include, :][:, syll_include])
        
        # Getting frequency information for node scaling
        group_frequencies = get_frequencies(use_labels)[syll_include]

        frequencies.append(group_frequencies)
    return trans_mats, frequencies


def visualize_transition_bigram(group, trans_mats, syll_include, save_dir=None, normalize='bigram'):
    """visualize the transition matrices for each group

    Parameters
    ----------
    group : list or np.ndarray
        the groups in the project
    trans_mats : list
        the list of transition matrices for each group
    normalize : str, optional
        the method to normalize the transition matrix, by default 'bigram'
    """

    # infer max_syllables
    max_syllables = trans_mats[0].shape[0]

    fig, ax = plt.subplots(1, len(group), figsize=(
        20, 20), sharex=False, sharey=True)
    title_map = dict(bigram='Bigram', columns='Incoming', rows='Outgoing')
    color_lim = max([x.max() for x in trans_mats])
    if len(group) == 1:
        axs = [ax]
    else:
        axs = ax.flat
    for i, g in enumerate(group):
        h = axs[i].imshow(trans_mats[i][:max_syllables,
                                        :max_syllables], cmap='cubehelix', vmax=color_lim)
        if i == 0:
            axs[i].set_ylabel('Incoming syllable')
            plt.yticks(np.arange(len(syll_include)), syll_include)
        cb = fig.colorbar(h, ax=axs[i], fraction=0.046, pad=0.04)
        cb.set_label(f'{title_map[normalize]} transition probability')
        axs[i].set_xlabel('Outgoing syllable')
        axs[i].set_title(g)
        axs[i].set_xticks(np.arange(len(syll_include)), syll_include)

    # saving the figures
    # saving the figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir=os.path.join(project_dir, model_dirname, 'figures')
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'transition_matrices.pdf'))
    fig.savefig(os.path.join(save_dir, 'transition_matrices.png'))


def generate_transition_matrices(project_dir, model_dirname, normalize='bigram', 
                                 min_frequency=0.005, syll_key='syllables_reindexed'):
    """generate the transition matrices for each session

    Parameters
    ----------
    progress_paths : dict
        the dictionary of paths to the files in the analysis progress
    normalize : str, optional
        the method to normalize the transition matrix, by default 'bigram'
    syll_key : str, optional
        the key to the syllable list in the progress file, by default 'syllables_reindexed'

    Returns
    -------
    trans_mats : list
        the list of transition matrices for each group

    """

    trans_mats, usages = None, None
    # index file
    index_file = os.path.join(project_dir, 'index.yaml')
    if not os.path.exists(index_file):
        generate_index(project_dir, model_dirname, index_file)

    with open(index_file, 'r') as f:
        index_data = yaml.safe_load(f)
    label_group = [session_info['group']
                   for session_info in index_data['files']]
    sessions = [session_info['name']
                for session_info in index_data['files']]
    group = sorted(list(set(label_group)))
    print('Group(s):', ', '.join(group))

    # load model reuslts
    results_dict = load_results(
        project_dir=project_dir, name=model_dirname)

    # filter out syllables by freqency
    model_labels = [results_dict[session][syll_key] for session in sessions]
    frequencies=get_frequencies(model_labels)
    syll_include=np.where(frequencies>min_frequency)[0]

    trans_mats, usages = get_group_trans_mats(
        model_labels, label_group, group, syll_include=syll_include, normalize=normalize)
    return trans_mats, usages, group, syll_include


def plot_transition_graph_group(groups, trans_mats, usages, syll_include, save_dir=None, layout='circular', node_scaling=2000):
    """plot the transition graph for each group

    Parameters
    ----------
    groups : list
        the list of groups to plot
    trans_mats : list
        the list of transition matrices for each group
    usages : list
        the list of syllable usage for each group
    layout : str, optional
        the layout of the graph, by default 'circular'
    node_scaling : int, optional
        the scaling factor for the node size, by default 2000
    """
    # Figure out the number of rows for the plot
    n_row = ceil(len(groups)/2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(16, 8*n_row))
    ax = all_axes.flat

    for i in range(len(groups)):
        print(groups[i])
        G = nx.from_numpy_array(trans_mats[i]*100)
        widths = nx.get_edge_attributes(G, 'weight')
        if layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        # get node list
        nodelist = G.nodes()
        # normalize the usage values
        sum_usages = sum(usages[i])
        normalized_usages = np.array(
            [u/sum_usages for u in usages[i]]) * node_scaling + 500
        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodelist,
                               node_size=normalized_usages,
                               node_color='white', edgecolors='red', ax=ax[i])
        nx.draw_networkx_edges(G, pos,
                               edgelist=widths.keys(),
                               width=list(widths.values()),
                               edge_color='black', ax=ax[i], alpha=0.6)
        nx.draw_networkx_labels(G, pos=pos,
                                labels=dict(zip(nodelist, syll_include)),
                                font_color='black', ax=ax[i])
        ax[i].set_title(groups[i])
    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis('off')
    # saving the figures
    # saving the figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir=os.path.join(project_dir, model_dirname, 'figures')
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'transition_graphs.pdf'))
    fig.savefig(os.path.join(save_dir, 'transition_graphs.png'))


def plot_transition_graph_difference(groups, trans_mats, usages, syll_include, save_dir=None, layout='circular', node_scaling=3000):
    """plot the difference of transition graph between groups

    Parameters
    ----------
    groups : list
        the list of groups to plot
    trans_mats : list
        the list of transition matrices for each group
    usages : list
        the list of syllable usage for each group
    layout : str, optional
        the layout of the graph, by default 'circular'
    node_scaling : int, optional
        the scaling factor for the node size, by default 3000
    """

    # find combinations
    group_combinations = list(combinations(groups, 2))

    # create group index dict
    group_idx_dict = {group: idx for idx, group in enumerate(groups)}

    # Figure out the number of rows for the plot
    n_row = ceil(len(group_combinations)/2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(16, 8*n_row))
    ax = all_axes.flat

    for i, pair in enumerate(group_combinations):
        left_ind = group_idx_dict[pair[0]]
        right_ind = group_idx_dict[pair[1]]
        # left tm minus right tm
        tm_diff = trans_mats[left_ind] - trans_mats[right_ind]
        # left usage minus right usage
        usages_diff = np.array(
            list(usages[left_ind])) - np.array(list(usages[right_ind]))
        normlized_usg_abs_diff = (
            np.abs(usages_diff)/np.abs(usages_diff).sum())*node_scaling+500

        G = nx.from_numpy_array(tm_diff * 1000)
        if layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)

        nodelist = G.nodes()
        widths = nx.get_edge_attributes(G, 'weight')

        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodelist,
                               node_size=normlized_usg_abs_diff,
                               node_color='white', edgecolors=['blue' if u > 0 else 'red' for u in usages_diff], ax=ax[i])
        nx.draw_networkx_edges(G, pos,
                               edgelist=widths.keys(),
                               width=np.abs(list(widths.values())),
                               edge_color=[
                                   'blue' if u > 0 else 'red' for u in widths.values()],
                               ax=ax[i], alpha=0.6)
        nx.draw_networkx_labels(G, pos=pos,
                                labels=dict(zip(nodelist, syll_include)),
                                font_color='black', ax=ax[i])
        ax[i].set_title(pair[0] + ' - ' + pair[1])

    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis('off')
    # add legend
    legend_elements = [Line2D([0], [0], color='r', lw=2, label=f'Up-regulated transistion'),
                       Line2D([0], [0], color='b', lw=2,
                              label=f'Down-regulated transistion'),
                       Line2D([0], [0], marker='o', color='w', label=f'Up-regulated usage',
                              markerfacecolor='w', markeredgecolor='r', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label=f'Down-regulated usage', markerfacecolor='w', markeredgecolor='b', markersize=10)]
    plt.legend(handles=legend_elements, loc='upper left', borderaxespad=0)
    # saving the figures
    # saving the figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir=os.path.join(project_dir, model_dirname, 'figures')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'transition_graphs_diff.pdf'))
        fig.savefig(os.path.join(save_dir, 'transition_graphs_diff.png'))


def changepoint_analysis(coordinates, *, anterior_bodyparts, posterior_bodyparts,
                         bodyparts=None, use_bodyparts=None, alpha=0.1,
                         derivative_ksize=3, gaussian_ksize=1, num_thresholds=20,
                         verbose=True, **kwargs):
    """
    Find changepoints in keypoint data. 

    Changepoints are peaks in a change score that is computed by:

        1. Differentiating (egocentrically aligned) keypoint coordinates
        2. Z-scoring the absolute values of each derivative
        3. Counting the number keypoint-coordinate pairs where the 
           Z-score crosses a threshold (in each frame).
        4. Computing a p-value for the number of threshold-crossings
           using a temporally shuffled null distribution
        5. Smoothing the resulting significance score across time

    Steps (3-5) are performed for a range of threshold values, and 
    the final outputs are based on the threshold that yields the 
    highest changepoint frequency.

    Parameters
    ----------
    coordinates : dict
        Keypoint observations as a dictionary mapping session names to
        ndarrays of shape (num_frames, num_keypoints, dim)

    anterior_bodyparts : iterable of str or int
        Anterior keypoints for egocentric alignment, either as indices
        or as strings if ``bodyparts`` is provided.

    posterior_bodyparts : iterable of str or int
        Posterior keypoints for egocentric alignment, either as indices
        or as strings if ``bodyparts`` is provided.

    bodyparts : iterable of str, optional
        Names of keypoints. Required for subsetting keypoints using
        ``use_bodyparts`` or if ``anterior_bodyparts`` and
        ``posterior_bodyparts`` are specified as strings.

    use_bodyparts : iterable of str, optional
        Subset of keypoints to use for changepoint analysis. If not
        provided, all keypoints are used.

    alpha : float, default=0.1
        False-discovery rate for statistical significance testing. Only
         changepoints with ``p < alpha`` are considered significant.

    derivative_ksize : int, default=3
        Size of the kernel used to differentiate keypoint coordinates. 
        For example if ``derivative_ksize=3``, the derivative would be

        .. math::

            \dot{y_t} = \frac{1}{3}( x_{t+3}+x_{t+2}+x_{t+1}-x_{t-1}-x_{t-2}-x_{t-3})

    gaussian_ksize : int, default=1
        Size of the kernel used to smooth the change score. 

    num_thresholds : int, default=20
        Number of thresholds to test.

    verbose : bool, default=True
        Print progress messages.

    Returns
    -------
    changepoints : dict
        Changepoints as a dictionary with the same keys as ``coordinates``.

    changescores : dict
        Change scores as a dictionary with the same keys as ``coordinates``.

    coordinates_ego: dict
        Keypoints in egocentric coordinates, in the same format as 
        ``coordinates``.

    derivatives : dict
        Z-scored absolute values of the derivatives for each egocentic 
        keypoint coordinate, in the same format as ``coordinates``

    threshold: float
        Threshold used to binarize Z-scored derivatives. 
    """
    if use_bodyparts is None and bodyparts is not None:
        use_bodyparts = bodyparts

    if isinstance(anterior_bodyparts[0], str):
        assert use_bodyparts is not None, fill(
            "Must provide `bodyparts` or `use_bodyparts` if `anterior_bodyparts` is a list of strings")
        anterior_idxs = [use_bodyparts.index(bp) for bp in anterior_bodyparts]
    else:
        anterior_idxs = anterior_bodyparts

    if isinstance(posterior_bodyparts[0], str):
        assert use_bodyparts is not None, fill(
            "Must provide `bodyparts` or `use_bodyparts` if `posterior_bodyparts` is a list of strings")
        posterior_idxs = [use_bodyparts.index(
            bp) for bp in posterior_bodyparts]
    else:
        posterior_idxs = posterior_bodyparts

    # Differentiating (egocentrically aligned) keypoint coordinates
    if verbose:
        print('Aligning keypoints')
    data, labels = format_data(
        coordinates, bodyparts=bodyparts, use_bodyparts=use_bodyparts)
    Y_ego, _, _ = align_egocentric(data['Y'], anterior_idxs, posterior_idxs)
    Y_flat = np.array(Y_ego).reshape(*Y_ego.shape[:2], -1)

    if verbose:
        print('Differentiating and z-scoring')
    dy = np.abs(filtered_derivative(Y_flat, derivative_ksize, axis=1))
    mask = np.broadcast_to(np.array(data['mask'])[:, :, na], dy.shape) > 0
    means = (dy * mask).sum(1) / mask.sum(1)
    dy_centered = dy - means[:, na, :]
    stds = np.sqrt((dy_centered**2 * mask).sum(1) / mask.sum(1))
    dy_zscored = dy_centered / (stds[:, na, :]+1e-8)

    # Count threshold crossings
    thresholds = np.linspace(
        np.percentile(dy_zscored, 1),
        np.percentile(dy_zscored, 99),
        num_thresholds)

    def get_changepoints(score, pvals, alpha):
        pts = argrelextrema(score, np.greater, order=1)[0]
        return pts[pvals[pts] < alpha]

    # get changescores for each threshold
    all_changescores, all_changepoints = [], []
    for threshold in tqdm(thresholds, disable=(not verbose), desc='Testing thresholds'):

        # permute within-session then combine across sessions
        crossings = (dy_zscored > threshold).sum(2)[mask[:, :, 0]]
        crossings_shuff = permute_cyclic(
            dy_zscored > threshold, mask, axis=1).sum(2)[mask[:, :, 0]]
        crossings_shuff = crossings_shuff + \
            np.random.uniform(-.1, .1, crossings_shuff.shape)

        # get significance score
        ps_combined = 1 - \
            (np.sort(crossings_shuff).searchsorted(crossings)-1)/len(crossings)
        ps_combined = fdrcorrection(ps_combined, alpha=alpha)[1]

        # separate back into sessions
        pvals = np.zeros(mask[:, :, 0].shape)
        pvals[mask[:, :, 0]] = ps_combined
        pvals = unbatch(pvals, labels)

        changescores = {
            k: gaussian_filter1d(-np.log10(ps), gaussian_ksize) for k, ps in pvals.items()}
        changepoints = {k: get_changepoints(
            changescores[k], ps, alpha) for k, ps in pvals.items()}
        all_changescores.append(changescores)
        all_changepoints.append(changepoints)

    # pick threshold with most changepoints
    num_changepoints = [sum(map(len, d.values())) for d in all_changepoints]
    changescores = all_changescores[np.argmax(num_changepoints)]
    changepoints = all_changepoints[np.argmax(num_changepoints)]
    threshold = thresholds[np.argmax(num_changepoints)]

    coordinates_ego = unbatch(np.array(Y_ego), labels)
    derivatives = unbatch(dy_zscored.reshape(Y_ego.shape), labels)
    return changepoints, changescores, coordinates_ego, derivatives, threshold


def generate_index(project_dir, model_dirname, index_filepath):
    """generate index file

    Parameters
    ----------
    project_dir : str
        path to project directory
    model_dirname : str
        model directory name
    index_filepath : str
        path to index file
    """
    # generate a new index file
    results_dict = load_results(project_dir=project_dir, name=model_dirname)
    files = []
    for session in results_dict.keys():
        file_dict = {'name': session, 'group': 'default'}
        files.append(file_dict)

    index_data = {'files': files}
    # write to file and progress_paths
    with open(index_filepath, 'w') as f:
        yaml.safe_dump(index_data, f, default_flow_style=False)


def get_behavioral_distance(project_dir, model_dirname, video_dir, keypoint_data_type, min_frequency=0.005, pre=5, post=15, min_duration=3, num_samples=40):
    """compute behavioral distance based on the trajectory correlation metric

    Parameters
    ----------
    project_dir : str
        path to project directory
    model_dirname : str
        model directory name
    video_dir : str
        path to video directory
    keypoint_data_type : str
        keypoint data type
    min_frequency : float, optional
        the minimum threshold for syllable usage to include in the dendrogram, by default 0.005
    pre : int, optional
        frames before syllable onset, by default 5
    post : int, optional
        frames after syllable onset, by default 15
    min_duration : int, optional
        min duration of syllable to include, by default 3
    num_samples : int, optional
        the number of samples, by default 40

    Returns
    -------
    Z : ndarray
        linkage matrix of behavioral distance
    syllable_keys : list
        list of syllable keys
       

    Raises
    ------
    Exception
        video directory not found
    NotImplementedError
        keypoint data type not supported
    """
    #find videos
    if video_dir is None:
        video_dir = load_config(project_dir).get('video_dir', None)
    if video_dir is None:
        raise Exception(
            'Unable to find video directory. Please specify video directory.')
    # load coordinate results
    if keypoint_data_type == 'deeplabcut':
        coordinates, _, _ = load_deeplabcut_results(video_dir)
    elif keypoint_data_type == 'sleap':
        coordinates, _, _ = load_sleap_results(video_dir)
    else:
        raise NotImplementedError('Input type not supported.')

    results = load_results(project_dir=project_dir, name=model_dirname)
    # TODO: add support for estimated coordinates
    syllables = {k: v['syllables_reindexed'] for k, v in results.items()}
    centroids = {k: v['centroid'] for k, v in results.items()}
    headings = {k: v['heading'] for k, v in results.items()}
    # get syllable instances that meet criteria
    syllable_instances = get_syllable_instances(
        syllables, pre=pre, post=post, min_duration=min_duration,
        min_frequency=min_frequency, min_instances=num_samples)

    if len(syllable_instances) == 0:
        warnings.warn(fill(
            'No syllables with sufficient instances to make a trajectory plot. '
            'This usually occurs when all frames have the same syllable label '
            '(use `plot_syllable_frequencies` to check if this is the case)'))
        return
    # sample instances 
    sampled_instances = sample_instances(syllable_instances, num_samples, 
                                        coordinates=coordinates, centroids=centroids, 
                                        headings=headings, mode='density', n_neighbors=num_samples)
    # get trajectories
    trajectories = {syllable: get_trajectories(
        instances, coordinates, pre=pre, post=post, 
        centroids=centroids, headings=headings
        ) for syllable,instances in sampled_instances.items()}
    # get mean trajefctories
    syllable_keys = sorted(trajectories.keys())
    # Xs shape is (n_syllables, n_frames, num_keypoints, xy)
    Xs = np.nanmean(np.array([trajectories[syllable] for syllable in syllable_keys]),axis=1)
    # reshape to (n_syllables, n_frames*num_keypoints*xy)
    Xs = Xs.reshape(Xs.shape[0],-1)

    # get distances and transform back to 1-d array
    Z = linkage(pdist(Xs, metric='correlation'), 'complete')
    return Z, list(syllable_keys)



def plot_dendrogram(project_dir, model_dirname, video_dir=None, keypoint_data_type='deeplabcut', min_frequency=0.005, save_dir=None, figsize=(10, 5)):
    """plot similarity dendrogram

    Parameters
    ----------
    project_dir : str
        path to project directory
    model_dirname : str
        model directory name
    video_dir : str, optional
        path to video dir, by default None
    keypoint_data_type : str, optional
        keypoint data type, by default 'deeplabcut'
    min_frequency : float, optional
        the minimum threshold for syllable usage to include in the dendrogram, by default 0.005
    save_dir : str, optional
        path to the directory to save the figure, by default None
    figsize : tuple, optional
        the size of the figure, by default (10, 5)
    """
    Z, syllable_keys = get_behavioral_distance(project_dir, model_dirname, video_dir, keypoint_data_type, min_frequency)
    sns.set_style('whitegrid', {'axes.grid' : False})
    fig, ax = plt.subplots(figsize=figsize)
    # get syllable info
    syll_info = None
    syll_info_path = os.path.join(project_dir, model_dirname, "syll_info.yaml")
    if syll_info_path is not None:
        if os.path.exists(syll_info_path):
            with open(syll_info_path, 'r') as f:
                syll_info = yaml.safe_load(f)
    syll_info = pd.DataFrame(syll_info).T.sort_index()
    syll_info = syll_info[syll_info.index.isin(syllable_keys)]
    labels = (syll_info['label']+"-" +syll_info.index.astype(str)).to_numpy()

    dendrogram(Z, distance_sort=False, no_plot=False, labels=labels, color_threshold=None, leaf_font_size=15, leaf_rotation=90, ax=ax)

    # saving the figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir=os.path.join(project_dir, model_dirname, 'figures')
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(join(save_dir, 'syllable_dendrogram.pdf'))
    fig.savefig(join(save_dir, 'syllable_dendrogram.png'))