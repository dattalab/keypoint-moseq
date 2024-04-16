from keypoint_moseq.util import filter_angle
from keypoint_moseq.io import load_results
from math import ceil
from matplotlib.lines import Line2D
from cytoolz import sliding_window
import networkx as nx
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from copy import deepcopy
from glob import glob
import panel as pn
from jax_moseq.utils import get_durations, get_frequencies

pn.extension("plotly", "tabulator")
na = np.newaxis


def get_syllable_names(project_dir, model_name, syllable_ixs):
    """Get syllable names from syll_info.csv file. Labels consist of the
    syllable index, followed by the syllable label, if it exists.

    Parameters
    ----------
    project_dir : str
        the path to the project directory
    model_name : str
        the name of the model directory
    syllable_ixs : list
        list of syllable indices to get names for

    Returns
    -------
    names: list of str
        list of syllable names
    """
    labels = {ix: f"{ix}" for ix in syllable_ixs}
    syll_info_path = os.path.join(project_dir, model_name, "syll_info.csv")
    if os.path.exists(syll_info_path):
        syll_info_df = pd.read_csv(syll_info_path, index_col=False).fillna("")

        for ix in syllable_ixs:
            if len(syll_info_df[syll_info_df.syllable == ix].label.values[0]) > 0:
                labels[ix] = (
                    f"{ix} ({syll_info_df[syll_info_df.syllable == ix].label.values[0]})"
                )
    names = [labels[ix] for ix in syllable_ixs]
    return names


def generate_index(project_dir, model_name, index_filepath):
    """Generate index file as a csv.

    Parameters
    ----------
    project_dir : str
        path to project directory
    model_name : str
        model directory name
    index_filepath : str
        path to index file
    """

    # load model results
    results_dict = load_results(project_dir, model_name)

    # check if file exists
    if os.path.exists(index_filepath):
        index_df = pd.read_csv(index_filepath, index_col=False)

        # all files are created
        if len(index_df) == len(results_dict.keys()):
            return
        else:
            # subset all the files
            files = index_df.name.values
            # find the missing file in results dict that's not in index
            for i in results_dict.keys():
                if i not in files:
                    index_df.loc[len(index_df.index)] = [i, "default"]
            # write index dataframe
            index_df.to_csv(index_filepath, index=False)
    else:
        # generate a new index file
        results_dict = load_results(project_dir, model_name)
        index_df = pd.DataFrame({"name": list(results_dict.keys()), "group": "default"})
        # write index dataframe
        index_df.to_csv(index_filepath, index=False)


def interactive_group_setting(project_dir, model_name):
    """Start the interactive group setting widget.

    Parameters
    ----------
    project_dir : str
        the path to the project directory
    model_name : str
        the name of the model directory
    """

    index_filepath = os.path.join(project_dir, "index.csv")

    if not os.path.exists(index_filepath):
        generate_index(project_dir, model_name, index_filepath)

    # making the interactive dataframe
    # open index dataframe

    # make a tabulator dataframe
    summary_data = pd.read_csv(index_filepath, index_col=False)

    titles = {"name": "recording name", "group": "group"}

    editors = {
        "name": None,
        "group": {
            "type": "textarea",
            "elementAttributes": {
                "maxlength": "100",
                "onkeydown": "if(event.keyCode == 13 && !event.shiftKey){this.blur();}",
            },
            "selectContents": True,
            "verticalNavigation": "editor",
            "shiftEnterSubmit": True,
        },
    }

    widths = {"name": 400}
    base_configuration = {"clipboard": "copy"}

    summary_table = pn.widgets.Tabulator(
        summary_data,
        editors=editors,
        layout="fit_data_table",
        selectable=1,
        show_index=False,
        titles=titles,
        widths=widths,
        configuration=base_configuration,
    )
    button = pn.widgets.Button(name="Save group info", button_type="primary")

    # call back function to save the index file
    def save_index(summary_data):
        # create index file from csv
        data_to_save = summary_data.copy()
        # remove newlines
        data_to_save["group"] = data_to_save["group"].str.strip()
        data_to_save.to_csv(index_filepath, index=False)

    # button click action
    def b(event, save=True):
        save_index(summary_data)

    button.on_click(b)

    return pn.Row(summary_table, pn.Column(button))


def compute_moseq_df(project_dir, model_name, *, fps=30, smooth_heading=True):
    """Compute moseq dataframe from results dict that contains all kinematic
    values by frame.

    Parameters
    ----------
    project_dir : str
        the path to the project directory
    model_name : str
        the name of the model directory
    results_dict : dict
        dictionary of results from model fitting
    use_bodyparts : bool
        boolean flag whether to include data for bodyparts
    smooth_heading : bool, optional
        boolean flag whether smooth the computed heading, by default True

    Returns
    -------
    moseq_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    """

    # load model results
    results_dict = load_results(project_dir, model_name)

    # load index file
    index_filepath = os.path.join(project_dir, "index.csv")
    if os.path.exists(index_filepath):
        index_data = pd.read_csv(index_filepath, index_col=False)
    else:
        print(
            "index.csv not found, if you want to include group information for each video, please run the Assign Groups widget first"
        )

    recording_name = []
    centroid = []
    velocity = []
    heading = []
    angular_velocity = []
    syllables = []
    frame_index = []
    s_group = []

    for k, v in results_dict.items():
        n_frame = v["centroid"].shape[0]
        recording_name.append([str(k)] * n_frame)
        centroid.append(v["centroid"])
        # velocity is pixel per second
        velocity.append(
            np.concatenate(
                (
                    [0],
                    np.sqrt(np.square(np.diff(v["centroid"], axis=0)).sum(axis=1))
                    * fps,
                )
            )
        )

        if index_data is not None:
            # find the group for each recording from index data
            s_group.append(
                [index_data[index_data["name"] == k]["group"].values[0]] * n_frame
            )
        else:
            # no index data
            s_group.append(["default"] * n_frame)
        frame_index.append(np.arange(n_frame))

        if smooth_heading:
            recording_heading = filter_angle(v["heading"])
        else:
            recording_heading = v["heading"]

        # heading in radian
        heading.append(recording_heading)

        # compute angular velocity (radian per second)
        gaussian_smoothed_heading = filter_angle(
            recording_heading, size=3, method="gaussian"
        )
        angular_velocity.append(
            np.concatenate(([0], np.diff(gaussian_smoothed_heading) * fps))
        )

        # add syllable data
        syllables.append(v["syllable"])

    # construct dataframe
    moseq_df = pd.DataFrame(np.concatenate(recording_name), columns=["name"])
    column_names = (
        ["centroid_x", "centroid_y"]
        if centroid[0].shape[1] == 2
        else ["centroid_x", "centroid_y", "centroid_z"]
    )
    moseq_df = pd.concat(
        [
            moseq_df,
            pd.DataFrame(np.concatenate(centroid), columns=column_names),
        ],
        axis=1,
    )
    moseq_df["heading"] = np.concatenate(heading)
    moseq_df["angular_velocity"] = np.concatenate(angular_velocity)
    moseq_df["velocity_px_s"] = np.concatenate(velocity)
    moseq_df["syllable"] = np.concatenate(syllables)
    moseq_df["frame_index"] = np.concatenate(frame_index)
    moseq_df["group"] = np.concatenate(s_group)

    # compute syllable onset
    change = np.diff(moseq_df["syllable"]) != 0
    indices = np.where(change)[0]
    indices += 1
    indices = np.concatenate(([0], indices))

    onset = np.full(moseq_df.shape[0], False)
    onset[indices] = True
    moseq_df["onset"] = onset
    return moseq_df


def compute_stats_df(
    project_dir,
    model_name,
    moseq_df,
    min_frequency=0.005,
    groupby=["group", "name"],
    fps=30,
):
    """Summary statistics for syllable frequencies and kinematic values.

    Parameters
    ----------
    moseq_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    threshold : float, optional
        usge threshold for the syllable to be included, by default 0.005
    groupby : list, optional
        the list of column names to group by, by default ['group', 'name']
    fps : int, optional
        frame per second information of the recording, by default 30

    Returns
    -------
    stats_df : pandas.DataFrame
        the summary statistics dataframe for syllable frequencies and kinematic values
    """
    # compute runlength encoding for syllables

    # load model results
    results_dict = load_results(project_dir, model_name)
    syllables = {k: res["syllable"] for k, res in results_dict.items()}
    # frequencies is array of frequencies for sorted syllables [syll_0, syll_1...]
    frequencies = get_frequencies(syllables)
    syll_include = np.where(frequencies > min_frequency)[0]

    # add group information
    # load index file
    index_filepath = os.path.join(project_dir, "index.csv")
    if os.path.exists(index_filepath):
        index_df = pd.read_csv(index_filepath, index_col=False)
    else:
        print(
            "index.csv not found, if you want to include group information for each video, please run the Assign Groups widget first"
        )

    # construct frequency dataframe
    # syllable frequencies within one session add up to 1
    frequency_df = []
    for k, v in results_dict.items():
        syll_freq = get_frequencies(v["syllable"])
        df = pd.DataFrame(
            {
                "name": k,
                "group": index_df[index_df["name"] == k]["group"].values[0],
                "syllable": np.arange(len(syll_freq)),
                "frequency": syll_freq,
            }
        )
        frequency_df.append(df)
    frequency_df = pd.concat(frequency_df)
    if "name" not in groupby:
        frequency_df.drop(columns=["name"], inplace=True)

    # filter out syllables that are used less than threshold in all recordings
    filtered_df = moseq_df[moseq_df["syllable"].isin(syll_include)].copy()

    # TODO: hard-coded heading for now, could add other scalars
    features = filtered_df.groupby(groupby + ["syllable"])[
        ["heading", "angular_velocity", "velocity_px_s"]
    ].agg(["mean", "std", "min", "max"])

    features.columns = ["_".join(col).strip() for col in features.columns.values]
    features.reset_index(inplace=True)

    # get durations
    trials = filtered_df["onset"].cumsum()
    trials.name = "trials"
    durations = filtered_df.groupby(groupby + ["syllable"] + [trials])["onset"].count()
    # average duration in seconds
    durations = durations.groupby(groupby + ["syllable"]).mean() / fps
    durations.name = "duration"
    # only keep the columns we need
    durations = durations.fillna(0).reset_index()[groupby + ["syllable", "duration"]]

    stats_df = pd.merge(features, frequency_df, on=groupby + ["syllable"])
    stats_df = pd.merge(stats_df, durations, on=groupby + ["syllable"])
    return stats_df


def generate_syll_info(project_dir, model_name, syll_info_path):
    # parse model results
    model_results = load_results(project_dir, model_name)
    unique_sylls = np.unique(
        np.concatenate([file["syllable"] for file in model_results.values()])
    )
    # construct the syllable dictionary
    # in the non interactive version there won't be any group info
    syll_info_df = pd.DataFrame(
        {
            "syllable": unique_sylls,
            "label": [""] * len(unique_sylls),
            "short_description": [""] * len(unique_sylls),
        }
    )

    grid_movies = glob(os.path.join(project_dir, model_name, "grid_movies", "*.mp4"))
    assert len(grid_movies) > 0, (
        "No grid movies found. Please run `generate_grid_movies` as described in the docs: "
        "https://keypoint-moseq.readthedocs.io/en/latest/modeling.html#visualization"
    )
    # make movie paths into a dataframe
    movie_df = pd.DataFrame(
        {
            "syllable": [
                int(os.path.splitext(os.path.basename(movie_path))[0][8:])
                for movie_path in grid_movies
            ],
            "movie_path": grid_movies,
        }
    )

    syll_info_df.merge(movie_df, on="syllable", how="outer").fillna("").to_csv(
        syll_info_path, index=False
    )


def label_syllables(project_dir, model_name, moseq_df):
    """Label syllables in the syllable grid movie.

    Parameters
    ----------
    project_dir : str
        the path to the project directory
    model_name : str
        the name of the model directory
    """

    # construct the syllable info path
    syll_info_path = os.path.join(project_dir, model_name, "syll_info.csv")

    # generate a new syll_info csv file
    if not os.path.exists(syll_info_path):
        # generate the syllable info csv file
        generate_syll_info(project_dir, model_name, syll_info_path)

    # ensure there is grid movies
    grid_movies = glob(os.path.join(project_dir, model_name, "grid_movies", "*.mp4"))
    assert len(grid_movies) > 0, (
        "No grid movies found. Please run `generate_grid_movies` as described in the docs: "
        "https://keypoint-moseq.readthedocs.io/en/latest/modeling.html#visualization"
    )

    # load syll_info
    syll_info_df = pd.read_csv(syll_info_path, index_col=False).fillna("")
    # split into with movie and without movie
    syll_info_df_with_movie = syll_info_df[
        syll_info_df.movie_path.str.contains(".mp4")
    ].copy()
    syll_info_df_without_movie = syll_info_df[
        ~syll_info_df.movie_path.str.contains(".mp4")
    ].copy()

    # create select widget only include the ones with a movie
    select = pn.widgets.Select(
        name="Select", options=sorted(list(syll_info_df_with_movie.syllable))
    )

    # call back function to create video displayer
    def show_movie(syllable):
        movie_path = syll_info_df_with_movie[
            syll_info_df_with_movie.syllable == select.value
        ].movie_path.values[0]
        return pn.pane.Video(movie_path, width=500, loop=False)

    # dynamic video displayer
    ivideo = pn.bind(show_movie, syllable=select)

    # create the labeler dataframe
    # only include the syllable that have grid movies
    include = syll_info_df_with_movie.syllable.values
    syll_df = moseq_df[["syllable"]].groupby("syllable").mean().reset_index().copy()
    syll_df = syll_df[syll_df.syllable.isin(include)]

    # get labels and description from syll info
    syll_df = syll_df.merge(
        syll_info_df_with_movie[["syllable", "label", "short_description"]]
    ).copy()

    # set up interactive table
    titles = {
        "syllable": "syllable",
        "label": "label",
        "short_description": "short description",
    }
    editors = {
        "name": None,
        "label": {
            "type": "textarea",
            "elementAttributes": {
                "maxlength": "100",
                "onkeydown": "if(event.keyCode == 13 && !event.shiftKey){this.blur();}",
            },
            "selectContents": True,
            "verticalNavigation": "editor",
            "shiftEnterSubmit": True,
        },
        "short description": {
            "type": "textarea",
            "elementAttributes": {"maxlength": "200"},
            "selectContents": True,
            "verticalNavigation": "editor",
            "shiftEnterSubmit": True,
        },
    }

    base_configuration = {"clipboard": "copy"}

    widths = {"syllable": 100}
    summary_table = pn.widgets.Tabulator(
        syll_df,
        titles=titles,
        editors=editors,
        layout="fit_data_table",
        selectable=1,
        show_index=False,
        widths=widths,
        configuration=base_configuration,
    )

    button = pn.widgets.Button(name="Save syllable info", button_type="primary")

    # call back function to save the index file
    def save_index(syll_df):
        # create index file from csv
        temp_df = syll_df.copy()
        temp_df["label"] = temp_df["label"].str.strip()
        temp_df["short_description"] = temp_df["short_description"].str.strip()
        temp_df = temp_df.merge(
            syll_info_df_with_movie[["syllable", "movie_path"]], on="syllable"
        ).copy()
        pd.concat([temp_df, syll_info_df_without_movie]).fillna("").to_csv(
            syll_info_path, index=False
        )

    # button click action
    def b(event, save=True):
        save_index(syll_df)

    button.on_click(b)

    # bind everything together
    return pn.Row(
        pn.Column(select, ivideo), pn.Column(summary_table, pn.Column(button))
    )


def get_tie_correction(x, N_m):
    """Assign tied rank values to the average of the ranks they would have
    received if they had not been tied for Kruskal-Wallis helper function.

    Parameters
    ----------
    x : pd.Series
        syllable usages for a single recording.
    N_m : int
        Number of total recordings.

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


def run_manual_KW_test(
    df_usage,
    merged_usages_all,
    num_groups,
    n_per_group,
    cum_group_idx,
    n_perm=10000,
    seed=42,
):
    """Run a manual Kruskal-Wallis test compare the results agree with the
    scipy.stats.kruskal function.

    Parameters
    ----------
    df_usage : pandas.DataFrame
        DataFrame with syllable usages. shape = (N_m, n_syllables)
    merged_usages_all : np.array
        numpy array format of the df_usage DataFrame.
    num_groups : int
        Number of unique groups
    n_per_group : list
        list of value counts for recordings per group. len == num_groups.
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
            perm_ranks[:, cum_group_idx[i] : cum_group_idx[i + 1]].sum(1) ** 2
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


def dunns_z_test_permute_within_group_pairs(
    df_usage, vc, real_ranks, X_ties, N_m, group_names, rnd, n_perm
):
    """Run Dunn's z-test statistic on combinations of all group pairs, handling
    pre- computed tied ranks.

    Parameters
    ----------
    df_usage : pandas.DataFrame
        DataFrame containing only pre-computed syllable stats.
    vc : pd.Series
        value counts of recordings in each group.
    real_ranks : np.array
        Array of syllable ranks.
    X_ties : np.array
        1-D list of tied ranks, where if value > 0, then rank is tied
    N_m : int
        Number of recordings.
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

    for i_n, j_n in combinations(group_names, 2):
        is_i = df_usage.group == i_n
        is_j = df_usage.group == j_n

        n_mice = is_i.sum() + is_j.sum()

        ranks_perm = real_ranks[(is_i | is_j)][rnd.rand(n_perm, n_mice).argsort(-1)]
        diff = np.abs(
            ranks_perm[:, : is_i.sum(), :].mean(1)
            - ranks_perm[:, is_i.sum() :, :].mean(1)
        )
        B = 1.0 / vc.loc[i_n] + 1.0 / vc.loc[j_n]

        # also do for real data
        group_ranks = real_ranks[(is_i | is_j)]
        real_diff = np.abs(
            group_ranks[: is_i.sum(), :].mean(0) - group_ranks[is_i.sum() :, :].mean(0)
        )

        # add to dict
        pair = (i_n, j_n)
        null_zs_within_group[pair] = diff / np.sqrt((A - X_ties) * B)
        real_zs_within_group[pair] = real_diff / np.sqrt((A - X_ties) * B)

    return null_zs_within_group, real_zs_within_group


def compute_pvalues_for_group_pairs(
    real_zs_within_group,
    null_zs,
    df_k_real,
    group_names,
    n_perm=10000,
    thresh=0.05,
    mc_method="fdr_bh",
):
    """Adjust the p-values from Dunn's z-test statistics and computes the
    resulting significant syllables with the adjusted p-values.

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

    def correct_p(x):
        return multipletests(x, alpha=thresh, method=mc_method)[1]

    df_pval_corrected = df_pval.apply(correct_p, axis=1, result_type="broadcast")

    return df_pval_corrected, ((df_pval_corrected[df_k_real.is_sig] < thresh).sum(0))


def run_kruskal(
    stats_df,
    statistic="frequency",
    n_perm=10000,
    seed=42,
    thresh=0.05,
    mc_method="fdr_bh",
):
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
    grouped_data = (
        stats_df.pivot_table(
            index=["group", "name"], columns="syllable", values=statistic
        )
        .replace(np.nan, 0)
        .reset_index()
    )
    # compute KW constants
    vc = grouped_data.group.value_counts().loc[grouped_data.group.unique()]
    n_per_group = vc.values
    group_names = vc.index

    cum_group_idx = np.insert(np.cumsum(n_per_group), 0, 0)
    num_groups = len(group_names)

    # get all syllable usage data
    df_only_stats = grouped_data.drop(["group", "name"], axis=1)
    syllable_data = grouped_data.drop(["group", "name"], axis=1).values

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
    dunn_results_df = df_z.reset_index().melt(id_vars=[("syllable", "")])
    dunn_results_df.rename(
        columns={"variable_0": "group1", "variable_1": "group2"}, inplace=True
    )

    # Get intersecting significant syllables between
    intersect_sig_syllables = {}
    for pair in df_pair_corrected_pvalues.columns.tolist():
        intersect_sig_syllables[pair] = np.where(
            (df_pair_corrected_pvalues[pair] < thresh) & (df_k_real.is_sig)
        )[0]

    return df_k_real, dunn_results_df, intersect_sig_syllables


# frequency plot stuff
def sort_syllables_by_stat_difference(
    stats_df, ctrl_group, exp_group, stat="frequency"
):
    """Sort syllables by the difference in the stat between the control and
    experimental group.

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
    mutation_df = (
        stats_df.drop(
            [
                col
                for col, dtype in stats_df.dtypes.items()
                if (dtype == "object" and col not in ["group", "syllable"])
            ],
            axis=1,
        )
        .groupby(["group", "syllable"])
        .mean()
    )

    # Get groups to measure mutation by
    control_df = mutation_df.loc[ctrl_group]
    exp_df = mutation_df.loc[exp_group]

    # compute mean difference at each syll frequency and reorder based on difference
    ordering = (exp_df[stat] - control_df[stat]).sort_values(ascending=False).index

    return list(ordering)


def sort_syllables_by_stat(stats_df, stat="frequency"):
    """Sort sylllabes by the stat and return the ordering and label mapping.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        the stats dataframe that contains kinematic data and the syllable label for each recording and each syllable
    stat : str, optional
        the statistic to sort on, by default 'frequency'

    Returns
    -------
    ordering : list
        the list of syllables sorted by the stat
    relabel_mapping : dict
        the mapping from the syllable to the new plotting label
    """

    # stats_df frequency normalized by session
    # mean frequency by syllable don't always refect the ordering from reindexing
    # use the syllable label as ordering instead
    if stat == "frequency":
        ordering = sorted(stats_df.syllable.unique())
    else:
        ordering = (
            stats_df.drop(
                [col for col, dtype in stats_df.dtypes.items() if dtype == "object"],
                axis=1,
            )
            .groupby("syllable")
            .mean()
            .sort_values(by=stat, ascending=False)
            .index
        )

    # Get sorted ordering
    ordering = list(ordering)

    # Get order mapping
    relabel_mapping = {o: i for i, o in enumerate(ordering)}

    return ordering, relabel_mapping


def _validate_and_order_syll_stats_params(
    complete_df,
    stat="frequency",
    order="stat",
    groups=None,
    ctrl_group=None,
    exp_group=None,
    colors=None,
    figsize=(10, 5),
):
    if not isinstance(figsize, (tuple, list)):
        print(
            "Invalid figsize. Input a integer-tuple or list of len(figsize) = 2. Setting figsize to (10, 5)"
        )
        figsize = (10, 5)

    unique_groups = complete_df["group"].unique()

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
            f"Invalid stat entered: {stat}. Must be a column in the supplied dataframe."
        )

    if order == "stat":
        ordering, _ = sort_syllables_by_stat(complete_df, stat=stat)
    elif order == "diff":
        if (
            ctrl_group is None
            or exp_group is None
            or not np.all(np.isin([ctrl_group, exp_group], groups))
        ):
            raise ValueError(
                f"Attempting to sort by {stat} differences, but {ctrl_group} or {exp_group} not in {groups}."
            )
        ordering = sort_syllables_by_stat_difference(
            complete_df, ctrl_group, exp_group, stat=stat
        )
    if colors is None:
        colors = []
    if len(colors) == 0 or len(colors) != len(groups):
        colors = sns.color_palette(n_colors=len(groups))

    return np.array(ordering), groups, colors, figsize


def save_analysis_figure(fig, plot_name, project_dir, model_name, save_dir):
    """Save an analysis figure.

    The figure is saved as both a .png and .pdf, either to `save_dir` if it is
    provided, or else to `project_dir/model_name/figures`.
    """
    if save_dir is None:
        save_dir = os.path.join(project_dir, model_name, "figures")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_name)
    fig.savefig(save_path + ".png", dpi=300)
    fig.savefig(save_path + ".pdf", dpi=300)
    print(f"Saved figure to {save_path}.png")


def plot_syll_stats_with_sem(
    stats_df,
    project_dir,
    model_name,
    save_dir=None,
    plot_sig=True,
    thresh=0.05,
    stat="frequency",
    order="stat",
    groups=None,
    ctrl_group=None,
    exp_group=None,
    colors=None,
    join=False,
    figsize=(8, 4),
):
    """Plot syllable statistics with standard error of the mean.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        the dataframe that contains kinematic data and the syllable label
    project_dir : str
        the project directory
    model_name : str
        the model directory name
    save_dir : str
        the path to save the analysis plots
    plot_sig : bool, optional
        whether to plot the significant syllables, by default True
    thresh : float, optional
        the threshold for significance, by default 0.05
    stat : str, optional
        the statistic to plot, by default 'frequency'
    order : str, optional
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

    # get significant syllables
    sig_sylls = None
    if groups is None:
        groups = stats_df["group"].unique()

    if plot_sig and len(stats_df["group"].unique()) > 1:
        # run kruskal wallis and dunn's test
        _, _, sig_pairs = run_kruskal(stats_df, statistic=stat, thresh=thresh)
        # plot significant syllables for control and experimental group when user specify something
        if ctrl_group in groups and exp_group in groups:
            # check if the group pair is in the sig pairs dict
            if (ctrl_group, exp_group) in sig_pairs.keys():
                sig_sylls = sig_pairs.get((ctrl_group, exp_group))
            # flip the order of the groups
            else:
                sig_sylls = sig_pairs.get((exp_group, ctrl_group))
        else:
            # plot everything if no group pair is specified
            sig_sylls = sig_pairs

    xlabel = f"Syllables sorted by {stat}"
    if order == "diff":
        xlabel += " difference"
    ordering, groups, colors, figsize = _validate_and_order_syll_stats_params(
        stats_df,
        stat=stat,
        order=order,
        groups=groups,
        ctrl_group=ctrl_group,
        exp_group=exp_group,
        colors=colors,
        figsize=figsize,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot each group's stat data separately, computes groupwise SEM, and orders data based on the stat/ordering parameters
    hue = "group" if groups is not None else None
    ax = sns.pointplot(
        data=stats_df,
        x="syllable",
        y=stat,
        hue=hue,
        order=ordering,
        errorbar=("ci", 68),
        ax=ax,
        hue_order=groups,
        palette=colors,
    )

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # add syllable labels if they exist
    syll_names = get_syllable_names(project_dir, model_name, ordering)
    plt.xticks(range(len(syll_names)), syll_names, rotation=90)

    # if a list of significant syllables is given, mark the syllables above the x-axis
    if sig_sylls is not None:
        init_y = -0.05
        # plot all sig syllables when no reasonable control and experimental group is specified
        if isinstance(sig_sylls, dict):
            for key in sig_sylls.keys():
                markings = []
                for s in sig_sylls[key]:
                    markings.append(np.where(ordering == s)[0])
                if len(markings) > 0:
                    markings = np.concatenate(markings)
                    plt.scatter(
                        markings, [init_y] * len(markings), color="r", marker="*"
                    )
                    plt.text(
                        plt.xlim()[1],
                        init_y,
                        f"{key[0]} vs. {key[1]} - Total {len(sig_sylls[key])} S.S.",
                    )
                    init_y += -0.05
                else:
                    print("No significant syllables found.")
        else:
            markings = []
            for s in sig_sylls:
                if s in ordering:
                    markings.append(np.where(ordering == s)[0])
                else:
                    continue
            if len(markings) > 0:
                markings = np.concatenate(markings)
                plt.scatter(markings, [-0.05] * len(markings), color="r", marker="*")
            else:
                print("No significant syllables found.")

        # manually define a new patch
        patch = Line2D(
            [],
            [],
            color="red",
            marker="*",
            markersize=9,
            label="Significant Syllable",
        )
        handles.append(patch)

    # add legend and axis labels
    legend = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1, 1))
    plt.xlabel(xlabel, fontsize=12)
    sns.despine()

    # save the figure
    plot_name = f"{stat}_{order}_stats"
    save_analysis_figure(fig, plot_name, project_dir, model_name, save_dir)
    return fig, legend


# transition matrix
def get_transitions(label_sequence):
    """Get the syllable transitions and their locations.

    Parameters
    ----------
    label_sequence : np.ndarray
        the sequence of syllable labels for a recording

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
    """The transition matrix for n-grams.

    Parameters
    ----------
    labels : list or np.ndarray
        recording state lists
    n : int, optional
        the number of successive states in the sequence, by default 2
    max_label : int, optional
        the maximum number of the syllable labels to include, by default 99

    Returns
    -------
    trans_mat : np.ndarray
        the transition matrices for the n-grams
    """
    trans_mat = np.zeros((max_label,) * n, dtype="float")
    for loc in sliding_window(n, labels):
        if any(l >= max_label for l in loc):
            continue
        trans_mat[loc] += 1
    return trans_mat


def normalize_transition_matrix(init_matrix, normalize):
    """Normalize the transition matrices.

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
    if normalize is None or normalize not in ("bigram", "rows", "columns"):
        return init_matrix
    if normalize == "bigram":
        init_matrix /= init_matrix.sum()
    elif normalize == "rows":
        init_matrix /= init_matrix.sum(axis=1, keepdims=True)
    elif normalize == "columns":
        init_matrix /= init_matrix.sum(axis=0, keepdims=True)

    return init_matrix


def get_transition_matrix(
    labels,
    max_syllable=100,
    normalize="bigram",
    smoothing=0.0,
    combine=False,
    disable_output=False,
) -> list:
    """Compute the transition matrix for the syllable labels.

    Parameters
    ----------
    labels : list or np.ndarray
        syllable labels per recording
    max_syllable : int, optional
        the maximum number of syllables to include, by default 100
    normalize : str, optional
        the method to normalize the transition matrix, by default 'bigram'
    smoothing : float, optional
        the smoothing value (pseudo count) to add to the transition matrix, by default 0.0
    combine : bool, optional
        whether to combine the transition matrices for all the recordings, by default False
    disable_output : bool, optional
        whether to disable the progress bar, by default False

    Returns
    -------
    all_mats : list
        the list of transition matrices for each recording
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
                transitions, n=2, max_label=max_syllable
            )
            init_matrix.append(trans_mat)

        init_matrix = np.sum(init_matrix, axis=0) + smoothing
        all_mats = normalize_transition_matrix(init_matrix, normalize)
    else:
        # Compute a transition matrix for each recording label list
        all_mats = []
        for v in labels:
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = (
                n_gram_transition_matrix(transitions, n=2, max_label=max_syllable)
                + smoothing
            )

            # Normalize matrix
            init_matrix = normalize_transition_matrix(trans_mat, normalize)
            all_mats.append(init_matrix)

    return all_mats


def get_group_trans_mats(labels, label_group, group, syll_include, normalize="bigram"):
    """Get the transition matrices for each group.

    Parameters
    ----------
    labels : list or np.ndarray
        recording state lists
    label_group : list or np.ndarray
        the group labels for each recording
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
        # list of syll labels in recordings in the group
        use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
        # find stack np array shape
        row_num = len(use_labels)
        max_len = max([len(lbl) for lbl in use_labels])
        # Get recordings to include in trans_mat
        # subset only syllable included
        trans_mats.append(
            get_transition_matrix(use_labels, normalize=normalize, combine=True)[
                syll_include, :
            ][:, syll_include]
        )

        # Getting frequency information for node scaling
        group_frequencies = get_frequencies(use_labels)[syll_include]

        frequencies.append(group_frequencies)
    return trans_mats, frequencies


def visualize_transition_bigram(
    project_dir,
    model_name,
    group,
    trans_mats,
    syll_include,
    save_dir=None,
    normalize="bigram",
    figsize=(12, 6),
    show_syllable_names=True,
):
    """Visualize the transition matrices for each group.

    Parameters
    ----------
    group : list or np.ndarray
        the groups in the project
    trans_mats : list
        the list of transition matrices for each group
    normalize : str, optional
        the method to normalize the transition matrix, by default 'bigram'
    figsize : tuple, optional
        the figure size, by default (12,6)
    show_syllable_names : bool, optional
        whether to show just syllable indexes (False) or syllable indexes and
        names (True)
    """

    # syllable info path
    syll_info_path = os.path.join(project_dir, model_name, "syll_info.csv")
    # initialize syllable names
    syll_names = [f"{ix}" for ix in syll_include]

    if show_syllable_names:
        if os.path.exists(syll_info_path):
            syll_names = get_syllable_names(project_dir, model_name, syll_include)

    # infer max_syllables
    max_syllables = trans_mats[0].shape[0]

    fig, ax = plt.subplots(1, len(group), figsize=figsize, sharex=False, sharey=True)
    title_map = dict(bigram="Bigram", columns="Incoming", rows="Outgoing")
    color_lim = max([x.max() for x in trans_mats])
    if len(group) == 1:
        axs = [ax]
    else:
        axs = ax.flat
    for i, g in enumerate(group):
        h = axs[i].imshow(
            trans_mats[i][:max_syllables, :max_syllables],
            cmap="cubehelix",
            vmax=color_lim,
        )
        if i == 0:
            axs[i].set_ylabel("Incoming syllable")
            plt.yticks(np.arange(len(syll_include)), syll_names)
        cb = fig.colorbar(h, ax=axs[i], fraction=0.046, pad=0.04)
        cb.set_label(f"{title_map[normalize]} transition probability")
        axs[i].set_xlabel("Outgoing syllable")
        axs[i].set_title(g)
        axs[i].set_xticks(np.arange(len(syll_include)), syll_names, rotation=90)

    # save the figure
    plot_name = "transition_matrices"
    save_analysis_figure(fig, plot_name, project_dir, model_name, save_dir)


def generate_transition_matrices(
    project_dir, model_name, normalize="bigram", min_frequency=0.005
):
    """Generate the transition matrices for each recording.

    Parameters
    ----------
    progress_paths : dict
        the dictionary of paths to the files in the analysis progress
    normalize : str, optional
        the method to normalize the transition matrix, by default 'bigram'

    Returns
    -------
    trans_mats : list
        the list of transition matrices for each group
    """
    trans_mats, usages = None, None
    # index file
    index_file = os.path.join(project_dir, "index.csv")
    if not os.path.exists(index_file):
        generate_index(project_dir, model_name, index_file)

    index_data = pd.read_csv(index_file, index_col=False)

    label_group = list(index_data.group.values)
    recordings = list(index_data.name.values)
    group = sorted(list(index_data.group.unique()))
    print("Group(s):", ", ".join(group))

    # load model reuslts
    results_dict = load_results(project_dir, model_name)

    # filter out syllables by freqency
    model_labels = [results_dict[recording]["syllable"] for recording in recordings]
    frequencies = get_frequencies(model_labels)
    syll_include = np.where(frequencies > min_frequency)[0]

    trans_mats, usages = get_group_trans_mats(
        model_labels,
        label_group,
        group,
        syll_include=syll_include,
        normalize=normalize,
    )
    return trans_mats, usages, group, syll_include


def plot_transition_graph_group(
    project_dir,
    model_name,
    groups,
    trans_mats,
    usages,
    syll_include,
    save_dir=None,
    layout="circular",
    node_scaling=2000,
    show_syllable_names=False,
):
    """Plot the transition graph for each group.

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
        the scaling factor for the node size, by default 2000,
    show_syllable_names : bool, optional
        whether to show just syllable indexes (False) or syllable indexes and
        names (True)
    """
    if show_syllable_names:
        syll_names = get_syllable_names(project_dir, model_name, syll_include)
    else:
        syll_names = [f"{ix}" for ix in syll_include]

    n_row = ceil(len(groups) / 2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(20, 10 * n_row))
    ax = all_axes.flat

    for i in range(len(groups)):
        G = nx.from_numpy_array(trans_mats[i] * 100)
        widths = nx.get_edge_attributes(G, "weight")
        if layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        # get node list
        nodelist = G.nodes()
        # normalize the usage values
        sum_usages = sum(usages[i])
        normalized_usages = (
            np.array([u / sum_usages for u in usages[i]]) * node_scaling + 1000
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=normalized_usages,
            node_color="white",
            edgecolors="red",
            ax=ax[i],
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=widths.keys(),
            width=list(widths.values()),
            edge_color="black",
            ax=ax[i],
            alpha=0.6,
        )
        nx.draw_networkx_labels(
            G,
            pos=pos,
            labels=dict(zip(nodelist, syll_names)),
            font_color="black",
            ax=ax[i],
        )
        ax[i].set_title(groups[i])
    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis("off")

    # save the figure
    plot_name = "transition_graphs"
    save_analysis_figure(fig, plot_name, project_dir, model_name, save_dir)


def plot_transition_graph_difference(
    project_dir,
    model_name,
    groups,
    trans_mats,
    usages,
    syll_include,
    save_dir=None,
    layout="circular",
    node_scaling=3000,
    show_syllable_names=False,
):
    """Plot the difference of transition graph between groups.

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
    show_syllable_names : bool, optional
        whether to show just syllable indexes (False) or syllable indexes and
        names (True)
    """
    if show_syllable_names:
        syll_names = get_syllable_names(project_dir, model_name, syll_include)
    else:
        syll_names = [f"{ix}" for ix in syll_include]

    # find combinations
    group_combinations = list(combinations(groups, 2))

    # create group index dict
    group_idx_dict = {group: idx for idx, group in enumerate(groups)}

    # Figure out the number of rows for the plot
    n_row = ceil(len(group_combinations) / 2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(16, 8 * n_row))
    ax = all_axes.flat

    for i, pair in enumerate(group_combinations):
        left_ind = group_idx_dict[pair[0]]
        right_ind = group_idx_dict[pair[1]]
        # left tm minus right tm
        tm_diff = trans_mats[left_ind] - trans_mats[right_ind]
        # left usage minus right usage
        usages_diff = np.array(list(usages[left_ind])) - np.array(
            list(usages[right_ind])
        )
        normlized_usg_abs_diff = (
            np.abs(usages_diff) / np.abs(usages_diff).sum()
        ) * node_scaling + 500

        G = nx.from_numpy_array(tm_diff * 1000)
        if layout == "circular":
            pos = nx.circular_layout(G)
        else:
            G_for_spring = nx.from_numpy_array(np.mean(trans_mats, axis=0))
            pos = nx.spring_layout(G_for_spring, iterations=500)

        nodelist = G.nodes()
        widths = nx.get_edge_attributes(G, "weight")

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=normlized_usg_abs_diff,
            node_color="white",
            edgecolors=["blue" if u > 0 else "red" for u in usages_diff],
            ax=ax[i],
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=widths.keys(),
            width=np.abs(list(widths.values())),
            edge_color=["blue" if u > 0 else "red" for u in widths.values()],
            ax=ax[i],
            alpha=0.6,
        )
        nx.draw_networkx_labels(
            G,
            pos=pos,
            labels=dict(zip(nodelist, syll_names)),
            font_color="black",
            ax=ax[i],
        )
        ax[i].set_title(pair[0] + " - " + pair[1])

    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis("off")
    # add legend
    legend_elements = [
        Line2D([0], [0], color="r", lw=2, label=f"Up-regulated transistion"),
        Line2D([0], [0], color="b", lw=2, label=f"Down-regulated transistion"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Up-regulated usage",
            markerfacecolor="w",
            markeredgecolor="r",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Down-regulated usage",
            markerfacecolor="w",
            markeredgecolor="b",
            markersize=10,
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper left", borderaxespad=0)

    # save the figure
    plot_name = "transition_graphs_diff"
    save_analysis_figure(fig, plot_name, project_dir, model_name, save_dir)
