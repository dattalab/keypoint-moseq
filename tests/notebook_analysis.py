# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: keypoint_moseq
#     language: python
#     name: keypoint_moseq
# ---

# %% [markdown]
# # Statistical Analysis
#
# [This notebook](https://github.com/dattalab/keypoint-moseq/blob/main/docs/source/analysis.ipynb) contains routines for analyzing the output of keypoint-MoSeq.
#
# ```{note}
# The interactive widgets require jupyterlab launched from the `keypoint_moseq` environment. They will not work properly in jupyter notebook.
# ```
#

# %% [markdown]
# ## Setup
#
# We assume you have already have keypoint-MoSeq outputs that are organized as follows.
# ```
# <project_dir>/               ** current working directory
# └── <model_name>/            ** model directory
#     ├── results.h5           ** model results
#     └── grid_movies/         ** [Optional] grid movies folder
# ```
# Use the code below to enter in your project directory and model name.

# %%
import keypoint_moseq as kpms

project_dir = "path/to/project"  # the full path to the project directory
model_name = (
    "model_name"  # name of model to analyze (e.g. something like `2023_05_23-15_19_03`)
)

# %% [markdown]
# ## Assign Groups
#
# The goal of this step is to assign group labels (such as "mutant" or "wildtype") to each recording. These labels are important later for performing group-wise comparisons.
# - The code below creates a table called `{project_dir}/index.csv` and launches a widget for editing the table. To use the widget:
#     - Click cells in the "group" column and enter new group labels.
#     - Hit `Save group info` when you're done.
# - **If the widget doesn't appear**, you also edit the table directly in Excel or LibreOffice Calc.

# %%
kpms.interactive_group_setting(project_dir, model_name)

# %% [markdown]
# ## Generate dataframes
#
# Generate a pandas dataframe called `moseq_df` that contains syllable labels and kinematic information for each frame across all the recording sessions.

# %%
moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)
moseq_df

# %% [markdown]
# Next generate a dataframe called `stats_df` that contains summary statistics for each syllable in each recording session, such as its usage frequency and its distribution of kinematic parameters.

# %%
stats_df = kpms.compute_stats_df(
    project_dir,
    model_name,
    moseq_df,
    min_frequency=0.005,  # threshold frequency for including a syllable in the dataframe
    groupby=["group", "name"],  # column(s) to group the dataframe by
    fps=30,
)  # frame rate of the video from which keypoints were inferred

stats_df

# %% [markdown]
# ### **Optional:** Save dataframes to csv
# Uncomment the code below to save the dataframes as .csv files

# %%
# import os

# # save moseq_df
# save_dir = os.path.join(project_dir, model_name) # directory to save the moseq_df dataframe
# moseq_df.to_csv(os.path.join(save_dir, 'moseq_df.csv'), index=False)
# print('Saved `moseq_df` dataframe to', save_dir)

# # save stats_df
# save_dir = os.path.join(project_dir, model_name)
# stats_df.to_csv(os.path.join(save_dir, 'stats_df'), index=False)
# print('Saved `stats_df` dataframe to', save_dir)

# %% [markdown]
# ##  Label syllables
#
# The goal of this step is name each syllable (e.g., "rear up" or "walk slowly").
# - The code below creates an empty table at `{project_dir}/{model_name}/syll_info.csv` and launches an interactive widget for editing the table. To use the widget:
#     - Select a syllable from the dropdown to display its grid movie.
#     - Enter a name into the `label` column of the table (and optionally a short description too).
#     - When you are done, hit `Save syllable info` at the bottom of the table.
# - **If the widget doesn't appear**, you can also edit the file directly in Excel or LibreOffice Calc.

# %%
kpms.label_syllables(project_dir, model_name, moseq_df)

# %% [markdown]
# ## Compare between groups
#
# Test for statistically significant differences between groups of recordings. The code below takes a syllable property (e.g. frequency or duration), plots its disribution for each syllable across for each group, and also tests whether the property differs significantly between groups. The results are summarized in a plot that is saved to `{project_dir}/{model_name}/analysis_figures`.
#
# There are two options for setting the order of syllables along the x-axis. When `order='stat'`, syllables are sorted by the mean value of the statistic. When `order='diff'`, syllables are sorted by the magnitude of difference between two groups that are determined by the `ctrl_group` and `exp_group` keywords. Note `ctrl_group` and `exp_group` are not related to significance testing.

# %%
kpms.plot_syll_stats_with_sem(
    stats_df,
    project_dir,
    model_name,
    plot_sig=True,  # whether to mark statistical significance with a star
    thresh=0.05,  # significance threshold
    stat="frequency",  # statistic to be plotted (e.g. 'duration' or 'velocity_px_s_mean')
    order="stat",  # order syllables by overall frequency ("stat") or degree of difference ("diff")
    ctrl_group="a",  # name of the control group for statistical testing
    exp_group="b",  # name of the experimental group for statistical testing
    figsize=(8, 4),  # figure size
    groups=stats_df["group"].unique(),  # groups to be plotted
)

# %% [markdown]
# ### Transition matrices
# Generate heatmaps showing the transition frequencies between syllables.

# %%
normalize = "bigram"  # normalization method ("bigram", "rows" or "columns")

trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
    project_dir,
    model_name,
    normalize=normalize,
    min_frequency=0.005,  # minimum syllable frequency to include
)

kpms.visualize_transition_bigram(
    project_dir,
    model_name,
    groups,
    trans_mats,
    syll_include,
    normalize=normalize,
    show_syllable_names=True,  # label syllables by index (False) or index and name (True)
)

# %% [markdown]
# ### Syllable Transition Graph
# Render transition rates in graph form, where nodes represent syllables and edges represent transitions between syllables, with edge width showing transition rate for each pair of syllables (secifically the max of the two transition rates in each direction).

# %%
# Generate a transition graph for each single group

kpms.plot_transition_graph_group(
    project_dir,
    model_name,
    groups,
    trans_mats,
    usages,
    syll_include,
    layout="circular",  # transition graph layout ("circular" or "spring")
    show_syllable_names=False,  # label syllables by index (False) or index and name (True)
)

# %%
# Generate a difference-graph for each pair of groups.

kpms.plot_transition_graph_difference(
    project_dir, model_name, groups, trans_mats, usages, syll_include, layout="circular"
)  # transition graph layout ("circular" or "spring")
