import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import keypoint_moseq as kpms
from keypoint_moseq.analysis import run_kruskal
from analysis_utils import check_figure_files, plot_syll_stats, syllable_names

# Arguments of the notebook's group comparison cell
STAT, THRESH, CTRL_GROUP, EXP_GROUP = "frequency", 0.05, "a", "b"
STAR_HEIGHT = -0.05  # y position of the significance stars
TOLERANCE = dict(rtol=1e-10, atol=0)


def marked_syllables(stats_df):
    """Significant syllables that `run_kruskal` reports for the notebook's
    control and experimental groups."""
    np.random.seed(0)
    _, _, sig_pairs = run_kruskal(stats_df, statistic=STAT, thresh=THRESH)
    if (CTRL_GROUP, EXP_GROUP) in sig_pairs:
        return list(sig_pairs[(CTRL_GROUP, EXP_GROUP)])
    return list(sig_pairs[(EXP_GROUP, CTRL_GROUP)])


def star_positions(stats_df):
    """x positions where `plot_syll_stats_with_sem` draws the significance
    stars: the position in the syllable order of each marked syllable that
    appears there."""
    ordering = sorted(stats_df["syllable"].unique())
    return [ordering.index(s) for s in marked_syllables(stats_df) if s in ordering]


def frequency_values(stats_df, group, syllable):
    """Frequency of `syllable` in each recording of `group`, zero where the
    syllable has no row."""
    rows = stats_df[stats_df["group"] == group]
    values = rows[rows["syllable"] == syllable].set_index("name")[STAT]
    return values.reindex(rows["name"].unique(), fill_value=0).to_numpy()


@pytest.fixture(scope="module")
def frozen_kruskal(frozen_stats_df):
    """`run_kruskal` outputs for the frozen `stats_df`. The global NumPy
    random state is seeded because its Kruskal-Wallis check draws from it."""
    np.random.seed(0)
    return run_kruskal(frozen_stats_df, statistic=STAT, thresh=THRESH)


def test_syll_stats_plot_files(tutorial_syll_stats_plot, tutorial_analysis_project):
    project_dir, model_name = tutorial_analysis_project
    save_path = os.path.join(project_dir, model_name, "figures", "frequency_stat_stats")

    check_figure_files(save_path, (2400, 1200))
    assert tutorial_syll_stats_plot["printed"].endswith(f"Saved figure to {save_path}.png\n")


def test_syll_stats_plot_syllable_names(
    tutorial_syll_stats_plot, tutorial_stats_df, tutorial_syllables_labeled
):
    syll_info = tutorial_syllables_labeled["saved_syll_info"]
    names = syllable_names(syll_info, sorted(tutorial_stats_df["syllable"].unique()))

    assert tutorial_syll_stats_plot["xlabel"] == "Syllables sorted by frequency"
    assert tutorial_syll_stats_plot["xticklabels"] == names
    assert any(name.endswith(")") for name in names)


def test_syll_stats_plot_group_means(tutorial_syll_stats_plot, tutorial_stats_df):
    ordering = sorted(tutorial_stats_df["syllable"].unique())
    groups = tutorial_stats_df["group"].unique()

    assert list(groups) == [CTRL_GROUP, EXP_GROUP]
    assert len(tutorial_syll_stats_plot["group_points"]) == len(groups)
    for group, points in zip(groups, tutorial_syll_stats_plot["group_points"]):
        rows = tutorial_stats_df[tutorial_stats_df["group"] == group]
        means = [rows.loc[rows["syllable"] == s, STAT].to_numpy().mean() for s in ordering]
        np.testing.assert_allclose(points, means, rtol=1e-12, err_msg=group)


@pytest.mark.parametrize("plot", ["tutorial", "frozen"])
def test_syll_stats_plot_significance_stars(
    plot, tutorial_syll_stats_plot, tutorial_stats_df, frozen_syll_stats_plot, frozen_stats_df
):
    summary, stats_df = {
        "tutorial": (tutorial_syll_stats_plot, tutorial_stats_df),
        "frozen": (frozen_syll_stats_plot, frozen_stats_df),
    }[plot]
    positions = star_positions(stats_df)

    np.testing.assert_array_equal(summary["stars"][:, 0], positions)
    assert (summary["stars"][:, 1] == STAR_HEIGHT).all()
    assert ("No significant syllables found." in summary["printed"]) == (len(positions) == 0)
    # The legend has an entry for significant syllables even when none are.
    assert summary["legend"] == [CTRL_GROUP, EXP_GROUP, "Significant Syllable"]


def test_frozen_syll_stats_plot_stars(frozen_syll_stats_plot, frozen_stats_df, data_regression):
    ordering = sorted(frozen_stats_df["syllable"].unique())
    positions = [int(x) for x in frozen_syll_stats_plot["stars"][:, 0]]

    assert len(positions) > 0
    data_regression.check(
        {"positions": positions, "syllables": [int(ordering[x]) for x in positions]}
    )


def test_run_kruskal_structure(tutorial_stats_df, data_regression):
    np.random.seed(0)
    kruskal, dunn, sig_pairs = run_kruskal(tutorial_stats_df, statistic=STAT, thresh=THRESH)
    num_syllables = tutorial_stats_df["syllable"].nunique()

    assert len(kruskal) == num_syllables
    assert len(dunn) == num_syllables * len(sig_pairs)
    assert (kruskal["is_sig"] == (kruskal["p_adj"] <= THRESH)).all()
    assert ((kruskal["p_adj"] > 0) & (kruskal["p_adj"] <= 1)).all()
    data_regression.check(
        {
            "kruskal_columns": list(kruskal.columns),
            "dunn_columns": [str(column) for column in dunn.columns],
            "group_pairs": [list(pair) for pair in sig_pairs],
        }
    )


def test_run_kruskal_statistics_per_syllable(tutorial_stats_df):
    """The Kruskal-Wallis results, indexed by syllable, test each syllable by
    comparing the groups' per-recording values, with zero
    for recordings that have no row for the syllable."""
    np.random.seed(0)
    kruskal, _, _ = run_kruskal(tutorial_stats_df, statistic=STAT, thresh=THRESH)

    for syllable in sorted(tutorial_stats_df["syllable"].unique()):
        expected = stats.kruskal(
            *(
                frequency_values(tutorial_stats_df, group, syllable)
                for group in (CTRL_GROUP, EXP_GROUP)
            )
        )
        np.testing.assert_allclose(
            kruskal.loc[syllable, ["statistic", "pvalue"]].to_numpy(dtype=float),
            [expected.statistic, expected.pvalue],
            rtol=1e-12,
            err_msg=f"syllable {syllable}",
        )


def test_run_kruskal_independent_of_global_random_state(tutorial_stats_df):
    outputs = []
    for seed in (1, 2):
        np.random.seed(seed)
        outputs.append(run_kruskal(tutorial_stats_df, statistic=STAT, thresh=THRESH))
    (kruskal_1, dunn_1, sig_1), (kruskal_2, dunn_2, sig_2) = outputs

    pd.testing.assert_frame_equal(kruskal_1, kruskal_2)
    pd.testing.assert_frame_equal(dunn_1, dunn_2)
    assert sig_1.keys() == sig_2.keys()
    for pair in sig_1:
        np.testing.assert_array_equal(sig_1[pair], sig_2[pair])


def test_run_kruskal_frozen_kruskal_wallis(frozen_kruskal, dataframe_regression):
    kruskal, _, _ = frozen_kruskal

    dataframe_regression.check(kruskal, default_tolerance=TOLERANCE)


def test_run_kruskal_frozen_dunn(frozen_kruskal, dataframe_regression):
    _, dunn, _ = frozen_kruskal

    dataframe_regression.check(
        pd.DataFrame(
            {
                "syllable": dunn[("syllable", "")],
                "group1": dunn["group1"],
                "group2": dunn["group2"],
                "value": dunn["value"],
            }
        ),
        default_tolerance=TOLERANCE,
    )


def test_run_kruskal_frozen_significant_syllables(frozen_kruskal, data_regression):
    _, _, sig_pairs = frozen_kruskal

    data_regression.check(
        {f"{a} vs {b}": [int(s) for s in syllables] for (a, b), syllables in sig_pairs.items()}
    )


def plot_with_significance(stats_df, project_dir):
    """The notebook's group comparison cell, with significance testing."""
    np.random.seed(0)
    kpms.plot_syll_stats_with_sem(
        stats_df,
        project_dir,
        "model",
        plot_sig=True,
        thresh=THRESH,
        stat=STAT,
        ctrl_group=CTRL_GROUP,
        exp_group=EXP_GROUP,
    )
    plt.close("all")


def test_stats_plot_warns_when_a_compared_group_has_few_recordings(frozen_stats_df, tmp_path):
    exp_names = sorted(frozen_stats_df.loc[frozen_stats_df["group"] == EXP_GROUP, "name"].unique())
    kept = frozen_stats_df[
        (frozen_stats_df["group"] == CTRL_GROUP) | frozen_stats_df["name"].isin(exp_names[:2])
    ]

    with pytest.warns(UserWarning, match="at least 3 recordings") as record:
        plot_with_significance(kept, str(tmp_path))

    message = "\n".join(str(w.message) for w in record)
    assert f"'{EXP_GROUP}'" in message
    for name in exp_names[:2]:
        assert name in message


def test_stats_plot_with_enough_recordings_per_group_does_not_warn_about_size(
    frozen_stats_df, tmp_path
):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        plot_with_significance(frozen_stats_df, str(tmp_path))

    assert not [w for w in record if "at least 3 recordings" in str(w.message)]


def test_stats_plot_warns_when_every_recording_is_in_default(frozen_stats_df, tmp_path):
    with pytest.warns(UserWarning, match="default") as record:
        plot_with_significance(frozen_stats_df.assign(group="default"), str(tmp_path))

    assert any("interactive_group_setting" in str(w.message) for w in record)


def renumbered(syllable):
    """Syllables 0, 1, 2, ... become 3, 5, 7, ...: the same order, but with
    gaps and not starting at 0."""
    return 2 * syllable + 3


def renumber(stats_df):
    stats_df = stats_df.copy()
    stats_df["syllable"] = renumbered(stats_df["syllable"])
    return stats_df


def test_run_kruskal_reports_syllable_labels(frozen_stats_df, frozen_kruskal):
    """Renumbering the syllables changes only the syllable labels in
    `run_kruskal`'s outputs."""
    kruskal, dunn, sig_pairs = frozen_kruskal
    np.random.seed(0)
    kruskal_new, dunn_new, sig_new = run_kruskal(
        renumber(frozen_stats_df), statistic=STAT, thresh=THRESH
    )

    syllables = sorted(frozen_stats_df["syllable"].unique())
    assert list(kruskal_new.index) == [renumbered(s) for s in syllables]
    pd.testing.assert_frame_equal(
        kruskal_new.reset_index(drop=True), kruskal.reset_index(drop=True)
    )
    assert list(dunn_new[("syllable", "")]) == [renumbered(s) for s in dunn[("syllable", "")]]
    columns = ["group1", "group2", "value"]
    pd.testing.assert_frame_equal(dunn_new[columns], dunn[columns])
    assert sig_new.keys() == sig_pairs.keys()
    assert any(len(syllables) for syllables in sig_pairs.values())
    for pair, syllables in sig_pairs.items():
        assert list(sig_new[pair]) == [renumbered(s) for s in syllables], pair


def test_significance_stars_follow_renumbered_syllables(
    frozen_syll_stats_plot, frozen_stats_df, tmp_path
):
    renumbered_plot = plot_syll_stats(
        renumber(frozen_stats_df), str(tmp_path / "demo_project"), "frozen_model"
    )

    assert len(frozen_syll_stats_plot["stars"]) > 0
    np.testing.assert_array_equal(renumbered_plot["stars"], frozen_syll_stats_plot["stars"])


def test_run_kruskal_with_a_gap_in_the_syllables():
    """Only syllable 5 differs between the groups, and syllables 2 to 4 are
    absent."""
    rng = np.random.default_rng(0)
    rows = [
        {
            "group": group,
            "name": f"{group}{i}",
            "syllable": syllable,
            STAT: 0.3 + rng.uniform(-0.02, 0.02) + (0.3 if group == "b" and syllable == 5 else 0),
        }
        for group in ("a", "b")
        for i in range(8)
        for syllable in (0, 1, 5)
    ]

    np.random.seed(0)
    kruskal, dunn, sig_pairs = run_kruskal(pd.DataFrame(rows), statistic=STAT, thresh=THRESH)

    assert list(kruskal.index) == [0, 1, 5]
    assert list(kruskal["is_sig"]) == [False, False, True]
    assert sorted(set(dunn[("syllable", "")])) == [0, 1, 5]
    assert list(sig_pairs[("a", "b")]) == [5]


def group_means(stats_df, group, stat):
    return stats_df[stats_df["group"] == group].groupby("syllable")[stat].mean()


def test_order_diff_sorts_by_exp_minus_ctrl(frozen_stats_df):
    ordering = kpms.analysis.sort_syllables_by_stat_difference(
        frozen_stats_df, CTRL_GROUP, EXP_GROUP, stat=STAT
    )

    difference = group_means(frozen_stats_df, EXP_GROUP, STAT) - group_means(
        frozen_stats_df, CTRL_GROUP, STAT
    )
    assert sorted(ordering) == sorted(frozen_stats_df["syllable"].unique())
    assert np.all(np.diff(difference.loc[ordering].to_numpy()) <= 0)


@pytest.mark.parametrize("stat", ["duration", "velocity_px_s_mean"])
def test_order_stat_sorts_other_stats_by_mean(frozen_stats_df, stat):
    ordering, relabel_mapping = kpms.analysis.sort_syllables_by_stat(frozen_stats_df, stat=stat)

    means = frozen_stats_df.groupby("syllable")[stat].mean()
    assert sorted(ordering) == sorted(means.index)
    assert np.all(np.diff(means.loc[ordering].to_numpy()) <= 0)
    assert relabel_mapping == {syllable: i for i, syllable in enumerate(ordering)}


def test_stats_plot_ordered_by_difference(frozen_stats_df, tmp_path):
    fig, _ = kpms.plot_syll_stats_with_sem(
        frozen_stats_df,
        str(tmp_path),
        "model",
        plot_sig=False,
        stat=STAT,
        order="diff",
        ctrl_group=CTRL_GROUP,
        exp_group=EXP_GROUP,
    )
    plt.close("all")

    assert fig is not None
