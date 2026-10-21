import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from keypoint_moseq.analysis import run_kruskal
from analysis_utils import check_figure_files, syllable_names

# Arguments of the notebook's group comparison cell
STAT, THRESH, CTRL_GROUP, EXP_GROUP = "frequency", 0.05, "a", "b"
STAR_HEIGHT = -0.05  # y position of the significance stars
TOLERANCE = dict(rtol=1e-10, atol=0)


def marked_syllables(stats_df):
    """Significant syllables that `run_kruskal` reports for the notebook's
    control and experimental groups. They are positions in the sorted
    syllable order, which `plot_syll_stats_with_sem` looks up as syllable
    indices; the two agree when the syllables are numbered from 0 without
    gaps."""
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
    """Row i of the Kruskal-Wallis results is the test of the i-th syllable
    in sorted order, comparing the groups' per-recording values, with zero
    for recordings that have no row for the syllable."""
    np.random.seed(0)
    kruskal, _, _ = run_kruskal(tutorial_stats_df, statistic=STAT, thresh=THRESH)

    for i, syllable in enumerate(sorted(tutorial_stats_df["syllable"].unique())):
        expected = stats.kruskal(
            *(
                frequency_values(tutorial_stats_df, group, syllable)
                for group in (CTRL_GROUP, EXP_GROUP)
            )
        )
        np.testing.assert_allclose(
            kruskal.loc[i, ["statistic", "pvalue"]].to_numpy(dtype=float),
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
