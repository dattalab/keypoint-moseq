import numpy as np
import pandas as pd
import pytest
from jax_moseq.utils import get_frequencies
from scipy import stats

import keypoint_moseq as kpms
from keypoint_moseq.util import filter_angle
from analysis_utils import FPS, cohort_group, mean_complete_run_lengths, synthetic_results

MIN_FREQUENCY = 0.005  # `min_frequency` of the notebook's stats_df cell
KINEMATICS = ["heading", "angular_velocity", "velocity_px_s"]
# pandas 2 stores text columns as `object` and pandas 3 as `str`, and pandas
# 3 requires Python 3.11 or later, so the dtype of a text column depends on
# the installed versions. Snapshots record either one under this name.
TEXT_DTYPE = "object or str"


def column_types(df):
    return {
        "columns": list(df.columns),
        "dtypes": {
            column: (
                TEXT_DTYPE
                if pd.api.types.infer_dtype(df[column], skipna=False) == "string"
                else str(dtype)
            )
            for column, dtype in df.dtypes.items()
        },
    }


def recording_rows(df, results):
    """Rows of `df` for each recording, in the order of `results`."""
    return {name: df[df["name"] == name] for name in results}


def test_moseq_df_structure(tutorial_moseq_df, data_regression):
    frames = tutorial_moseq_df.groupby("name", sort=False).size()
    groups = tutorial_moseq_df.groupby("name", sort=False)["group"].unique()

    data_regression.check(
        {
            **column_types(tutorial_moseq_df),
            "frames": {name: int(count) for name, count in frames.items()},
            "groups": {name: list(values) for name, values in groups.items()},
        }
    )


def test_moseq_df_matches_results(tutorial_moseq_df, tutorial_analysis_project):
    project_dir, model_name = tutorial_analysis_project
    results = kpms.load_results(project_dir, model_name)
    lengths = [len(recording["syllable"]) for recording in results.values()]
    concatenated = lambda key: np.concatenate([r[key] for r in results.values()])

    np.testing.assert_array_equal(tutorial_moseq_df["name"], np.repeat(list(results), lengths))
    np.testing.assert_array_equal(
        tutorial_moseq_df["frame_index"], np.concatenate([np.arange(n) for n in lengths])
    )
    np.testing.assert_array_equal(tutorial_moseq_df["syllable"], concatenated("syllable"))
    np.testing.assert_array_equal(
        tutorial_moseq_df[["centroid_x", "centroid_y"]], concatenated("centroid")
    )
    np.testing.assert_array_equal(
        tutorial_moseq_df["group"], np.repeat([cohort_group(n) for n in results], lengths)
    )


def test_moseq_df_kinematics(tutorial_moseq_df, tutorial_analysis_project):
    project_dir, model_name = tutorial_analysis_project
    results = kpms.load_results(project_dir, model_name)

    for name, rows in recording_rows(tutorial_moseq_df, results).items():
        centroid = results[name]["centroid"]
        heading = filter_angle(results[name]["heading"])
        smoothed_heading = filter_angle(heading, size=3, method="gaussian")
        speed = np.linalg.norm(np.diff(centroid, axis=0), axis=1) * FPS

        np.testing.assert_allclose(
            rows["velocity_px_s"], np.concatenate([[0], speed]), rtol=1e-12, err_msg=name
        )
        np.testing.assert_array_equal(rows["heading"], heading, err_msg=name)
        np.testing.assert_allclose(
            rows["angular_velocity"],
            np.concatenate([[0], np.angle(np.exp(1j * np.diff(smoothed_heading))) * FPS]),
            rtol=1e-12,
            err_msg=name,
        )


def test_angular_velocity_where_the_heading_wraps(tmp_path):
    """A heading turning at a constant 0.1 rad per frame wraps from π to -π
    about every 63 frames."""
    results = synthetic_results(["rec_a"])
    results["rec_a"]["heading"] = np.angle(np.exp(1j * 0.1 * np.arange(300)))
    moseq_df = kpms.compute_moseq_df(
        str(tmp_path), "model", results_dict=results, fps=FPS, smooth_heading=False
    )

    # leaves out the frames near each end, where smoothing pads the heading
    np.testing.assert_allclose(moseq_df["angular_velocity"][20:-20], 0.1 * FPS)


def test_moseq_df_onsets(tutorial_moseq_df, tutorial_analysis_project):
    project_dir, model_name = tutorial_analysis_project
    results = kpms.load_results(project_dir, model_name)

    for name, rows in recording_rows(tutorial_moseq_df, results).items():
        syllables = results[name]["syllable"]
        np.testing.assert_array_equal(
            rows["onset"], np.concatenate([[False], syllables[1:] != syllables[:-1]]), err_msg=name
        )


def edge_runs_results(n_frames=18):
    """Two recordings whose first and last runs are the only runs of
    syllables 0 and 7. rec_b starts with the syllable rec_a ends with."""
    results = synthetic_results(["rec_a", "rec_b"], n_frames=n_frames)
    results["rec_a"]["syllable"] = np.repeat([0, 1, 2, 7], [5, 3, 4, 6])
    results["rec_b"]["syllable"] = np.repeat([7, 1, 2, 0], [6, 5, 2, 5])
    return results


def test_a_recordings_first_frame_is_not_an_onset(tmp_path):
    results = edge_runs_results()
    moseq_df = kpms.compute_moseq_df(str(tmp_path), "model", results_dict=results, fps=FPS)

    for name, rows in recording_rows(moseq_df, results).items():
        np.testing.assert_array_equal(
            rows["frame_index"][rows["onset"]],
            np.flatnonzero(np.diff(results[name]["syllable"])) + 1,
            err_msg=name,
        )


@pytest.mark.parametrize(
    "groupby, durations",
    [
        (
            ["group", "name"],
            {
                ("rec_a", 0): np.nan,
                ("rec_a", 1): 3,
                ("rec_a", 2): 4,
                ("rec_a", 7): np.nan,
                ("rec_b", 0): np.nan,
                ("rec_b", 1): 5,
                ("rec_b", 2): 2,
                ("rec_b", 7): np.nan,
            },
        ),
        (["group"], {0: np.nan, 1: 4, 2: 3, 7: np.nan}),
    ],
)
def test_stats_df_durations_leave_out_each_recordings_first_and_last_runs(
    tmp_path, groupby, durations
):
    """The first and last runs of a recording are cut short by its edges, so
    syllables 0 and 7 have no duration. Their frequencies still count them."""
    results = edge_runs_results()
    moseq_df = kpms.compute_moseq_df(str(tmp_path), "model", results_dict=results, fps=FPS)

    stats_df = kpms.compute_stats_df(moseq_df, min_frequency=0.0, groupby=groupby)

    # stats_df keeps a row per recording even when grouped by group only
    keys = ["name", "syllable"] if "name" in groupby else ["syllable"]
    duration = stats_df.drop_duplicates(keys).set_index(keys)["duration"]
    assert sorted(duration.index) == sorted(durations)
    np.testing.assert_allclose(
        duration.loc[list(durations)],
        np.array(list(durations.values()), dtype=float) / FPS,
        rtol=1e-12,
    )
    if "name" in groupby:
        frequency = stats_df.set_index(keys)["frequency"]
        for name, syllable in durations:
            assert frequency.loc[(name, syllable)] == pytest.approx(
                get_frequencies(results[name]["syllable"])[syllable]
            )


def test_moseq_df_without_index_uses_default_group(analysis_project_without_groups):
    project_dir, model_name = analysis_project_without_groups

    moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)

    assert (moseq_df["group"] == "default").all()


def test_filter_angle_values(ndarrays_regression):
    """`compute_moseq_df` smooths headings with `filter_angle`. The tutorial
    headings come from the fitted model, so the filter is snapshotted on a
    fixed series instead: a random walk from the legacy `RandomState`, whose
    stream NumPy keeps fixed."""
    steps = np.random.RandomState(0).normal(scale=0.3, size=1000)
    angles = (np.cumsum(steps) + np.pi) % (2 * np.pi) - np.pi
    median = filter_angle(angles)

    ndarrays_regression.check(
        {
            "median": median,
            "median_then_gaussian": filter_angle(median, size=3, method="gaussian"),
        },
        default_tolerance=dict(rtol=1e-10, atol=1e-12),
    )


def test_stats_df_structure(tutorial_stats_df, data_regression):
    groups = tutorial_stats_df.groupby("name", sort=False)["group"].unique()

    data_regression.check(
        {
            **column_types(tutorial_stats_df),
            "groups": {name: list(values) for name, values in groups.items()},
        }
    )


def test_stats_df_rows(tutorial_stats_df, tutorial_analysis_project):
    project_dir, model_name = tutorial_analysis_project
    results = kpms.load_results(project_dir, model_name)
    frequencies = get_frequencies({name: r["syllable"] for name, r in results.items()})
    included = np.flatnonzero(frequencies > MIN_FREQUENCY)

    rows = [
        (group, name, int(syllable))
        for group, name, syllable in zip(
            tutorial_stats_df["group"], tutorial_stats_df["name"], tutorial_stats_df["syllable"]
        )
    ]
    # a syllable a recording never used still gets a row, with frequency 0
    expected = {
        (cohort_group(name), name, int(syllable))
        for name in results
        for syllable in included
    }

    assert len(rows) == len(set(rows))
    assert set(rows) == expected


def test_stats_df_frequency_and_duration(tutorial_stats_df, tutorial_analysis_project):
    project_dir, model_name = tutorial_analysis_project
    results = kpms.load_results(project_dir, model_name)

    num_states = max(r["syllable"].max() for r in results.values()) + 1
    for name, rows in recording_rows(tutorial_stats_df, results).items():
        syllables = rows["syllable"].to_numpy()
        mean_run_lengths = mean_complete_run_lengths(results[name]["syllable"], syllables)

        # unused syllables have frequency 0 and a NaN duration
        np.testing.assert_array_equal(
            rows["frequency"],
            get_frequencies(results[name]["syllable"], num_states=num_states)[syllables],
            err_msg=name,
        )
        np.testing.assert_allclose(
            rows["duration"], mean_run_lengths / FPS, rtol=1e-12, err_msg=name
        )


def test_stats_df_kinematics(tutorial_stats_df, tutorial_moseq_df, tutorial_analysis_project):
    project_dir, model_name = tutorial_analysis_project
    results = kpms.load_results(project_dir, model_name)
    frames = recording_rows(tutorial_moseq_df, results)

    for name, rows in recording_rows(tutorial_stats_df, results).items():
        for _, row in rows.iterrows():
            values = frames[name].loc[frames[name]["syllable"] == row["syllable"], KINEMATICS]
            values = values.to_numpy()
            if len(values) == 0:
                columns = [c for c in row.index if c.startswith(tuple(KINEMATICS))]
                assert row[columns].isna().all(), f"{name} syllable {row['syllable']}"
                continue
            std = (
                values.std(axis=0, ddof=1)
                if len(values) > 1
                else np.full(len(KINEMATICS), np.nan)
            )
            expected = {
                "mean": values.mean(axis=0),
                "std": std,
                "min": values.min(axis=0),
                "max": values.max(axis=0),
            }
            # heading is an angle, so its mean and std are circular
            heading = values[:, KINEMATICS.index("heading")]
            expected["mean"][0] = stats.circmean(heading, high=np.pi, low=-np.pi)
            if len(values) > 1:
                expected["std"][0] = stats.circstd(heading, high=np.pi, low=-np.pi)
            for statistic, expected_values in expected.items():
                columns = [f"{kinematic}_{statistic}" for kinematic in KINEMATICS]
                np.testing.assert_allclose(
                    row[columns].to_numpy(dtype=float),
                    expected_values,
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"{name} syllable {row['syllable']} {statistic}",
                )


def test_stats_df_heading_wraps_around_pi(tmp_path):
    results = synthetic_results(["rec_a"], n_frames=300)
    syllables = results["rec_a"]["syllable"]
    near_pi = np.where(np.arange(300) % 2 == 0, np.pi - 0.05, -np.pi + 0.05)
    results["rec_a"]["heading"] = np.where(syllables == 0, near_pi, results["rec_a"]["heading"])
    moseq_df = kpms.compute_moseq_df(
        str(tmp_path), "model", results_dict=results, fps=FPS, smooth_heading=False
    )

    stats_df = kpms.compute_stats_df(moseq_df, min_frequency=0.0)

    row = stats_df[stats_df["syllable"] == 0].iloc[0]
    assert abs(row["heading_mean"]) > np.pi - 0.06
    assert row["heading_std"] < 0.1
    assert row["heading_min"] == pytest.approx(-np.pi + 0.05)
    assert row["heading_max"] == pytest.approx(np.pi - 0.05)


def test_stats_df_has_a_row_for_a_syllable_a_recording_never_used(tmp_path):
    results = synthetic_results(["rec_a", "rec_b"], num_syllables=3)
    used = results["rec_b"]["syllable"]
    results["rec_b"]["syllable"] = np.where(used == 1, 2, used)
    moseq_df = kpms.compute_moseq_df(str(tmp_path), "model", results_dict=results, fps=FPS)

    stats_df = kpms.compute_stats_df(moseq_df, min_frequency=0.0)

    row = stats_df[(stats_df["name"] == "rec_b") & (stats_df["syllable"] == 1)]
    assert len(row) == 1
    assert row["frequency"].item() == 0
    assert row[["duration", "heading_mean", "velocity_px_s_mean"]].isna().all(axis=None)


def test_stats_df_with_default_group(tutorial_moseq_df):
    moseq_df = tutorial_moseq_df.assign(group="default")

    stats_df = kpms.compute_stats_df(
        moseq_df,
        min_frequency=0.005,
        groupby=["group", "name"],
    )

    assert (stats_df["group"] == "default").all()
    assert set(stats_df["name"]) == set(moseq_df["name"])


def test_stats_df_needs_only_moseq_df(tmp_path):
    """No results.h5 or index.csv exists anywhere; everything comes from
    moseq_df."""
    results = synthetic_results(["rec_a", "rec_b"])
    moseq_df = kpms.compute_moseq_df(str(tmp_path), "model", results_dict=results, fps=FPS)

    stats_df = kpms.compute_stats_df(moseq_df, min_frequency=0.0)

    for name, rows in recording_rows(stats_df, results).items():
        syllables = rows["syllable"].to_numpy()
        np.testing.assert_array_equal(
            rows["frequency"],
            get_frequencies(results[name]["syllable"])[syllables],
            err_msg=name,
        )


def test_stats_df_rejects_a_recording_with_several_groups(tmp_path):
    results = synthetic_results(["rec_a", "rec_b"])
    moseq_df = kpms.compute_moseq_df(str(tmp_path), "model", results_dict=results, fps=FPS)
    first_rows = moseq_df.index[moseq_df["name"] == "rec_a"][:10]
    moseq_df.loc[first_rows, "group"] = "KO"

    with pytest.raises(ValueError) as error:
        kpms.compute_stats_df(moseq_df)

    message = str(error.value)
    assert "rec_a" in message
    assert "KO" in message
    assert "default" in message
    assert "rec_b" not in message
