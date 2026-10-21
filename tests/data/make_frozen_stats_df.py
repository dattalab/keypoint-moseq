"""Write the frozen `stats_df` used by the analysis notebook tests.

Usage::

    python tests/data/make_frozen_stats_df.py MODEL_DIR OUTPUT_CSV

`MODEL_DIR` is a fitted model directory containing `results.h5`. The script
copies `results.h5` into a temporary project, assigns the cohort groups
(recordings starting with "21_" to "a", the rest to "b") through the
group-setting widget, runs the analysis notebook's `moseq_df` and `stats_df`
cells, and writes `stats_df` to `OUTPUT_CSV`. It then prints the syllables
that `run_kruskal` marks as significant between the groups.

`frozen_stats_df.csv` was generated from model 2025_02_11-10_07_45, a
full-scale fit of the tutorial dataset in the maintainers' test assets.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import keypoint_moseq as kpms
from keypoint_moseq.analysis import run_kruskal


def main(model_dir, output_csv):
    model_name = os.path.basename(os.path.normpath(model_dir))
    with tempfile.TemporaryDirectory() as project_dir:
        os.makedirs(os.path.join(project_dir, model_name))
        shutil.copy(
            os.path.join(model_dir, "results.h5"),
            os.path.join(project_dir, model_name),
        )

        widget = kpms.interactive_group_setting(project_dir, model_name)
        table, button = widget[0], widget[1][0]
        table.patch(
            {
                "group": [
                    (row, "a" if name.startswith("21_") else "b")
                    for row, name in enumerate(table.value["name"])
                ]
            }
        )
        button.clicks += 1

        moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)
        stats_df = kpms.compute_stats_df(
            project_dir,
            model_name,
            moseq_df,
            min_frequency=0.005,
            groupby=["group", "name"],
            fps=30,
        )

    stats_df.to_csv(output_csv, index=False)
    reloaded = pd.read_csv(output_csv, float_precision="round_trip")
    pd.testing.assert_frame_equal(reloaded, stats_df, check_exact=True)

    np.random.seed(0)
    _, _, significant = run_kruskal(reloaded, statistic="frequency", thresh=0.05)
    print(f"Wrote {output_csv}: {len(reloaded)} rows")
    for pair, syllables in significant.items():
        print(f"Significant between {pair[0]} and {pair[1]}: {list(syllables)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
