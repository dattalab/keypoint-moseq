"""check_nan_proportions(breakdown=True) shows the fraction of NaN frames for
each recording and bodypart: as a styled table when IPython is available,
and as a printed table otherwise."""

import sys

import IPython.display
import numpy as np
import pandas as pd

import keypoint_moseq as kpms

BODYPARTS = ["nose", "tail"]


def coordinates_with_nans():
    """mouseA's nose is NaN in 1 of 4 frames, mouseB's tail in 2 of 4. A
    keypoint counts as NaN when any of its coordinates is."""
    mouseA = np.ones((4, 2, 2))
    mouseA[0, 0, 0] = np.nan
    mouseB = np.ones((4, 2, 2))
    mouseB[1:3, 1, 1] = np.nan
    return {"mouseB": mouseB, "mouseA": mouseA}


EXPECTED = pd.DataFrame(
    [[0.25, 0.0], [0.0, 0.5]], index=["mouseA", "mouseB"], columns=BODYPARTS
)


def test_breakdown_displays_a_shaded_table(monkeypatch):
    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)

    kpms.check_nan_proportions(coordinates_with_nans(), BODYPARTS, breakdown=True)

    (styler,) = displayed
    pd.testing.assert_frame_equal(styler.data, EXPECTED)
    html = styler.to_html()
    assert "Proportion of NaNs" in html
    assert "background-color: rgba(" in html


def test_breakdown_prints_a_table_without_ipython(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "IPython.display", None)

    kpms.check_nan_proportions(coordinates_with_nans(), BODYPARTS, breakdown=True)

    printed = capsys.readouterr().out
    assert printed.startswith("Proportion of NaNs\n")
    lines = printed.splitlines()
    assert any("nose" in line and "tail" in line for line in lines)
    assert any("mouseA" in line and "0.25" in line for line in lines)
    assert any("mouseB" in line and "0.5" in line for line in lines)
