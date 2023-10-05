"""This module contains the widget components in the analysis and visualiation
pipeline."""

# group setting widget imports
import os
import pandas as pd
import yaml
import ipywidgets as widgets
from IPython.display import display, clear_output

# video viewer widget additional imports
import io
import imageio
import base64
from bokeh.io import show
from bokeh.models import Div, CustomJS, Slider

# syllable labeler widget and controller additional imports
import numpy as np
import pandas as pd
from copy import deepcopy
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import Div, CustomJS, Slider
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText


def show_trajectory_gif(project_dir, model_name):
    """Show trajectory gif for syllable labeling.

    Parameters
    ----------
    progress_paths : dict
        dictionary of paths and filenames for progress tracking
    """
    trajectory_gifs_path = os.path.join(
        project_dir, model_name, "trajectory_plots", "all_trajectories.gif"
    )

    assert os.path.exists(trajectory_gifs_path), (
        f"Trajectory plots not found at {trajectory_gifs_path}. "
        "See documentation for generating trajectory plots: "
        "https://keypoint-moseq.readthedocs.io/en/latest/tutorial.html#visualization"
    )

    with open(trajectory_gifs_path, "rb") as file:
        image = file.read()
    out = widgets.Image(value=image, format="gif")
    display(out)
