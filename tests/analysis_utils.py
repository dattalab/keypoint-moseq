"""Helpers for testing the analysis notebook's outputs."""

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection
from PIL import Image

FPS = 30  # frame rate assumed by the analysis notebook's dataframe cells
PDF_SIGNATURE = b"%PDF"
PNG_SIGNATURE = b"\x89PNG"


def cohort_group(name):
    """Group of a tutorial recording in the analysis tests: "a" for the
    recordings from 2021 and "b" for those from 2022."""
    return "a" if name.startswith("21_") else "b"


def syllable_label(syllable):
    return f"label {syllable}"


def syllable_description(syllable):
    return f"description of syllable {syllable}"


def read_syll_info(path):
    """`syll_info.csv` read the way kpms reads it."""
    return pd.read_csv(path, index_col=False).fillna("")


def syllable_names(syll_info, syllables):
    """Tick labels of `syllables`: the index, followed by the label from
    `syll_info` in parentheses when there is one."""
    labels = dict(zip(syll_info["syllable"], syll_info["label"]))
    return [f"{s} ({labels[s]})" if labels[s] else f"{s}" for s in syllables]


def runs(labels):
    """Label and length of each run of repeated labels."""
    labels = np.asarray(labels)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(labels)) + 1])
    lengths = np.diff(np.concatenate([starts, [len(labels)]]))
    return labels[starts], lengths


def bigram_counts(labels, num_labels=100):
    """Counts of transitions between consecutive runs of `labels` below
    `num_labels`. Like `get_transitions`, the sequence of runs starts at the
    second run, so the transition out of the first run is not counted."""
    run_labels, _ = runs(labels)
    transitions = run_labels[1:]
    pairs = np.stack([transitions[:-1], transitions[1:]], axis=1)
    pairs = pairs[(pairs < num_labels).all(axis=1)]
    counts = np.zeros((num_labels, num_labels))
    np.add.at(counts, (pairs[:, 0], pairs[:, 1]), 1)
    return counts


def graph_edge_weights(matrix):
    """Edge weights of the undirected graph `networkx.from_numpy_array`
    builds from `matrix`. Nonzero entries are added in row-major order and a
    later entry replaces an earlier one, so when both directions of a pair
    are nonzero the edge keeps the entry below the diagonal."""
    weights = {}
    for u, v in zip(*np.nonzero(matrix)):
        weights[(min(u, v), max(u, v))] = matrix[u, v]
    return weights


def check_figure_files(save_path, png_size):
    """The png and pdf saved by `save_analysis_figure` at `save_path`
    (without extension) have valid headers, and the png has `png_size`."""
    for extension, signature in [("pdf", PDF_SIGNATURE), ("png", PNG_SIGNATURE)]:
        with open(f"{save_path}.{extension}", "rb") as f:
            assert f.read(len(signature)) == signature, extension
    with Image.open(f"{save_path}.png") as image:
        assert image.size == png_size


def syll_stats_plot_summary(fig, legend):
    """Tick labels, legend entries, significance stars and per-group points
    of a figure from `plot_syll_stats_with_sem`."""
    ax = fig.axes[0]
    num_syllables = len(ax.get_xticks())
    stars = [collection.get_offsets() for collection in ax.collections]
    return {
        "xlabel": ax.get_xlabel(),
        "xticklabels": [label.get_text() for label in ax.get_xticklabels()],
        "legend": [text.get_text() for text in legend.get_texts()],
        "stars": np.concatenate(stars) if stars else np.empty((0, 2)),
        # One line of markers per group; the error bars are separate
        # two-point lines without markers.
        "group_points": [
            np.asarray(line.get_ydata(), dtype=float)
            for line in ax.lines
            if line.get_marker() not in (None, "", "None")
            and len(line.get_ydata()) == num_syllables
        ],
    }


def bigram_plot_summary(fig):
    """Titles, images, color limits, tick labels and colorbar labels of a
    figure from `visualize_transition_bigram`."""
    image_axes = [ax for ax in fig.axes if ax.images]
    colorbar_axes = [ax for ax in fig.axes if not ax.images]
    return {
        "titles": [ax.get_title() for ax in image_axes],
        "images": [np.asarray(ax.images[0].get_array()) for ax in image_axes],
        "clims": [ax.images[0].get_clim() for ax in image_axes],
        "xticklabels": [
            [label.get_text() for label in ax.get_xticklabels()] for ax in image_axes
        ],
        "yticklabels": [label.get_text() for label in image_axes[0].get_yticklabels()],
        "colorbar_labels": [ax.get_ylabel() for ax in colorbar_axes],
    }


def graph_plot_summary(ax):
    """Nodes, edges and node labels that networkx drew on `ax`. Edges are
    identified by the indices of the nodes at their ends."""
    (nodes,) = [c for c in ax.collections if isinstance(c, PathCollection)]
    (edges,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    positions = np.asarray(nodes.get_offsets())
    segments = edges.get_segments()
    node_at = {tuple(xy): i for i, xy in enumerate(positions)}
    # A single color given for all nodes or edges is stored once.
    rgb = lambda colors, n: np.broadcast_to(np.asarray(colors)[:, :3], (n, 3))
    return {
        "title": ax.get_title(),
        "labels": [text.get_text() for text in ax.texts],
        "node_sizes": np.asarray(nodes.get_sizes()),
        "node_edgecolors": rgb(nodes.get_edgecolor(), len(positions)),
        "edges": [
            tuple(sorted(node_at[tuple(end)] for end in segment)) for segment in segments
        ],
        "edge_widths": np.asarray(edges.get_linewidths()),
        "edge_colors": rgb(edges.get_edgecolor(), len(segments)),
    }
