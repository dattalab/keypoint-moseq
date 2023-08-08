import os
import cv2
import tqdm
import imageio
import warnings
import logging
import h5py
import numpy as np
import plotly
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from vidio.read import OpenCVReader
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from textwrap import fill
from PIL import Image
from keypoint_moseq.util import *
from keypoint_moseq.io import load_results, _get_path
from jax_moseq.models.keypoint_slds import center_embedding
from jax_moseq.utils import get_durations, get_frequencies

# set matplotlib defaults
plt.rcParams["figure.dpi"] = 100

# suppress warnings from imageio
logging.getLogger().setLevel(logging.ERROR)


def crop_image(image, centroid, crop_size):
    """Crop an image around a centroid.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to crop.

    centroid: tuple of int
        (x,y) coordinates of the centroid.

    crop_size: int or tuple(int,int)
        Size of the crop around the centroid. Either a single int for a square
        crop, or a tuple of ints (w,h) for a rectangular crop.


    Returns
    -------
    image: ndarray of shape (crop_size, crop_size, 3)
        Cropped image.
    """
    if isinstance(crop_size, tuple):
        w, h = crop_size
    else:
        w, h = crop_size, crop_size
    x, y = int(centroid[0]), int(centroid[1])

    x_min = max(0, x - w // 2)
    y_min = max(0, y - h // 2)
    x_max = min(image.shape[1], x + w // 2)
    y_max = min(image.shape[0], y + h // 2)

    cropped = image[y_min:y_max, x_min:x_max]
    padded = np.zeros((h, w, *image.shape[2:]), dtype=image.dtype)
    pad_x = (w - cropped.shape[1]) // 2
    pad_y = (h - cropped.shape[0]) // 2
    padded[
        pad_y : pad_y + cropped.shape[0], pad_x : pad_x + cropped.shape[1]
    ] = cropped
    return padded


def plot_scree(pca, savefig=True, project_dir=None, fig_size=(3, 2)):
    """Plot explained variance as a function of the number of PCs.

    Parameters
    ----------
    pca : :py:func:`sklearn.decomposition.PCA`
        Fitted PCA model

    savefig : bool, True
        Whether to save the figure to a file. If true, the figure is saved to
        `{project_dir}/pca_scree.pdf`.

    project_dir : str, default=None
        Path to the project directory. Required if `savefig` is True.

    fig_size : tuple, (2.5,2)
        Size of the figure in inches.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Figure handle
    """
    fig = plt.figure()
    num_pcs = len(pca.components_)
    plt.plot(np.arange(num_pcs) + 1, np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("PCs")
    plt.ylabel("Explained variance")
    plt.gcf().set_size_inches(fig_size)
    plt.grid()
    plt.tight_layout()

    if savefig:
        assert project_dir is not None, fill(
            "The `savefig` option requires a `project_dir`"
        )
        plt.savefig(os.path.join(project_dir, "pca_scree.pdf"))
    plt.show()
    return fig


def plot_pcs(
    pca,
    *,
    use_bodyparts,
    skeleton,
    keypoint_colormap="autumn",
    savefig=True,
    project_dir=None,
    scale=1,
    plot_n_pcs=10,
    axis_size=(2, 1.5),
    ncols=5,
    node_size=30.0,
    linewidth=2.0,
    interactive=True,
    **kwargs,
):
    """
    Visualize the components of a fitted PCA model.

    For each PC, a subplot shows the mean pose (semi-transparent) along with a
    perturbation of the mean pose in the direction of the PC.

    Parameters
    ----------
    pca : :py:func:`sklearn.decomposition.PCA`
        Fitted PCA model

    use_bodyparts : list of str
        List of bodyparts to that are used in the model; used to index bodypart
        names in the skeleton.

    skeleton : list
        List of edges that define the skeleton, where each edge is a pair of
        bodypart names.

    keypoint_colormap : str
        Name of a matplotlib colormap to use for coloring the keypoints.

    savefig : bool, True
        Whether to save the figure to a file. If true, the figure is saved to
        `{project_dir}/pcs-{xy/xz/yz}.pdf` (`xz` and `yz` are only included
        for 3D data).

    project_dir : str, default=None
        Path to the project directory. Required if `savefig` is True.

    scale : float, default=0.5
        Scale factor for the perturbation of the mean pose.

    plot_n_pcs : int, default=10
        Number of PCs to plot.

    axis_size : tuple of float, default=(2,1.5)
        Size of each subplot in inches.

    ncols : int, default=5
        Number of columns in the figure.

    node_size : float, default=30.0
        Size of the keypoints in the figure.

    linewidth: float, default=2.0
        Width of edges in skeleton

    interactive : bool, default=True
        For 3D data, whether to generate an interactive 3D plot.
    """
    k = len(use_bodyparts)
    d = len(pca.mean_) // (k - 1)
    Gamma = np.array(center_embedding(k))
    edges = get_edges(use_bodyparts, skeleton)
    cmap = plt.colormaps[keypoint_colormap]
    plot_n_pcs = min(plot_n_pcs, pca.components_.shape[0])

    magnitude = np.sqrt((pca.mean_**2).mean()) * scale
    ymean = Gamma @ pca.mean_.reshape(k - 1, d)
    ypcs = (pca.mean_ + magnitude * pca.components_).reshape(-1, k - 1, d)
    ypcs = Gamma[np.newaxis] @ ypcs[:plot_n_pcs]

    if d == 2:
        dims_list, names = [[0, 1]], ["xy"]
    if d == 3:
        dims_list, names = [[0, 1], [0, 2]], ["xy", "xz"]

    for dims, name in zip(dims_list, names):
        nrows = int(np.ceil(plot_n_pcs / ncols))
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        for i, ax in enumerate(axs.flat):
            if i >= plot_n_pcs:
                ax.axis("off")
                continue

            for e in edges:
                ax.plot(
                    *ymean[:, dims][e].T,
                    color=cmap(e[0] / (k - 1)),
                    zorder=0,
                    alpha=0.25,
                    linewidth=linewidth,
                )
                ax.plot(
                    *ypcs[i][:, dims][e].T,
                    color="k",
                    zorder=2,
                    linewidth=linewidth + 0.2,
                )
                ax.plot(
                    *ypcs[i][:, dims][e].T,
                    color=cmap(e[0] / (k - 1)),
                    zorder=3,
                    linewidth=linewidth,
                )

            ax.scatter(
                *ymean[:, dims].T,
                c=np.arange(k),
                cmap=cmap,
                s=node_size,
                zorder=1,
                alpha=0.25,
                linewidth=0,
            )
            ax.scatter(
                *ypcs[i][:, dims].T,
                c=np.arange(k),
                cmap=cmap,
                s=node_size,
                zorder=4,
                edgecolor="k",
                linewidth=0.2,
            )

            ax.set_title(f"PC {i+1}", fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")

        fig.set_size_inches((axis_size[0] * ncols, axis_size[1] * nrows))
        plt.tight_layout()

        if savefig:
            assert project_dir is not None, fill(
                "The `savefig` option requires a `project_dir`"
            )
            plt.savefig(os.path.join(project_dir, f"pcs-{name}.pdf"))
        plt.show()

    if interactive and d == 3:
        plot_pcs_3D(
            ymean,
            ypcs,
            edges,
            keypoint_colormap,
            savefig,
            project_dir,
            node_size / 3,
            linewidth * 2,
        )


def plot_syllable_frequencies(
    project_dir=None,
    model_name=None,
    results=None,
    path=None,
    minlength=10,
    min_frequency=0.005,
):
    """Plot a histogram showing the frequency of each syllable.

    Caller must provide a results dictionary, a path to a results .h5, or a
    project directory and model name, in which case the results are loaded
    from `{project_dir}/{model_name}/results.h5`.

    Parameters
    ----------
    results : dict, default=None
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.extract_results`)

    model_name: str, default=None
        Name of the model. Required to load results if `results` is None and
        `path` is None.

    project_dir: str, default=None
        Project directory. Required to load results if `results` is None and
        `path` is None.

    path: str, default=None
        Path to a results file. If None, results will be loaded from
        `{project_dir}/{model_name}/results.h5`.

    minlength: int, default=10
        Minimum x-axis length of the histogram.

    min_frequency: float, default=0.005
        Minimum frequency of syllables to include in the histogram.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the histogram.

    ax : matplotlib.axes.Axes
        Axes containing the histogram.
    """
    if results is None:
        results = load_results(project_dir, model_name, path)

    syllables = {k: res["syllable"] for k, res in results.items()}
    frequencies = get_frequencies(syllables)
    frequencies = frequencies[frequencies > min_frequency]
    xmax = max(
        minlength, np.max(np.nonzero(frequencies > min_frequency)[0]) + 1
    )

    fig, ax = plt.subplots()
    ax.bar(range(len(frequencies)), frequencies, width=1)
    ax.set_ylabel("probability")
    ax.set_xlabel("syllable rank")
    ax.set_xlim(-1, xmax + 1)
    ax.set_title("Frequency distribution")
    ax.set_yticks([])
    return fig, ax


def plot_duration_distribution(
    project_dir=None,
    model_name=None,
    results=None,
    path=None,
    lim=None,
    num_bins=30,
    fps=None,
    show_median=True,
):
    """Plot a histogram showing the frequency of each syllable.

    Caller must provide a results dictionary, a path to a results .h5, or a
    project directory and model name, in which case the results are loaded from
    `{project_dir}/{model_name}/results.h5`.

    Parameters
    ----------
    results : dict, default=None
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.extract_results`)

    model_name: str, default=None
        Name of the model. Required to load results if `results` is None and
        `path` is None.

    project_dir: str, default=None
        Project directory. Required to load results if `results` is None and
        `path` is None.

    path: str, default=None
        Path to a results file. If None, results will be loaded from
        `{project_dir}/{model_name}/results.h5`.

    lim: tuple, default=None
        x-axis limits as a pair of ints (in units of frames). If None, the
        limits are set to (0, 95th-percentile).

    num_bins: int, default=30
        Number of bins in the histogram.

    fps: int, default=None
        Frames per second. Used to convert x-axis from frames to seconds.

    show_median: bool, default=True
        Whether to show the median duration as a vertical line.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the histogram.

    ax : matplotlib.axes.Axes
        Axes containing the histogram.
    """
    if results is None:
        results = load_results(project_dir, model_name, path)

    syllables = {k: res["syllable"] for k, res in results.items()}
    durations = get_durations(syllables)

    if lim is None:
        lim = int(np.percentile(durations, 95))
    binsize = max(int(np.floor(lim / num_bins)), 1)

    if fps is not None:
        durations = durations / fps
        binsize = binsize / fps
        lim = lim / fps
        xlabel = "syllable duration (s)"
    else:
        xlabel = "syllable duration (frames)"

    fig, ax = plt.subplots()
    ax.hist(durations, range=(0, lim), bins=(int(lim / binsize)), density=True)
    ax.set_xlim([0, lim])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability")
    ax.set_title("Duration distribution")
    ax.set_yticks([])
    if show_median:
        ax.axvline(np.median(durations), color="k", linestyle="--")
    return fig, ax


def plot_progress(
    model,
    data,
    checkpoint_path,
    iteration,
    project_dir=None,
    model_name=None,
    path=None,
    savefig=True,
    fig_size=None,
    window_size=600,
    min_frequency=0.001,
    min_histogram_length=10,
):
    """Plot the progress of the model during fitting.

    The figure shows the following plots:
        - Duration distribution:
            The distribution of state durations for the most recent iteration
            of the model.
        - Frequency distribution:
            The distribution of state frequencies for the most recent iteration
            of the model.
        - Median duration:
            The median state duration across iterations.
        - State sequence history
            The state sequence across iterations in a random window (a new
            window is selected each time the progress is plotted).

    Parameters
    ----------
    model : dict
        Model dictionary containing `states`

    data : dict
        Data dictionary containing `mask`

    checkpoint_path : str
        Path to an HDF5 file containing model checkpoints.

    iteration : int
        Current iteration of model fitting

    project_dir : str, default=None
        Path to the project directory. Required if `savefig` is True.

    model_name : str, default=None
        Name of the model. Required if `savefig` is True.

    savefig : bool, default=True
        Whether to save the figure to a file. If true, the figure is either
        saved to `path` or, to `{project_dir}/{model_name}-progress.pdf` if
        `path` is None.

    fig_size : tuple of float, default=None
        Size of the figure in inches.

    window_size : int, default=600
        Window size for state sequence history plot.

    min_frequency : float, default=.001
        Minimum frequency for including a state in the frequency distribution
        plot.

    min_histogram_length : int, default=10
        Minimum x-axis length of the frequency distribution plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plots.

    axs : list of matplotlib.axes.Axes
        Axes containing the plots.
    """
    z = np.array(model["states"]["z"])
    mask = np.array(data["mask"])
    durations = get_durations(z, mask)
    frequencies = get_frequencies(z, mask)

    with h5py.File(checkpoint_path, "r") as f:
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])

    if len(saved_iterations) > 1:
        fig, axs = plt.subplots(
            1, 4, gridspec_kw={"width_ratios": [1, 1, 1, 3]}
        )
        if fig_size is None:
            fig_size = (12, 2.5)
    else:
        fig, axs = plt.subplots(1, 2)
        if fig_size is None:
            fig_size = (4, 2.5)

    frequencies = np.sort(frequencies[frequencies > min_frequency])[::-1]
    xmax = max(len(frequencies), min_histogram_length)
    axs[0].bar(range(len(frequencies)), frequencies, width=1)
    axs[0].set_ylabel("probability")
    axs[0].set_xlabel("syllable rank")
    axs[0].set_xlim([-1, xmax + 1])
    axs[0].set_title("Frequency distribution")
    axs[0].set_yticks([])

    lim = int(np.percentile(durations, 95))
    binsize = max(int(np.floor(lim / 30)), 1)
    axs[1].hist(
        durations, range=(1, lim), bins=(int(lim / binsize)), density=True
    )
    axs[1].set_xlim([1, lim])
    axs[1].set_xlabel("syllable duration (frames)")
    axs[1].set_ylabel("probability")
    axs[1].set_title("Duration distribution")
    axs[1].set_yticks([])

    if len(saved_iterations) > 1:
        window_size = int(min(window_size, mask.max(0).sum() - 1))
        nz = np.stack(np.array(mask[:, window_size:]).nonzero(), axis=1)
        batch_ix, start = nz[np.random.randint(nz.shape[0])]

        sample_state_history = []
        median_durations = []

        for i in saved_iterations:
            with h5py.File(checkpoint_path, "r") as f:
                z = np.array(f[f"model_snapshots/{i}/states/z"])
                sample_state_history.append(
                    z[batch_ix, start : start + window_size]
                )
                median_durations.append(np.median(get_durations(z, mask)))

        axs[2].scatter(saved_iterations, median_durations)
        axs[2].set_ylim([-1, np.max(median_durations) * 1.1])
        axs[2].set_xlabel("iteration")
        axs[2].set_ylabel("duration")
        axs[2].set_title("Median duration")

        axs[3].imshow(
            sample_state_history,
            cmap=plt.cm.jet,
            aspect="auto",
            interpolation="nearest",
        )
        axs[3].set_xlabel("Time (frames)")
        axs[3].set_ylabel("Iterations")
        axs[3].set_title("State sequence history")

        yticks = [
            int(y)
            for y in axs[3].get_yticks()
            if y < len(saved_iterations) and y > 0
        ]
        yticklabels = saved_iterations[yticks]
        axs[3].set_yticks(yticks)
        axs[3].set_yticklabels(yticklabels)

    title = f"Iteration {iteration}"
    if model_name is not None:
        title = f"{model_name}: {title}"
    fig.suptitle(title)
    fig.set_size_inches(fig_size)
    plt.tight_layout()

    if savefig:
        path = _get_path(project_dir, model_name, path, "fitting_progress.pdf")
        plt.savefig(path)
    plt.show()
    return fig, axs


def write_video_clip(frames, path, fps=30, quality=7):
    """Write a video clip to a file.

    Parameters
    ----------
    frames : np.ndarray
        Video frames as a 4D array of shape `(num_frames, height, width, 3)`
        or a 3D array of shape `(num_frames, height, width)`.

    path : str
        Path to save the video clip.

    fps : int, default=30
        Framerate of video encoding.

    quality : int, default=7
        Quality of video encoding.
    """
    with imageio.get_writer(
        path, pixelformat="yuv420p", fps=fps, quality=quality
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _grid_movie_tile(
    key,
    start,
    end,
    videos,
    centroids,
    headings,
    dot_color,
    window_size,
    scaled_window_size,
    pre,
    post,
    dot_radius,
    overlay_keypoints,
    edges,
    coordinates,
    plot_options,
):
    scale_factor = scaled_window_size / window_size
    cs = centroids[key][start - pre : start + post]
    h, c = headings[key][start], cs[pre]
    r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

    tile = []

    if videos is not None:  # overlay keypoints on video frame, then transform
        frames = videos[key][start - pre : start + post]
        c = r @ c - window_size // 2
        M = [[np.cos(h), np.sin(h), -c[0]], [-np.sin(h), np.cos(h), -c[1]]]

        for ii, (frame, c) in enumerate(zip(frames, cs)):
            if overlay_keypoints:
                coords = coordinates[key][start - pre + ii]
                frame = overlay_keypoints_on_image(
                    frame, coords, edges=edges, **plot_options
                )

            frame = cv2.warpAffine(
                frame, np.float32(M), (window_size, window_size)
            )
            frame = cv2.resize(frame, (scaled_window_size, scaled_window_size))
            if 0 <= ii - pre <= end - start and dot_radius > 0:
                pos = tuple(
                    [int(x) for x in M @ np.append(c, 1) * scale_factor]
                )
                cv2.circle(frame, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            tile.append(frame)

    else:  # first transform keypoints, then overlay on black background
        assert overlay_keypoints, fill(
            "If no videos are provided, then `overlay_keypoints` must "
            "be True. Otherwise there is nothing to show"
        )
        scale_factor = scaled_window_size / window_size
        coords = coordinates[key][start - pre : start + post]
        coords = (coords - c) @ r.T * scale_factor + scaled_window_size // 2
        cs = (cs - c) @ r.T * scale_factor + scaled_window_size // 2
        background = np.zeros((scaled_window_size, scaled_window_size, 3))
        for ii, (uvs, c) in enumerate(zip(coords, cs)):
            frame = overlay_keypoints_on_image(
                background.copy(), uvs, edges=edges, **plot_options
            )
            if 0 <= ii - pre <= end - start and dot_radius > 0:
                pos = (int(c[0]), int(c[1]))
                cv2.circle(frame, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            tile.append(frame)

    return np.stack(tile)


def grid_movie(
    instances,
    rows,
    cols,
    videos,
    centroids,
    headings,
    window_size,
    dot_color=(255, 255, 255),
    dot_radius=4,
    pre=30,
    post=60,
    scaled_window_size=None,
    edges=[],
    overlay_keypoints=False,
    coordinates=None,
    plot_options={},
):
    """Generate a grid movie and return it as an array of frames.

    Grid movies show many instances of a syllable. Each instance contains a
    snippet of video (and/or keypoint-overlay) centered on the animal and
    synchronized to the onset of the syllable. A dot appears at syllable onset
    and disappears at syllable offset.

    Parameters
    ----------
    instances: list of tuples `(key, start, end)`
        List of syllable instances to include in the grid movie, where each
        instance is specified as a tuple with the video name, start frame and
        end frame. The list must have length `rows*cols`. The video names must
        also be keys in `videos`.

    rows: int, cols : int
        Number of rows and columns in the grid movie grid

    videos: dict or None
        Dictionary mapping video names to video readers. Frames from
        each reader should be accessible via `__getitem__(int or slice)`. If
        None, the the grid movie will not include video frames.

    centroids: dict
        Dictionary mapping video names to arrays of shape `(n_frames, 2)` with
        the x,y coordinates of animal centroid on each frame

    headings: dict
        Dictionary mapping video names to arrays of shape `(n_frames,)` with
        the heading of the animal on each frame (in radians)

    window_size: int
        Size of the window around the animal. This should be a multiple of 16
        or imageio will complain.

    dot_color: tuple of ints, default=(255,255,255)
        RGB color of the dot indicating syllable onset and offset

    dot_radius: int, default=4
        Radius of the dot indicating syllable onset and offset

    pre: int, default=30
        Number of frames before syllable onset to include in the movie

    post: int, default=60
        Number of frames after syllable onset to include in the movie

    scaled_window_size: int, default=None
        Window size after scaling the video. If None, the no scaling is
        performed (i.e. `scaled_window_size = window_size`)

    overlay_keypoints: bool, default=False
        If True, overlay the pose skeleton on the video frames.

    edges: list of tuples, default=[]
        List of edges defining pose skeleton. Used when
        `overlay_keypoints=True`.

    coordinates: dict, default=None
        Dictionary mapping video names to arrays of shape `(n_frames, 2)`.
        Used when `overlay_keypoints=True`.

    plot_options: dict, default={}
        Dictionary of options to pass to `overlay_keypoints_on_image`.
        Used when `overlay_keypoints=True`.

    Returns
    -------
    frames: array of shape `(post+pre, width, height, 3)`
        Array of frames in the grid movie where::

            width = rows * scaled_window_size
            height = cols * scaled_window_size
    """
    if videos is None:
        assert overlay_keypoints, fill(
            "If no videos are provided, then `overlay_keypoints` must "
            "be True. Otherwise there is nothing to show"
        )

    if scaled_window_size is None:
        scaled_window_size = window_size

    tiles = []
    for key, start, end in instances:
        tiles.append(
            _grid_movie_tile(
                key,
                start,
                end,
                videos,
                centroids,
                headings,
                dot_color,
                window_size,
                scaled_window_size,
                pre,
                post,
                dot_radius,
                overlay_keypoints,
                edges,
                coordinates,
                plot_options,
            )
        )

    tiles = np.stack(tiles).reshape(
        rows, cols, post + pre, scaled_window_size, scaled_window_size, 3
    )
    frames = np.concatenate(np.concatenate(tiles, axis=2), axis=2)
    return frames


def get_grid_movie_window_size(
    sampled_instances,
    centroids,
    headings,
    coordinates,
    pre,
    post,
    pctl=90,
    fudge_factor=1.1,
    blocksize=16,
):
    """Automatically determine the window size for a grid movie.

    The window size is set such that across all sampled instances,
    the animal is fully visible in at least `pctl` percent of frames.

    Parameters
    ----------
    sampled_instances: dict
        Dictionary mapping syllables to lists of instances, where each
        instance is specified as a tuple with the video name, start frame
        and end frame.

    centroids: dict
        Dictionary mapping video names to arrays of shape `(n_frames, 2)`
        with the x,y coordinates of animal centroid on each frame

    headings: dict
        Dictionary mapping video names to arrays of shape `(n_frames,)`
        with the heading of the animal on each frame (in radians)

    coordinates: dict
        Dictionary mapping recording names to keypoint coordinates as
        ndarrays of shape (n_frames, n_bodyparts, 2).

    pre, post: int
        Number of frames before/after syllable onset that are included
        in the grid movies.

    pctl: int, default=95
        Percentile of frames in which the animal should be fully visible.

    fudge_factor: float, default=1.1
        Factor by which to multiply the window size.

    blocksize: int, default=16
        Window size is rounded up to the nearest multiple of `blocksize`.
    """
    all_trajectories = get_instance_trajectories(
        sum(sampled_instances.values(), []),
        coordinates,
        pre=pre,
        post=post,
        centroids=centroids,
        headings=headings,
    )

    all_trajectories = np.concatenate(all_trajectories, axis=0)
    all_trajectories = all_trajectories[
        ~np.isnan(all_trajectories).all((1, 2))
    ]
    max_distances = np.nanmax(np.abs(all_trajectories), axis=1)
    window_size = np.percentile(max_distances, pctl) * fudge_factor * 2
    window_size = int(np.ceil(window_size / blocksize) * blocksize)
    return window_size


def generate_grid_movies(
    results,
    project_dir=None,
    model_name=None,
    output_dir=None,
    video_dir=None,
    video_paths=None,
    rows=4,
    cols=6,
    filter_size=9,
    pre=30,
    post=60,
    min_frequency=0.005,
    min_duration=3,
    dot_radius=4,
    dot_color=(255, 255, 255),
    quality=7,
    window_size=None,
    coordinates=None,
    bodyparts=None,
    use_bodyparts=None,
    sampling_options={},
    video_extension=None,
    max_video_size=1920,
    skeleton=[],
    overlay_keypoints=False,
    keypoints_only=False,
    fps=30,
    plot_options={},
    use_dims=[0, 1],
    keypoint_colormap="autumn",
    **kwargs,
):
    """Generate grid movies for a modeled dataset.

    Grid movies show many instances of a syllable and are useful in
    figuring out what behavior the syllable captures
    (see :py:func:`keypoint_moseq.viz.grid_movie`). This method
    generates a grid movie for each syllable that is used sufficiently
    often (i.e. has at least `rows*cols` instances with duration
    of at least `min_duration` and an overall frequency of at least
    `min_frequency`). The grid movies are saved to `output_dir` if
    specified, or else to `{project_dir}/{model_name}/grid_movies`.

    Parameters
    ----------
    results: dict
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.extract_results`)

    project_dir: str, default=None
        Project directory. Required to save grid movies if `output_dir`
        is None.

    model_name: str, default=None
        Name of the model. Required to save grid movies if
        `output_dir` is None.

    output_dir: str, default=None
        Directory where grid movies should be saved. If None, grid
        movies will be saved to `{project_dir}/{model_name}/grid_movies`.

    video_dir: str, default=None
        Directory containing videos of the modeled data (see
        :py:func:`keypoint_moseq.io.find_matching_videos`).
        Unless `keypoints_only=True`, either `video_dir` or
        `video_paths` must be provided.

    video_paths: dict, default=None
        Dictionary mapping recording names to video paths. The recording
        names must correspond to keys in `results['syllables']`.
        Unless `keypoints_only=True`, either `video_dir` or
        `video_paths` must be provided.

    filter_size: int, default=9
        Size of the median filter applied to centroids and headings

    min_frequency: float, default=0.005
        Minimum frequency of a syllable to be included in the grid movies.

    min_duration: int, default=3
        Minimum duration of a syllable instance to be included in the
        grid movie for that syllable.

    sampling_options: dict, default={}
        Dictionary of options for sampling syllable instances (see
        :py:func:`keypoint_moseq.util.sample_instances`).

    coordinates: dict, default=None
        Dictionary mapping recording names to keypoint coordinates as
        ndarrays of shape (n_frames, n_bodyparts, [2 or 3]). Required when
        `window_size=None`, or `overlay_keypoints=True`, or if using
        density-based sampling (i.e. when `sampling_options['mode']=='density'`;
        see :py:func:`keypoint_moseq.util.sample_instances`).

    bodyparts: list of str, default=None
        List of bodypart names in `coordinates`. Required when `coordinates` is
        provided and bodyparts were reindexed for modeling.

    use_bodyparts: list of str, default=None
        Ordered list of bodyparts used for modeling. Required when
        `coordinates` is provided and bodyparts were reindexed
        for modeling.

    quality: int, default=7
        Quality of the grid movies. Higher values result in higher
        quality movies but larger file sizes.

    rows, cols, pre, post, dot_radius, dot_color, window_size
        See :py:func:`keypoint_moseq.viz.grid_movie`

    video_extension: str, default=None
        Preferred video extension (passed to
        :py:func:`keypoint_moseq.util.find_matching_videos`)

    window_size: int, default=None
        Size of the window around the animal. If None, the window
        size is determined automatically based on the size of the
        animal. If provided explicitly, `window_size` should be a
        multiple of 16 or imageio will complain.

    max_video_size: int, default=4000
        Maximum size of the grid movie in pixels. If the grid movie
        is larger than this, it will be downsampled.

    skeleton: list of tuples, default=[]
        List of tuples specifying the skeleton. Used when
        `overlay_keypoints=True`.

    overlay_keypoints: bool, default=False
        Whether to overlay the keypoints on the grid movie.

    keypoints_only: bool, default=False
        Whether to only show the keypoints (i.e. no video frames).
        Overrides `overlay_keypoints`. When this option is used,
        the framerate should be explicitly specified using `fps`.

    fps: int, default=30
        Framerate of the grid movie. When `keypoints_only=False`,
        this parameter is ignored and the framerate is determined
        inferred from the video files.

    plot_options: dict, default={}
        Dictionary of options to pass to
        :py:func:`keypoint_moseq.viz.overlay_keypoints_on_image`.

    use_dims: pair of ints, default=[0,1]
        Dimensions to use for plotting keypoints. Only used when
        `overlay_keypoints=True` and the keypoints are 3D.

    keypoint_colormap: str, default='autumn'
        Colormap used to color keypoints. Used when
        `overlay_keypoints=True`.


    See :py:func:`keypoint_moseq.viz.grid_movie` for the remaining parameters.
    """
    # check inputs
    if not keypoints_only:
        assert (video_dir is not None) or (video_paths is not None), fill(
            "Either `video_dir` or `video_paths` is required unless `keypoints_only=True`"
        )
    elif not overlay_keypoints:
        warnings.warn(
            "Setting `overlay_keypoints=True` since `keypoints_only=True`"
        )
        overlay_keypoints = True

    if window_size is None or overlay_keypoints:
        assert coordinates is not None, fill(
            "`coordinates` must be provided if `window_size` is None "
            "or `overlay_keypoints` is True"
        )

    # prepare output directory
    output_dir = _get_path(
        project_dir, model_name, output_dir, "grid_movies", "output_dir"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Writing grid movies to {output_dir}")

    # reindex coordinates if necessary
    if not (bodyparts is None or use_bodyparts is None or coordinates is None):
        coordinates = reindex_by_bodyparts(
            coordinates, bodyparts, use_bodyparts
        )

    # get edges for plotting skeleton
    edges = []
    if len(skeleton) > 0 and overlay_keypoints:
        edges = get_edges(use_bodyparts, skeleton)

    # load results
    if results is None:
        results = load_results(project_dir, model_name)

    syllables = {k: v["syllable"] for k, v in results.items()}
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    # load video readers if necessary
    if video_paths is None and not keypoints_only:
        video_paths = find_matching_videos(
            results.keys(),
            video_dir,
            as_dict=True,
            video_extension=video_extension,
        )
        videos = {k: OpenCVReader(path) for k, path in video_paths.items()}
        fps = list(videos.values())[0].fps
    else:
        videos = None

    # sample instances for each syllable
    syllable_instances = get_syllable_instances(
        syllables,
        pre=pre,
        post=post,
        min_duration=min_duration,
        min_frequency=min_frequency,
        min_instances=rows * cols,
    )

    if len(syllable_instances) == 0:
        warnings.warn(
            fill(
                "No syllables with sufficient instances to make a grid movie. "
                "This usually occurs when all frames have the same syllable label "
                "(use `plot_syllable_frequencies` to check if this is the case)"
            )
        )
        return

    sampled_instances = sample_instances(
        syllable_instances,
        rows * cols,
        coordinates=coordinates,
        centroids=centroids,
        headings=headings,
        **sampling_options,
    )

    # if the data is 3D, pick 2 dimensions to use for plotting
    keypoint_dimension = next(iter(centroids.values())).shape[-1]
    if keypoint_dimension == 3:
        ds = np.array(use_dims)
        centroids = {k: v[:, ds] for k, v in centroids.items()}
        if coordinates is not None:
            coordinates = {k: v[:, :, ds] for k, v in coordinates.items()}

    # smooth centroids and headings
    centroids, headings = filter_centroids_headings(
        centroids, headings, filter_size=filter_size
    )

    # determine window size for grid movies
    if window_size is None:
        window_size = get_grid_movie_window_size(
            sampled_instances, centroids, headings, coordinates, pre, post
        )

    # possibly reduce window size to keep grid movies under max_video_size
    scaled_window_size = max_video_size / max(rows, cols)
    scaled_window_size = int(np.floor(scaled_window_size / 16) * 16)
    scaled_window_size = min(scaled_window_size, window_size)
    scale_factor = scaled_window_size / window_size

    if scale_factor < 1:
        warnings.warn(
            "\n"
            + fill(
                f"Videos will be downscaled by a factor of {scale_factor:.2f} "
                f"so that the grid movies are under {max_video_size} pixels. "
                "Use `max_video_size` to increase or decrease this size limit."
            )
            + "\n\n"
        )

    # add colormap to plot options
    plot_options.update({"keypoint_colormap": keypoint_colormap})

    # generate grid movies
    for syllable, instances in tqdm.tqdm(
        sampled_instances.items(), desc="Generating grid movies", ncols=72
    ):
        frames = grid_movie(
            instances,
            rows,
            cols,
            videos,
            centroids,
            headings,
            edges=edges,
            window_size=window_size,
            scaled_window_size=scaled_window_size,
            dot_color=dot_color,
            pre=pre,
            post=post,
            dot_radius=dot_radius,
            overlay_keypoints=overlay_keypoints,
            coordinates=coordinates,
            plot_options=plot_options,
        )

        path = os.path.join(output_dir, f"syllable{syllable}.mp4")
        write_video_clip(frames, path, fps=fps, quality=quality)


def get_limits(
    coordinates,
    pctl=1,
    blocksize=None,
    left=0.2,
    right=0.2,
    top=0.2,
    bottom=0.2,
):
    """Get axis limits based on the coordinates of all keypoints.

    For each axis, limits are determined using the percentiles
    `pctl` and `100-pctl` and then padded by `padding`.

    Parameters
    ----------
    coordinates: ndarray or dict
        Coordinates as an ndarray of shape (..., 2), or a dict
        with values that are ndarrays of shape (..., 2).

    pctl: float, default=1
        Percentile to use for determining the axis limits.

    blocksize: int, default=None
        Axis limits are cast to integers and padded so that the width
        and height are multiples of `blocksize`. This is useful
        when they are used for generating cropped images for a video.

    left, right, top, bottom: float, default=0.1
        Fraction of the axis range to pad on each side.

    Returns
    -------
    lims: ndarray of shape (2,dim)
        Axis limits, in the format `[[xmin,ymin,...],[xmax,ymax,...]]`.
    """
    if isinstance(coordinates, dict):
        X = np.concatenate(list(coordinates.values())).reshape(-1, 2)
    else:
        X = coordinates.reshape(-1, 2)

    xmin, ymin = np.nanpercentile(X, pctl, axis=0)
    xmax, ymax = np.nanpercentile(X, 100 - pctl, axis=0)

    width = xmax - xmin
    height = ymax - ymin
    xmin -= width * left
    xmax += width * right
    ymin -= height * bottom
    ymax += height * top

    lims = np.array([[xmin, ymin], [xmax, ymax]])

    if blocksize is not None:
        lims = np.round(lims)
        padding = np.mod(lims[0] - lims[1], blocksize) / 2
        lims[0] -= padding
        lims[1] += padding
        lims = np.ceil(lims).astype(int)

    return lims


def rasterize_figure(fig):
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    raster_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    raster = raster_flat.reshape((height, width, 3))
    return raster


def plot_trajectories(
    titles,
    Xs,
    lims,
    edges=[],
    n_cols=4,
    invert=False,
    keypoint_colormap="autumn",
    node_size=50,
    line_width=3,
    alpha=0.2,
    num_timesteps=10,
    plot_width=4,
    overlap=(0.2, 0),
    return_rasters=False,
):
    """Plot one or more pose trajectories on a common axis and return the axis.

    (See :py:func:`keypoint_moseq.viz.generate_trajectory_plots`)

    Parameters
    ----------
    titles: list of str
        List of titles for each trajectory plot.

    Xs: list of ndarray
        List of pose trajectories as ndarrays of shape
        (n_frames, n_keypoints, 2).

    edges: list of tuples, default=[]
        List of edges, where each edge is a tuple of two integers

    lims: ndarray
        Axis limits used for all the trajectory plots. The limits
        should be provided as an array of shape (2,2) with the format
        `[[xmin,ymin],[xmax,ymax]]`.

    n_cols: int, default=4
        Number of columns in the figure (used when plotting multiple
        trajectories).

    invert: bool, default=False
        Determines the background color of the figure. If `True`,
        the background will be black.

    keypoint_colormap : str or list
        Name of a matplotlib colormap or a list of colors as (r,b,g)
        tuples in the same order as as the keypoints.

    node_size: int, default=50
        Size of each keypoint.

    line_width: int, default=3
        Width of the lines connecting keypoints.

    alpha: float, default=0.2
        Opacity of fade-out layers.

    num_timesteps: int, default=10
        Number of timesteps to plot for each trajectory. The pose
        at each timestep is determined by linearly interpolating
        between the keypoints.

    plot_width: int, default=4
        Width of each trajectory plot in inches. The height  is
        determined by the aspect ratio of `lims`. The final figure
        width is `fig_width * min(n_cols, len(X))`.

    overlap: tuple of float, default=(0.2,0)
        Amount of overlap between each trajectory plot as a tuple
        with the format `(x_overlap, y_overlap)`. The values should
        be between 0 and 1.

    return_rasters: bool, default=False
        Rasterize the matplotlib canvas after plotting each step of
        the trajecory. This is used to generate an animated video/gif
        of the trajectory.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Figure handle

    ax: matplotlib.axes.Axes
        Axis containing the trajectory plots.
    """
    fill_color = "k" if invert else "w"
    if isinstance(keypoint_colormap, list):
        colors = keypoint_colormap
    else:
        colors = plt.colormaps[keypoint_colormap](
            np.linspace(0, 1, Xs[0].shape[1])
        )

    n_cols = min(n_cols, len(Xs))
    n_rows = np.ceil(len(Xs) / n_cols)
    offsets = np.stack(
        np.meshgrid(
            np.arange(n_cols) * np.diff(lims[:, 0]) * (1 - overlap[0]),
            np.arange(n_rows) * np.diff(lims[:, 1]) * (overlap[1] - 1),
        ),
        axis=-1,
    ).reshape(-1, 2)[: len(Xs)]

    Xs = interpolate_along_axis(
        np.linspace(0, Xs[0].shape[0], num_timesteps),
        np.arange(Xs[0].shape[0]),
        np.array(Xs),
        axis=1,
    )

    Xs = Xs + offsets[:, None, None]
    xmin, ymin = lims[0] + offsets.min(0)
    xmax, ymax = lims[1] + offsets.max(0)

    fig, ax = plt.subplots(frameon=False)
    ax.fill_between(
        [xmin, xmax],
        y1=[ymax, ymax],
        y2=[ymin, ymin],
        facecolor=fill_color,
        zorder=0,
        clip_on=False,
    )

    title_xy = (lims * np.array([[0.5, 0.1], [0.5, 0.9]])).sum(0)
    title_color = "w" if invert else "k"

    for xy, text in zip(offsets + title_xy, titles):
        ax.text(
            *xy,
            text,
            c=title_color,
            ha="center",
            va="top",
            zorder=Xs.shape[1] * 4 + 4,
        )

    # final extents in axis
    final_width = xmax - xmin
    final_height = title_xy[1] - ymin

    fig_width = plot_width * (n_cols - (n_cols - 1) * overlap[0])
    fig_height = final_height / final_width * fig_width
    fig.set_size_inches((fig_width, fig_height))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    rasters = []  # for making a gif

    for i in range(Xs.shape[1]):
        for X, offset in zip(Xs, offsets):
            for ii, jj in edges:
                ax.plot(
                    *X[i, (ii, jj)].T,
                    c="k",
                    zorder=i * 4,
                    linewidth=line_width,
                    clip_on=False,
                )

            for ii, jj in edges:
                ax.plot(
                    *X[i, (ii, jj)].T,
                    c=colors[ii],
                    zorder=i * 4 + 1,
                    linewidth=line_width * 0.9,
                    clip_on=False,
                )

            ax.scatter(
                *X[i].T,
                c=colors,
                zorder=i * 4 + 2,
                edgecolor="k",
                linewidth=0.4,
                s=node_size,
                clip_on=False,
            )

        if i < Xs.shape[1] - 1:
            ax.fill_between(
                [xmin, xmax],
                y1=[ymax, ymax],
                y2=[ymin, ymin],
                facecolor=fill_color,
                alpha=alpha,
                zorder=i * 4 + 3,
                clip_on=False,
            )

        if return_rasters:
            rasters.append(rasterize_figure(fig))

    return fig, ax, rasters


def save_gif(image_list, gif_filename, duration=0.5):
    # Convert NumPy arrays to PIL Image objects
    pil_images = [Image.fromarray(np.uint8(img)) for img in image_list]

    # Save the PIL Images as an animated GIF
    pil_images[0].save(
        gif_filename,
        save_all=True,
        append_images=pil_images[1:],
        duration=int(duration * 1000),
        loop=0,
    )


def generate_trajectory_plots(
    coordinates,
    results,
    project_dir=None,
    model_name=None,
    output_dir=None,
    pre=5,
    post=15,
    min_frequency=0.005,
    min_duration=3,
    skeleton=[],
    bodyparts=None,
    use_bodyparts=None,
    keypoint_colormap="autumn",
    plot_options={},
    padding={"left": 0.1, "right": 0.1, "top": 0.2, "bottom": 0.2},
    save_individually=True,
    save_gifs=True,
    save_mp4s=False,
    fps=30,
    projection_planes=["xy", "xz"],
    interactive=True,
    density_sample=True,
    sampling_options={"mode": "density", "n_neighbors": 50},
    **kwargs,
):
    """
    Generate trajectory plots for a modeled dataset.

    Each trajectory plot shows a sequence of poses along the average
    trajectory through latent space associated with a given syllable.
    A separate figure (and gif, optionally) is saved for each syllable,
    along with a single figure showing all syllables in a grid. The
    plots are saved to `{output_dir}` if it is provided, otherwise
    they are saved to `{project_dir}/{model_name}/trajectory_plots`.

    Plot-related parameters are described below. For the remaining
    parameters see (:py:func:`keypoint_moseq.util.get_typical_trajectories`)

    Parameters
    ----------
    coordinates: dict
        Dictionary mapping recording names to keypoint coordinates as
        ndarrays of shape (n_frames, n_bodyparts, [2 or 3]).

    results: dict
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.extract_results`).

    project_dir: str, default=None
        Project directory. Required to save trajectory plots if
        `output_dir` is None.

    model_name: str, default=None
        Name of the model. Required to save trajectory plots if
        `output_dir` is None.

    output_dir: str, default=None
        Directory where trajectory plots should be saved. If None,
        plots will be saved to `{project_dir}/{model_name}/trajectory_plots`.

    skeleton : list, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of bodypart names or a pair of indexes.

    keypoint_colormap : str
        Name of a matplotlib colormap to use for coloring the keypoints.

    plot_options: dict, default={}
        Dictionary of options for trajectory plots (see
        :py:func:`keypoint_moseq.util.plot_trajectories`).

    padding: dict, default={'left':0.1, 'right':0.1, 'top':0.2, 'bottom':0.2}
        Padding around trajectory plots. Controls the the distance
        between trajectories (when multiple are shown in one figure)
        as well as the title offset.

    save_individually: bool, default=True
        If True, a separate figure is saved for each syllable (in
        addition to the grid figure).

    save_gifs: bool, default=True
        Whether to save an animated gif of the trajectory plots.

    save_mp4s: bool, default=False
        Whether to save videos of the trajectory plots as .mp4 files

    fps: int, default=30
        Framerate of the videos from which keypoints were derived.
        Used to set the framerate of gifs when `save_gif=True`.

    projection_planes: list (subset of ['xy', 'yz', 'xz']), default=['xy','xz']
        For 3D data, defines the 2D plane(s) on which to project keypoint
        coordinates. A separate plot will be saved for each plane with
        the name of the plane (e.g. 'xy') as a suffix. This argument is
        ignored for 2D data.

    interactive: bool, default=True
        For 3D data, whether to create an visualization that can be
        rotated and zoomed. This argument is ignored for 2D data.
    """
    plot_options.update({"keypoint_colormap": keypoint_colormap})
    edges = [] if len(skeleton) == 0 else get_edges(use_bodyparts, skeleton)

    output_dir = _get_path(
        project_dir, model_name, output_dir, "trajectory_plots", "output_dir"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving trajectory plots to {output_dir}")

    typical_trajectories = get_typical_trajectories(
        coordinates,
        results,
        pre,
        post,
        min_frequency,
        min_duration,
        bodyparts,
        use_bodyparts,
        density_sample,
        sampling_options,
    )

    syllable_ixs = sorted(typical_trajectories.keys())
    titles = [f"Syllable{s}" for s in syllable_ixs]
    Xs = np.stack([typical_trajectories[s] for s in syllable_ixs])

    if Xs.shape[-1] == 3:
        projection_planes = [
            "".join(sorted(plane.lower())) for plane in projection_planes
        ]
        assert set(projection_planes) <= set(["xy", "yz", "xz"]), fill(
            "`projection_planes` must be a subset of `['xy','yz','xz']`"
        )
        all_Xs = [
            Xs[
                ...,
                np.array({"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[plane]),
            ]
            for plane in projection_planes
        ]
        suffixes = ["." + plane for plane in projection_planes]
    else:
        all_Xs = [Xs * np.array([1, -1])]  # flip y-axis
        suffixes = [""]

    for Xs_2D, suffix in zip(all_Xs, suffixes):
        lims = get_limits(Xs_2D, pctl=0, **padding)

        # individual plots
        if save_individually:
            desc = "Generating trajectory plots"
            for title, X in tqdm.tqdm(
                zip(titles, Xs_2D), desc=desc, total=len(titles), ncols=72
            ):
                fig, ax, rasters = plot_trajectories(
                    [title],
                    X[None],
                    lims,
                    edges=edges,
                    return_rasters=(save_gifs or save_mp4s),
                    **plot_options,
                )

                plt.savefig(os.path.join(output_dir, f"{title}{suffix}.pdf"))
                plt.close(fig=fig)

                if save_gifs:
                    frame_duration = (pre + post) / len(rasters) / fps
                    path = os.path.join(output_dir, f"{title}{suffix}.gif")
                    save_gif(rasters, path, duration=frame_duration)
                if save_mp4s:
                    use_fps = len(rasters) / (pre + post) * fps
                    path = os.path.join(output_dir, f"{title}{suffix}.mp4")
                    write_video_clip(rasters, path, fps=use_fps)

        # grid plot
        fig, ax, rasters = plot_trajectories(
            titles,
            Xs_2D,
            lims,
            edges=edges,
            return_rasters=(save_gifs or save_mp4s),
            **plot_options,
        )

        plt.savefig(os.path.join(output_dir, f"all_trajectories{suffix}.pdf"))
        plt.show()

        if save_gifs:
            frame_duration = (pre + post) / len(rasters) / fps
            path = os.path.join(output_dir, f"all_trajectories{suffix}.gif")
            save_gif(rasters, path, duration=frame_duration)
        if save_mp4s:
            use_fps = len(rasters) / (pre + post) * fps
            path = os.path.join(output_dir, f"all_trajectories{suffix}.mp4")
            write_video_clip(rasters, path, fps=use_fps)

    if interactive and Xs.shape[-1] == 3:
        plot_trajectories_3D(Xs, titles, edges, output_dir, **plot_options)


def overlay_keypoints_on_image(
    image,
    coordinates,
    edges=[],
    keypoint_colormap="autumn",
    node_size=2,
    line_width=1,
    copy=False,
    opacity=1.0,
):
    """Overlay keypoints on an image.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to overlay keypoints on.

    coordinates: ndarray of shape (num_keypoints, 2)
        Array of keypoint coordinates.

    edges: list of tuples, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of indexes.

    keypoint_colormap: str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints.

    node_size: int, default=10
        Size of the keypoints.

    line_width: int, default=2
        Width of the skeleton lines.

    copy: bool, default=False
        Whether to copy the image before overlaying keypoints.

    opacity: float, default=1.0
        Opacity of the overlay graphics (0.0-1.0).

    Returns
    -------
    image: ndarray of shape (height, width, 3)
        Image with keypoints overlayed.
    """
    if copy or opacity < 1.0:
        canvas = image.copy()
    else:
        canvas = image

    # get colors from matplotlib and convert to 0-255 range for opencv
    colors = plt.colormaps[keypoint_colormap](
        np.linspace(0, 1, coordinates.shape[0])
    )
    colors = [tuple([int(c) for c in cs[:3] * 255]) for cs in colors]

    # overlay skeleton
    for i, j in edges:
        if np.isnan(coordinates[i, 0]) or np.isnan(coordinates[j, 0]):
            continue
        pos1 = (int(coordinates[i, 0]), int(coordinates[i, 1]))
        pos2 = (int(coordinates[j, 0]), int(coordinates[j, 1]))
        canvas = cv2.line(
            canvas, pos1, pos2, colors[i], line_width, cv2.LINE_AA
        )

    # overlay keypoints
    for i, (x, y) in enumerate(coordinates):
        if np.isnan(x) or np.isnan(y):
            continue
        pos = (int(x), int(y))
        canvas = cv2.circle(
            canvas, pos, node_size, colors[i], -1, lineType=cv2.LINE_AA
        )

    if opacity < 1.0:
        image = cv2.addWeighted(image, 1 - opacity, canvas, opacity, 0)
    return image


def overlay_trajectory_on_video(
    frames,
    trajectory,
    smoothing_kernel=1,
    highlight=None,
    min_opacity=0.2,
    max_opacity=1,
    num_ghosts=5,
    interval=2,
    plot_options={},
    edges=[],
):
    """
    Overlay a trajectory of keypoints on a video.
    """
    if smoothing_kernel > 0:
        trajectory = gaussian_filter1d(trajectory, smoothing_kernel, axis=0)

    opacities = np.repeat(
        np.linspace(max_opacity, min_opacity, num_ghosts + 1), interval
    )
    for i in np.arange(0, trajectory.shape[0], interval):
        for j, opacity in enumerate(opacities):
            if i + j < frames.shape[0]:
                plot_options["opacity"] = opacity
                if highlight is not None:
                    start, end, highlight_factor = highlight
                    if i + j < start or i + j > end:
                        plot_options["opacity"] *= highlight_factor
                frames[i + j] = overlay_keypoints_on_image(
                    frames[i + j], trajectory[i], edges=edges, **plot_options
                )
    return frames


def overlay_keypoints_on_video(
    video_path,
    coordinates,
    skeleton=[],
    bodyparts=None,
    use_bodyparts=None,
    output_path=None,
    show_frame_numbers=True,
    text_color=(255, 255, 255),
    crop_size=None,
    frames=None,
    quality=7,
    centroid_smoothing_filter=10,
    plot_options={},
):
    """Overlay keypoints on a video.

    Parameters
    ----------
    video_path: str
        Path to a video file.

    coordinates: ndarray of shape (num_frames, num_keypoints, 2)
        Array of keypoint coordinates.

    skeleton: list of tuples, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of bodypart names or a pair of indexes.

    bodyparts: list of str, default=None
        List of bodypart names in `coordinates`. Required if
        `skeleton` is defined using bodypart names.

    use_bodyparts: list of str, default=None
        Subset of bodyparts to plot. If None, all bodyparts are plotted.

    output_path: str, default=None
        Path to save the video. If None, the video is saved to
        `video_path` with the suffix `_keypoints`.

    show_frame_numbers: bool, default=True
        Whether to overlay the frame number in the video.

    text_color: tuple of int, default=(255,255,255)
        Color for the frame number overlay.

    crop_size: int, default=None
        Size of the crop around the keypoints to overlay on the video.
        If None, the entire video is used.

    frames: iterable of int, default=None
        Frames to overlay keypoints on. If None, all frames are used.

    quality: int, default=7
        Quality of the output video.

    centroid_smoothing_filter: int, default=10
        Amount of smoothing to determine cropping centroid.

    plot_options: dict, default={}
        Additional keyword arguments to pass to
        :py:func:`keypoint_moseq.viz.overlay_keypoints`.
    """
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_keypoints.mp4"

    if bodyparts is not None:
        if use_bodyparts is not None:
            coordinates = reindex_by_bodyparts(
                coordinates, bodyparts, use_bodyparts
            )
        else:
            use_bodyparts = bodyparts
        edges = get_edges(use_bodyparts, skeleton)
    else:
        edges = skeleton

    if crop_size is not None:
        outliers = np.any(np.isnan(coordinates), axis=2)
        interpolated_coordinates = interpolate_keypoints(coordinates, outliers)
        crop_centroid = np.nanmedian(interpolated_coordinates, axis=1)
        crop_centroid = gaussian_filter1d(
            crop_centroid, centroid_smoothing_filter, axis=0
        )

    with imageio.get_reader(video_path) as reader:
        fps = reader.get_meta_data()["fps"]
        if frames is None:
            frames = np.arange(reader.count_frames())

        with imageio.get_writer(
            output_path, pixelformat="yuv420p", fps=fps, quality=quality
        ) as writer:
            for frame in tqdm.tqdm(frames, ncols=72):
                image = reader.get_data(frame)

                image = overlay_keypoints_on_image(
                    image, coordinates[frame], edges=edges, **plot_options
                )

                if crop_size is not None:
                    image = crop_image(image, crop_centroid[frame], crop_size)

                if show_frame_numbers:
                    image = cv2.putText(
                        image,
                        f"Frame {frame}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        1,
                        cv2.LINE_AA,
                    )

                writer.append_data(image)


def matplotlib_colormap_to_plotly(cmap):
    """
    Convert a matplotlib colormap to a plotly colormap.

    Parameters
    ----------
    cmap: str
        Name of a matplotlib colormap.

    Returns
    -------
    pl_colorscale: list
        Plotly colormap.
    """
    cmap = plt.colormaps[cmap]
    pl_entries = 255
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = (np.array(cmap(k * h)[:3]) * 255).astype(np.uint8)
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])
    return pl_colorscale


def add_3D_pose_to_plotly_fig(
    fig,
    coords,
    edges,
    keypoint_colormap="autumn",
    node_size=6.0,
    linewidth=3.0,
    visible=True,
    opacity=1,
):
    """
    Add a 3D pose to a plotly figure.

    Parameters
    ----------
    fig: plotly figure
        Figure to which the pose should be added.

    coords: ndarray (N,3)
        3D coordinates of the pose.

    edges: list of index pairs
        Skeleton edges

    keypoint_colormap: str, default='autumn'
        Colormap to use for coloring keypoints.

    node_size: float, default=6.0
        Size of keypoints.

    linewidth: float, default=3.0
        Width of skeleton edges.

    visibility: bool, default=True
        Initial visibility state of the nodes and edges

    opacity: float, default=1
        Opacity of the nodes and edges (0-1)
    """
    marker = {
        "size": node_size,
        "color": np.linspace(0, 1, len(coords)),
        "colorscale": matplotlib_colormap_to_plotly(keypoint_colormap),
        "line": dict(color="black", width=0.5),
        "opacity": opacity,
    }

    line = {"width": linewidth, "color": f"rgba(0,0,0,{opacity})"}

    fig.add_trace(
        plotly.graph_objs.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            visible=visible,
            marker=marker,
        )
    )

    for e in edges:
        fig.add_trace(
            plotly.graph_objs.Scatter3d(
                x=coords[e, 0],
                y=coords[e, 1],
                z=coords[e, 2],
                mode="lines",
                visible=visible,
                line=line,
            )
        )


def plot_pcs_3D(
    ymean,
    ypcs,
    edges,
    keypoint_colormap,
    savefig,
    project_dir=None,
    node_size=6,
    linewidth=2,
    height=400,
    mean_pose_opacity=0.2,
):
    """
    Visualize the components of a fitted PCA model based on 3D components.

    For each PC, a subplot shows the mean pose (semi-transparent) along
    with a perturbation of the mean pose in the direction of the PC.

    Parameters
    ----------
    ymean : ndarray (num_bodyparts, 3)
        Mean pose.

    ypcs : ndarray (num_pcs, num_bodyparts, 3)
        Perturbations of the mean pose in the direction of each PC.

    edges : list of index pairs
        Skeleton edges.

    keypoint_colormap : str
        Name of a matplotlib colormap to use for coloring the keypoints.

    savefig : bool
        Whether to save the figure to a file. If true, the figure is
        saved to `{project_dir}/pcs.html`

    project_dir : str, default=None
        Path to the project directory. Required if `savefig` is True.

    node_size : float, default=30.0
        Size of the keypoints in the figure.

    linewidth: float, default=2.0
        Width of edges in skeleton

    height : int, default=400
        Height of the figure in pixels.

    mean_pose_opacity: float, default=0.4
        Opacity of the mean pose
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

    def visibility_mask(i):
        visible = np.zeros((len(edges) + 1) * (len(ypcs) + 1))
        visible[-(len(edges) + 1) :] = 1
        visible[(len(edges) + 1) * i : (len(edges) + 1) * (i + 1)] = 1
        return visible > 0

    steps = []
    for i, coords in enumerate(ypcs):
        add_3D_pose_to_plotly_fig(
            fig,
            coords,
            edges,
            visible=(i == 0),
            node_size=node_size,
            linewidth=linewidth,
            keypoint_colormap=keypoint_colormap,
        )

        steps.append(
            dict(
                method="update",
                label=f"PC {i+1}",
                args=[{"visible": visibility_mask(i)}],
            )
        )

    add_3D_pose_to_plotly_fig(
        fig,
        ymean,
        edges,
        opacity=mean_pose_opacity,
        node_size=node_size,
        linewidth=linewidth,
        keypoint_colormap=keypoint_colormap,
    )

    fig.update_layout(
        height=height,
        showlegend=False,
        sliders=[dict(steps=steps)],
        scene=dict(
            xaxis=dict(showgrid=False, showbackground=False),
            yaxis=dict(showgrid=False, showbackground=False),
            zaxis=dict(showgrid=False, showline=True, linecolor="black"),
            bgcolor="white",
            aspectmode="data",
        ),
        margin=dict(l=20, r=20, b=0, t=0, pad=10),
    )

    if savefig:
        assert project_dir is not None, fill(
            "The `savefig` option requires a `project_dir`"
        )
        save_path = os.path.join(project_dir, f"pcs.html")
        fig.write_html(save_path)
        print(f"Saved interactive plot to {save_path}")

    fig.show()


def plot_trajectories_3D(
    Xs,
    titles,
    edges,
    output_dir,
    keypoint_colormap="autumn",
    node_size=8,
    linewidth=3,
    height=500,
    skiprate=1,
):
    """
    Visualize a set of 3D trajectories.

    Parameters
    ----------
    Xs : list of ndarrays (num_syllables, num_frames, num_bodyparts, 3)
        Trajectories to visualize.

    titles : list of str
        Title for each trajectory.

    edges : list of index pairs
        Skeleton edges.

    output_dir : str
        Path to save the interactive plot.

    keypoint_colormap : str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints.

    node_size : float, default=8.0
        Size of the keypoints in the figure.

    linewidth: float, default=3.0
        Width of edges in skeleton

    height : int, default=500
        Height of the figure in pixels.

    skiprate : int, default=1
        Plot every `skiprate` frames.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    Xs = Xs[:, ::skiprate]

    def visibility_mask(i):
        n = (len(edges) + 1) * len(Xs[1])
        visible = np.zeros(n * len(Xs))
        visible[n * i : n * (i + 1)] = 1
        return visible > 0

    steps = []
    for i, X in enumerate(Xs):
        opacities = np.linspace(0.3, 1, len(X) + 1)[1:] ** 2
        for coords, opacity in zip(X, opacities):
            add_3D_pose_to_plotly_fig(
                fig,
                coords,
                edges,
                visible=(i == 0),
                node_size=node_size,
                linewidth=linewidth,
                keypoint_colormap=keypoint_colormap,
                opacity=opacity,
            )

        steps.append(
            dict(
                method="update",
                label=titles[i],
                args=[{"visible": visibility_mask(i)}],
            )
        )

    fig.update_layout(
        height=height,
        showlegend=False,
        sliders=[dict(steps=steps)],
        scene=dict(
            xaxis=dict(showgrid=False, showbackground=False),
            yaxis=dict(showgrid=False, showbackground=False),
            zaxis=dict(showgrid=False, showline=True, linecolor="black"),
            bgcolor="white",
            aspectmode="data",
        ),
        margin=dict(l=20, r=20, b=0, t=0, pad=10),
    )

    if output_dir is not None:
        save_path = os.path.join(output_dir, f"all_trajectories.html")
        fig.write_html(save_path)
        print(f"Saved interactive trajectories plot to {save_path}")

    fig.show()


def plot_similarity_dendrogram(
    coordinates,
    results,
    project_dir=None,
    model_name=None,
    save_path=None,
    metric="cosine",
    pre=5,
    post=15,
    min_frequency=0.005,
    min_duration=3,
    bodyparts=None,
    use_bodyparts=None,
    density_sample=False,
    sampling_options={},
    figsize=(6, 3),
    **kwargs,
):
    """Plot a dendrogram showing the similarity between syllable trajectories.

    The dendrogram is saved to `{save_path}` if it is provided, or
    else to `{project_dir}/{model_name}/similarity_dendrogram.pdf`. Plot-
    related parameters are described below. For the remaining parameters
    see (:py:func:`keypoint_moseq.util.get_typical_trajectories`)

    Parameters
    ----------
    coordinates: dict
        Dictionary mapping recording names to keypoint coordinates as
        ndarrays of shape (n_frames, n_bodyparts, [2 or 3]).

    results: dict
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.extract_results`).

    project_dir: str, default=None
        Project directory. Required to save figure if `save_path` is None.

    model_name: str, default=None
        Model name. Required to save figure if `save_path` is None.

    save_path: str, default=None
        Path to save the dendrogram plot (do not include an extension).
        If None, the plot will be saved  to
        `{project_dir}/{name}/similarity_dendrogram.[pdf/png]`.

    metric: str, default='cosine'
        Distance metric to use. See :py:func:`scipy.spatial.pdist` for options.

    figsize: tuple of float, default=(10,5)
        Size of the dendrogram plot.
    """
    save_path = _get_path(
        project_dir, model_name, save_path, "similarity_dendrogram"
    )

    distances, syllable_ixs = syllable_similarity(
        coordinates,
        results,
        metric,
        pre,
        post,
        min_frequency,
        min_duration,
        bodyparts,
        use_bodyparts,
        density_sample,
        sampling_options,
    )

    Z = linkage(squareform(distances), "complete")

    fig, ax = plt.subplots(1, 1)
    labels = [f"Syllable {s}" for s in syllable_ixs]
    dendrogram(Z, labels=labels, leaf_font_size=10, ax=ax, leaf_rotation=90)

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("lightgray")
    ax.set_title("Syllable similarity")
    fig.set_size_inches(figsize)

    print(f"Saving dendrogram plot to {save_path}")
    for ext in ["pdf", "png"]:
        plt.savefig(save_path + "." + ext)
