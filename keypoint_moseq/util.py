import os
import glob
import tabulate
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import fill
import jax, jax.numpy as jnp
from scipy.ndimage import median_filter, convolve1d, gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from jax_moseq.models.keypoint_slds import inverse_rigid_transform
from jax_moseq.utils import get_frequencies, batch
from vidio.read import OpenCVReader

na = jnp.newaxis


def np_io(fn):
    """Converts a function involving jax arrays to one that inputs and outputs
    numpy arrays."""
    return lambda *args, **kwargs: jax.device_get(
        fn(*jax.device_put(args), **jax.device_put(kwargs))
    )


def print_dims_to_explain_variance(pca, f):
    """Print the number of principal components requred to explain a given
    fraction of  variance.

    Parameters
    ----------
    pca: sklearn.decomposition._pca.PCA, A fit PCA model
    f: float, Target variance fraction
    """
    cs = np.cumsum(pca.explained_variance_ratio_)
    if cs[-1] < f:
        print(f"All components together only explain {cs[-1]*100}% of variance.")
    else:
        print(
            f">={f*100}% of variance exlained by {(cs>f).nonzero()[0].min()+1} components."
        )


def list_files_with_exts(filepath_pattern, ext_list, recursive=True):
    """This function lists all the files matching a pattern and with a an
    extension in a list of extensions.

    Parameters
    ----------
    filepath_pattern : str or list
        A filepath pattern or a list thereof. Filepath patterns can be be a
        single file, a directory, or a path with wildcards (e.g.,
        '/path/to/dir/prefix*').

    ext_list : list of str
        A list of file extensions to search for.

    recursive : bool, default=True
        Whether to search for files recursively.

    Returns
    -------
    list
        A list of file paths.
    """
    if isinstance(filepath_pattern, list):
        matches = []
        for fp in filepath_pattern:
            matches += list_files_with_exts(fp, ext_list, recursive=recursive)
        return sorted(set(matches))

    else:
        # make sure extensions all start with "." and are lowercase
        ext_list = ["." + ext.strip(".").lower() for ext in ext_list]

        if os.path.isdir(filepath_pattern):
            filepath_pattern = os.path.join(filepath_pattern, "*")

        # find all matches (recursively)
        matches = glob.glob(filepath_pattern)
        if recursive:
            for match in list(matches):
                matches += glob.glob(os.path.join(match, "**"), recursive=True)

        # filter matches by extension
        matches = [
            match for match in matches if os.path.splitext(match)[1].lower() in ext_list
        ]
        return matches


def find_matching_videos(
    keys,
    video_dir,
    as_dict=False,
    recursive=True,
    recording_name_suffix="",
    video_extension=None,
):
    """
    Find video files for a set of recording names. The filename of each video
    is assumed to be a prefix within the recording name, i.e. the recording
    name has the form `{video_name}{more_text}`. If more than one video matches
    a recording name, the longest match will be used. For example given the
    following video directory::

        video_dir
        ├─ videoname1.avi
        └─ videoname2.avi

    the videos would be matched to recording names as follows::

        >>> keys = ['videoname1blahblah','videoname2yadayada']
        >>> find_matching_videos(keys, video_dir, as_dict=True)

        {'videoname1blahblah': 'video_dir/videoname1.avi',
         'videoname2blahblah': 'video_dir/videoname2.avi'}

    A suffix can also be specified, in which case the recording name is assumed
    to have the form `{video_name}{suffix}{more_text}`.

    Parameters
    -------
    keys: iterable
        Recording names (as strings)

    video_dir: str
        Path to the video directory.

    video_extension: str, default=None
        Extension of the video files. If None, videos are assumed to have the
        one of the following extensions: "mp4", "avi", "mov"

    recursive: bool, default=True
        If True, search recursively for videos in subdirectories of
        `video_dir`.

    as_dict: bool, default=False
        Determines whether to return a dict mapping recording names to video
        paths, or a list of paths in the same order as `keys`.

    recording_name_suffix: str, default=None
        Suffix to append to the video name when searching for a match.

    Returns
    -------
    video_paths: list or dict (depending on `as_dict`)
    """

    if video_extension is None:
        extensions = [".mp4", ".avi", ".mov"]
    else:
        if video_extension[0] != ".":
            video_extension = "." + video_extension
        extensions = [video_extension]

    videos = list_files_with_exts(video_dir, extensions, recursive=recursive)
    videos_to_paths = {os.path.splitext(os.path.basename(f))[0]: f for f in videos}

    video_paths = []
    for key in keys:
        matches = [
            v
            for v in videos_to_paths
            if os.path.basename(key).startswith(v + recording_name_suffix)
        ]
        assert len(matches) > 0, fill(f"No matching videos found for {key}")

        longest_match = sorted(matches, key=lambda v: len(v))[-1]
        video_paths.append(videos_to_paths[longest_match])

    if as_dict:
        return dict(zip(keys, video_paths))
    else:
        return video_paths


def pad_along_axis(arr, pad_widths, axis=0, value=0):
    """Pad an array along a single axis.

    Parameters
    -------
    arr: ndarray, Array to be padded
    pad_widths: tuple (int,int), Amount of padding on either end
    axis: int, Axis along which to add padding
    value: float, Value of padded array elements

    Returns
    _______
    padded_arr: ndarray
    """
    pad_widths_full = [(0, 0)] * len(arr.shape)
    pad_widths_full[axis] = pad_widths
    padded_arr = np.pad(arr, pad_widths_full, constant_values=value)
    return padded_arr


def filter_angle(angles, size=9, axis=0, method="median"):
    """Perform median filtering on time-series of angles by transforming to a
    (cos,sin) representation, filtering in R^2, and then transforming back into
    angle space.

    Parameters
    -------
    angles: ndarray
        Array of angles (in radians)

    size: int, default=9
        Size of the filtering kernel

    axis: int, default=0
        Axis along which to filter

    method: str, default='median'
        Method for filtering. Options are 'median' and 'gaussian'

    Returns
    -------
    filtered_angles: ndarray
    """
    if method == "median":
        kernel = np.where(np.arange(len(angles.shape)) == axis, size, 1)
        filter = lambda x: median_filter(x, kernel)
    elif method == "gaussian":
        filter = lambda x: gaussian_filter1d(x, size, axis=axis)
    return np.arctan2(filter(np.sin(angles)), filter(np.cos(angles)))


def get_centroids_headings(
    coordinates,
    anterior_idxs,
    posterior_idxs,
    bodyparts=None,
    use_bodyparts=None,
    **kwargs,
):
    """Compute centroids and headings from keypoint coordinates.

    Parameters
    -------
    coordinates: dict
        Dictionary mapping recording names to keypoint coordinates as
        ndarrays of shape (n_frames, n_bodyparts, [2 or 3]).

    anterior_idxs: array-like of int
        Indices of anterior bodyparts (after reindexing by `use_bodyparts`
        when the latter is specified).

    posterior_idxs: array-like of int
        Indices of anterior bodyparts (after reindexing by `use_bodyparts`
        when the latter is specified).

    bodyparts: list of str, default=None
        List of bodypart names in `coordinates`. Used to reindex coordinates
        when `use_bodyparts` is specified.

    use_bodyparts: list of str, default=None
        Ordered list of bodyparts used to reindex `coordinates`.

    Returns
    -------
    centroids: dict
        Dictionary mapping recording names to centroid coordinates as ndarrays
        of shape (n_frames, [2 or 3]).

    headings: dict
        Dictionary mapping recording names to heading angles (in radians) as 1d
        arrays of shape (n_frames,).
    """
    if bodyparts is not None and use_bodyparts is not None:
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)

    centroids, headings = {}, {}
    for key, coords in coordinates.items():
        coords = interpolate_keypoints(coords, np.isnan(coords).any(-1))
        centroids[key] = np.median(coords, axis=1)
        anterior_loc = coords[:, posterior_idxs].mean(1)
        posterior_loc = coords[:, anterior_idxs].mean(1)
        heading_vec = anterior_loc - posterior_loc
        headings[key] = np.arctan2(*heading_vec.T[::-1]) + np.pi

    return centroids, headings


def filter_centroids_headings(centroids, headings, filter_size=9):
    """Perform median filtering on centroids and headings.

    Parameters
    -------
    centroids: dict
        Centroids stored as a dictionary mapping recording names to ndarrays,
        of shape (n_frames, [2 or 3]).

    headings: dict
        Dictionary mapping recording names to heading angles (in radians) as 1d
        arrays of shape (n_frames,).

    filter_size: int, default=9
        Kernel size for median filtering

    Returns
    -------
    filtered_centroids: dict
    filtered_headings: dict
    """
    centroids = {k: median_filter(v, (filter_size, 1)) for k, v in centroids.items()}
    headings = {k: filter_angle(v, size=filter_size) for k, v in headings.items()}
    return centroids, headings


def get_syllable_instances(
    stateseqs,
    min_duration=3,
    pre=30,
    post=60,
    min_frequency=0,
    min_instances=0,
):
    """Map each syllable to a list of instances when it occured. Only include
    instances that meet the criteria specified by `pre`, `post`, and
    `min_duration`. Only include syllables that meet the criteria specified by
    `min_frequency` and `min_instances`.

    Parameters
    -------
    stateseqs: dict {str : 1d array}
        Dictionary mapping names to syllable sequences

    min_duration: int, default=3
        Mininum duration for inclusion of a syllable instance

    pre: int, default=30
        Syllable instances that start before this location in the state
        sequence will be excluded

    post: int, default=60
        Syllable instances that end after this location in the state sequence
        will be excluded

    min_frequency: int, default=0
        Minimum allowed frequency (across all state sequences) for inclusion of
        a syllable

    min_instances: int, default=0
        Minimum number of instances (across all state sequences) for inclusion
        of a syllable

    Returns
    -------
    syllable_instances: dict
        Dictionary mapping each syllable to a list of instances. Each instance
        is a tuple (name,start,end) representing subsequence
        `stateseqs[name][start:end]`.
    """
    num_syllables = int(max(map(max, stateseqs.values())) + 1)
    syllable_instances = [[] for syllable in range(num_syllables)]

    for key, stateseq in stateseqs.items():
        transitions = np.nonzero(stateseq[1:] != stateseq[:-1])[0] + 1
        starts = np.insert(transitions, 0, 0)
        ends = np.append(transitions, len(stateseq))
        for s, e, syllable in zip(starts, ends, stateseq[starts]):
            if e - s >= min_duration and s >= pre and s < len(stateseq) - post:
                syllable_instances[syllable].append((key, s, e))

    frequencies_filter = get_frequencies(stateseqs) >= min_frequency
    counts_filter = np.array(list(map(len, syllable_instances))) >= min_instances
    use_syllables = np.all([frequencies_filter, counts_filter], axis=0).nonzero()[0]
    return {syllable: syllable_instances[syllable] for syllable in use_syllables}


def get_edges(use_bodyparts, skeleton):
    """Represent the skeleton as a list of index-pairs.

    Parameters
    -------
    use_bodyparts: list
        Bodypart names

    skeleton: list
        Pairs of bodypart names as tuples (bodypart1,bodypart2)

    Returns
    -------
    edges: list
        Pairs of indexes representing the enties of `skeleton`
    """
    edges = []
    if len(skeleton) > 0:
        if isinstance(skeleton[0][0], int):
            edges = skeleton
        else:
            assert use_bodyparts is not None, fill(
                "If skeleton edges are specified using bodypart names, "
                "`use_bodyparts` must be specified"
            )

            for bp1, bp2 in skeleton:
                if bp1 in use_bodyparts and bp2 in use_bodyparts:
                    edges.append([use_bodyparts.index(bp1), use_bodyparts.index(bp2)])
    return edges


def reindex_by_bodyparts(data, bodyparts, use_bodyparts, axis=1):
    """Use an ordered list of bodyparts to reindex keypoint coordinates.

    Parameters
    -------
    data: dict or ndarray
        A single array of keypoint coordinates or a dict mapping from names to
        arrays of keypoint coordinates

    bodyparts: list
        Label for each keypoint represented in `data`

    use_bodyparts: list
        Ordered subset of keypoint labels

    axis: int, default=1
        The axis in `data` that represents keypoints. It is required that
        `data.shape[axis]==len(bodyparts)`.

    Returns
    -------
    reindexed_data: ndarray or dict
        Keypoint coordinates in the same form as `data` with reindexing
        applied.
    """
    ix = np.array([bodyparts.index(bp) for bp in use_bodyparts])
    if isinstance(data, np.ndarray):
        return np.take(data, ix, axis)
    else:
        return {k: np.take(v, ix, axis) for k, v in data.items()}


def get_instance_trajectories(
    syllable_instances,
    coordinates,
    pre=0,
    post=None,
    centroids=None,
    headings=None,
    filter_size=9,
):
    """Extract keypoint trajectories for a collection of syllable instances.

    If centroids and headings are provided, each trajectory is transformed into
    the ego-centric reference frame from the moment of syllable onset. When
    `post` is not None, trajectories will all terminate a fixed number of
    frames after syllable onset.

    Parameters
    -------
    syllable_instances: list
        List of syllable instances, where each instance is a tuple of the form
        (name,start,end)

    coordinates: dict
        Dictionary mapping names to coordinates, formatted as ndarrays with
        shape (num_frames, num_keypoints, d)

    pre: int, default=0
        Number of frames to include before syllable onset

    post: int, defualt=None
        Determines the length of the trajectory. When `post=None`, the
        trajectory terminates at the end of the syllable instance. Otherwise
        the trajectory terminates at a fixed number of frames after syllable
        (where the number is determined by `post`).

    centroids: dict, default=None
        Dictionary with the same keys as `coordinates` mapping each name to an
        ndarray with shape (num_frames, d)

    headings: dict, default=None
        Dictionary with the same keys as `coordinates` mapping each name to a
        1d array of heading angles in radians

    filter_size: int, default=9
        Size of median filter applied to `centroids` and `headings`

    Returns
    -------
    trajectories: list
        List or array of trajectories (a list is used when `post=None`,
        otherwise an array). Each trajectory is an array of shape
        (n_frames, n_bodyparts, [2 or 3]).
    """
    if centroids is not None and headings is not None:
        centroids, headings = filter_centroids_headings(
            centroids, headings, filter_size=filter_size
        )

    if post is None:
        trajectories = [
            coordinates[key][s - pre : e] for key, s, e in syllable_instances
        ]
        if centroids is not None and headings is not None:
            trajectories = [
                np_io(inverse_rigid_transform)(x, centroids[key][s], headings[key][s])
                for x, (key, s, e) in zip(trajectories, syllable_instances)
            ]
    else:
        trajectories = np.array(
            [coordinates[key][s - pre : s + post] for key, s, e in syllable_instances]
        )
        if centroids is not None and headings is not None:
            c = np.array([centroids[key][s] for key, s, e in syllable_instances])[
                :, None
            ]
            h = np.array([headings[key][s] for key, s, e in syllable_instances])[
                :, None
            ]
            trajectories = np_io(inverse_rigid_transform)(trajectories, c, h)

    return trajectories


def sample_instances(
    syllable_instances,
    num_samples,
    mode="random",
    pca_samples=50000,
    pca_dim=4,
    n_neighbors=50,
    coordinates=None,
    pre=5,
    post=15,
    centroids=None,
    headings=None,
    filter_size=9,
):
    """Sample a fixed number of instances for each syllable.

    Parameters
    ----------
    syllable_instances: dict
        Mapping from each syllable to a list of instances, where each instance
        is a tuple of the form (name,start,end)

    num_samples: int
        Number of samples return for each syllable

    mode: str, {'random', 'density'}, default='random'
        Sampling method to use. Options are:

        - 'random': Instances are chosen randomly (without replacement)
        - 'density': For each syllable, a syllable-specific density function is
          computed in trajectory space and compared to the overall density
          across all syllables. An exemplar instance that maximizes this ratio
          is chosen for each syllable, and its nearest neighbors are randomly
          sampled.

    pca_samples: int, default=50000
        Number of trajectories to sample when fitting a PCA model for density
        estimation (used when `mode='density'`)

    pca_dim: int, default=4
        Number of principal components to use for density estimation (used when
        `mode='density'`)

    n_neighbors: int, defualt=50
        Number of neighbors to use for density estimation and for sampling the
        neighbors of the examplar syllable instance (used when
        `mode='density'`)

    coordinates, pre, pos, centroids, heading, filter_size
        Passed to :py:func:`keypoint_moseq.util.get_instance_trajectories`

    Returns
    -------
    sampled_instances: dict
        Dictionary in the same format as `syllable_instances` mapping each
        syllable to a list of sampled instances.
    """
    assert mode in ["random", "density"]
    assert all([len(v) >= num_samples for v in syllable_instances.values()])
    assert n_neighbors >= num_samples

    if mode == "random":
        sampled_instances = {
            syllable: [
                instances[i]
                for i in np.random.choice(len(instances), num_samples, replace=False)
            ]
            for syllable, instances in syllable_instances.items()
        }
        return sampled_instances

    elif mode == "density":
        assert not (coordinates is None or headings is None or centroids is None), fill(
            "`coordinates`, `headings` and `centroids` are required when "
            '`mode == "density"`'
        )

        for key in coordinates.keys():
            outliers = np.isnan(coordinates[key]).any(-1)
            coordinates[key] = interpolate_keypoints(coordinates[key], outliers)

        trajectories = {
            syllable: get_instance_trajectories(
                instances,
                coordinates,
                pre=pre,
                post=post,
                centroids=centroids,
                headings=headings,
                filter_size=filter_size,
            )
            for syllable, instances in syllable_instances.items()
        }
        X = np.vstack(list(trajectories.values()))

        if X.shape[0] > pca_samples:
            X = X[np.random.choice(X.shape[0], pca_samples, replace=False)]

        pca = PCA(n_components=pca_dim).fit(X.reshape(X.shape[0], -1))
        Xpca = pca.transform(X.reshape(X.shape[0], -1))
        all_nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(Xpca)

        sampled_instances = {}

        for syllable, X in trajectories.items():
            Xpca = pca.transform(X.reshape(X.shape[0], -1))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(Xpca)
            distances, indices = nbrs.kneighbors(Xpca)
            local_density = 1 / distances.mean(1)

            distances, _ = all_nbrs.kneighbors(Xpca)
            global_density = 1 / distances.mean(1)
            exemplar = np.argmax(local_density / global_density)
            samples = np.random.choice(indices[exemplar], num_samples, replace=False)
            sampled_instances[syllable] = [
                syllable_instances[syllable][i] for i in samples
            ]

        return sampled_instances

    else:
        raise ValueError("Invalid mode: {}".format(mode))


def interpolate_along_axis(x, xp, fp, axis=0):
    """Linearly interpolate along a given axis.

    Parameters
    ----------
    x: 1D array
        The x-coordinates of the interpolated values
    xp: 1D array
        The x-coordinates of the data points
    fp: ndarray
        The y-coordinates of the data points. fp.shape[axis] must
        be equal to the length of xp.

    Returns
    -------
    x_interp: ndarray
        The interpolated values, with the same shape as fp except along the
        interpolation axis.
    """
    assert len(xp.shape) == len(x.shape) == 1
    assert fp.shape[axis] == len(xp)
    assert len(xp) > 0, "xp must be non-empty; cannot interpolate without datapoints"

    fp = np.moveaxis(fp, axis, 0)
    shape = fp.shape[1:]
    fp = fp.reshape(fp.shape[0], -1)

    x_interp = np.zeros((len(x), fp.shape[1]))
    for i in range(fp.shape[1]):
        x_interp[:, i] = np.interp(x, xp, fp[:, i])
    x_interp = x_interp.reshape(len(x), *shape)
    x_interp = np.moveaxis(x_interp, 0, axis)
    return x_interp


def interpolate_keypoints(coordinates, outliers):
    """Use linear interpolation to impute the coordinates of outliers.

    Parameters
    ----------
    coordinates : ndarray of shape (num_frames, num_keypoints, dim)
        Keypoint observations.
    outliers : ndarray of shape (num_frames, num_keypoints)
        Binary indicator whose true entries are outlier points.

    Returns
    -------
    interpolated_coordinates : ndarray with same shape as `coordinates`
        Keypoint observations with outliers imputed.
    """
    interpolated_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[1]):
        xp = np.nonzero(~outliers[:, i])[0]
        if len(xp) > 0:
            interpolated_coordinates[:, i, :] = interpolate_along_axis(
                np.arange(coordinates.shape[0]), xp, coordinates[xp, i, :]
            )
    return interpolated_coordinates


def filtered_derivative(Y_flat, ksize, axis=0):
    """Compute the filtered derivative of a signal along a given axis.

    When `ksize=3`, for example, the filtered derivative is

    .. math::

        \\dot{y_t} = \\frac{1}{3}( x_{t+3}+x_{t+2}+x_{t+1}-x_{t-1}-x_{t-2}-x_{t-3})


    Parameters
    ----------
    Y_flat: ndarray
        The signal to differentiate

    ksize: int
        The size of the filter. Must be odd.

    axis: int, default=0
        The axis along which to differentiate

    Returns
    -------
    dY: ndarray
        The filtered derivative of the signal
    """
    kernel = np.ones(ksize + 1) / (ksize + 1)
    pre = convolve1d(Y_flat, kernel, origin=-(ksize + 1) // 2, axis=axis)
    post = convolve1d(Y_flat, kernel, origin=ksize // 2, axis=axis)
    return post - pre


def permute_cyclic(arr, mask=None, axis=0):
    """Cyclically permute an array along a given axis.

    Parameters
    ----------
    arr: ndarray
        The array to permute

    mask: ndarray, optional
        A boolean mask indicating which elements to permute. If None, all
        elements are permuted.

    axis: int, default=0
        The axis along which to permute

    Returns
    -------
    arr_permuted: ndarray
        The permuted array
    """
    if mask is None:
        mask = np.ones_like(arr)

    arr = np.moveaxis(arr, axis, 0)
    mask = np.moveaxis(mask, axis, 0)

    shape = arr.shape
    arr = arr.reshape(arr.shape[0], -1)
    mask = mask.reshape(mask.shape[0], -1)

    arr_permuted = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        arr_permuted[mask[:, i] > 0, i] = np.roll(
            arr[mask[:, i] > 0, i], np.random.randint(0, mask[:, i].sum())
        )

    arr_permuted = arr_permuted.reshape(shape)
    arr_permuted = np.moveaxis(arr_permuted, 0, axis)
    return arr_permuted


def _print_colored_table(row_labels, col_labels, values):
    try:
        from IPython.display import display

        display_available = True
    except ImportError:
        display_available = False

    title = "Proportion of NaNs"
    df = pd.DataFrame(values, index=row_labels, columns=col_labels)

    if display_available:

        def colorize(val):
            color = plt.get_cmap("Reds")(val * 0.8)
            return f"background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})"

        colored_df = df.style.applymap(colorize).set_caption("Proportion of NaNs")
        display(colored_df)
        return colored_df
    else:
        print(title)
        print(tabulate(df, headers="keys", tablefmt="simple_grid", showindex=True))


def check_nan_proportions(
    coordinates, bodyparts, warning_threshold=0.5, breakdown=False, **kwargs
):
    """Check if any bodyparts have a high proportion of NaNs.

    Parameters
    ----------
    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays of
        shape (n_frames, n_bodyparts, 2)

    bodyparts: list of str
        Name of each bodypart. The order of the names should match the order of
        the bodyparts in `coordinates`.

    warning_threshold: float, default=0.5
        If the proportion of NaNs for a bodypart is greater than
        `warning_threshold`, then a warning is printed.

    breakdown: bool, default=False
        Whether to print a table detailing the proportion of NaNs for each
        bodyparts in each array of `coordinates`.
    """
    if breakdown:
        keys = sorted(coordinates.keys())
        nan_props = [np.isnan(coordinates[k]).any(-1).mean(0) for k in keys]
        _print_colored_table(keys, bodyparts, nan_props)
    else:
        all_coords = np.concatenate(list(coordinates.values()))
        nan_props = np.isnan(all_coords).any(-1).mean(0)
        if np.any(nan_props > warning_threshold):
            bps = [bp for bp, p in zip(bodyparts, nan_props) if p > warning_threshold]
            warnings.warn(
                "\nCoordinates for the following bodyparts are missing (set to NaN) in at least "
                "{}% of frames:\n - {}\n\n".format(
                    warning_threshold * 100, "\n - ".join(bps)
                )
            )
            warnings.warn(
                "This may cause problems during modeling. See "
                "https://keypoint-moseq.readthedocs.io/en/latest/FAQs.html#high-proportion-of-nans"
                " for additional information."
            )


def format_data(
    coordinates,
    confidences=None,
    keys=None,
    seg_length=None,
    bodyparts=None,
    use_bodyparts=None,
    conf_pseudocount=1e-3,
    added_noise_level=0.1,
    **kwargs,
):
    """Format keypoint coordinates and confidences for inference.

    Data are transformed as follows:
        1. Coordinates and confidences are each merged into a single array
           using :py:func:`keypoint_moseq.util.batch`. Each row of the merged
           arrays is a segment from one recording.
        2. The keypoints axis is reindexed according to the order of elements
           in `use_bodyparts` with respect to their initial orer in
           `bodyparts`.
        3. Uniform noise proportional to `added_noise_level` is added to the
           keypoint coordinates to prevent degenerate solutions during fitting.
        4. Keypoint confidences are augmented by `conf_pseudocount`.
        5. Wherever NaNs occur in the coordinates, they are replaced by values
           imputed using linear interpolation, and the corresponding
           confidences are set to `conf_pseudocount`.

    Parameters
    ----------
    coordinates: dict
        Keypoint coordinates for a collection of recordings. Values must be
        numpy arrays of shape (T,K,D) where K is the number of keypoints and
        D={2 or 3}.

    confidences: dict, default=None
        Nonnegative confidence values for the keypoints in `coordinates` as
        numpy arrays of shape (T,K).

    keys: list of str, default=None
        (See :py:func:`keypoint_moseq.util.batch`)

    bodyparts: list, default=None
        Label for each keypoint represented in `coordinates`. Required to
        reindex coordinates and confidences according to `use_bodyparts`.

    use_bodyparts: list, default=None
        Ordered subset of keypoint labels to be used for modeling. If
        `use_bodyparts=None`, then all keypoints are used.

    conf_pseudocount: float, default=1e-3
        Pseudocount used to augment keypoint confidences.

    seg_length: int, default=None
        Length of each segment. If `seg_length=None`, a length is chosen so
        that no time-series are broken into multiple segments. If all
        time-series are shorter than `seg_length`, then  `seg_length` is set to
        the length of the shortest time-series.

    Returns
    -------
    data: dict with the following items

        Y: jax array with shape (n_segs, seg_length, K, D)
            Keypoint coordinates from all recordings broken into fixed-length
            segments.

        conf: jax array with shape (n_segs, seg_length, K)
            Confidences from all recordings broken into fixed-length segments.
            If no input is provided for `confidences`, then
            `data["conf"]=None`.

        mask: jax array with shape (n_segs, seg_length)
            Binary array where 0 indicates areas of padding
            (see :py:func:`keypoint_moseq.util.batch`).

    metadata: tuple (keys, bounds)
        Metadata for the rows of `Y`, `conf` and `mask`, as a tuple with a
        array of recording names and an array of (start,end) times. See
        :py:func:`jax_moseq.utils.batch` for details.
    """
    if keys is None:
        keys = sorted(coordinates.keys())
    else:
        bad_keys = set(keys) - set(coordinates.keys())
        assert len(bad_keys) == 0, fill(f"Keys {bad_keys} not found in coordinates")

    assert len(keys) > 0, "No recordings found"

    num_keypoints = [coordinates[key].shape[-2] for key in keys]
    assert len(set(num_keypoints)) == 1, fill(
        f"All recordings must have the same number of keypoints, but "
        f"found {set(num_keypoints)} keypoints across recordings."
    )

    if bodyparts is not None:
        assert len(bodyparts) == num_keypoints[0], fill(
            f"The number of keypoints in `coordinates` ({num_keypoints[0]}) "
            f"does not match the number of labels in `bodyparts` "
            f"({len(bodyparts)})"
        )

    if any(["/" in key for key in keys]):
        warnings.warn(
            fill(
                'WARNING: Recording names should not contain "/", this will cause '
                "problems with saving/loading hdf5 files."
            )
        )

    if confidences is None:
        confidences = {key: np.ones_like(coordinates[key][..., 0]) for key in keys}

    if bodyparts is not None and use_bodyparts is not None:
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)
        confidences = reindex_by_bodyparts(confidences, bodyparts, use_bodyparts)

    for key in keys:
        outliers = np.isnan(coordinates[key]).any(-1)
        coordinates[key] = interpolate_keypoints(coordinates[key], outliers)
        confidences[key] = np.where(outliers, 0, np.nan_to_num(confidences[key]))

    if seg_length is not None:
        max_recording_length = max([coordinates[key].shape[0] for key in keys])
        seg_length = min(seg_length, max_recording_length)

    Y, mask, metadata = batch(coordinates, seg_length=seg_length, keys=keys)
    Y = Y.astype(float)

    min_segment_length = np.diff(metadata[1], axis=1).min()
    assert min_segment_length >= 4, (
        f"The shortest segment has length  {min_segment_length} which is below the "
        "minimum of 4. Try increasing `seg_length` in the config (e.g. add "
        f"{min_segment_length} to its current value) and also make sure that all your "
        "input recordings are at least 4 frames long."
    )

    conf = batch(confidences, seg_length=seg_length, keys=keys)[0]
    if np.min(conf) < 0:
        conf = np.maximum(conf, 0)
        warnings.warn(
            fill("Negative confidence values are not allowed and will be set to 0.")
        )
    conf = conf + conf_pseudocount

    if added_noise_level > 0:
        Y += np.random.uniform(-added_noise_level, added_noise_level, Y.shape)

    data = jax.device_put({"mask": mask, "Y": Y, "conf": conf})
    return data, metadata


def get_typical_trajectories(
    coordinates,
    results,
    pre=5,
    post=15,
    min_frequency=0.005,
    min_duration=3,
    bodyparts=None,
    use_bodyparts=None,
    density_sample=True,
    sampling_options={"n_neighbors": 50},
):
    """Generate representative keypoint trajectories for each syllable.

    Parameters
    ----------
    coordinates: dict
        Dictionary mapping recording names to keypoint coordinates as ndarrays
        of shape (n_frames, n_bodyparts, 2).

    results: dict
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.extract_results`).

    pre: int, default=5, post: int, default=15
        Defines the temporal window around syllable onset for computing the
        average trajectory. Note that the window is independent of the actual
        duration of the syllable.

    min_frequency: float, default=0.005
        Minimum frequency of a syllable to plotted.

    min_duration: float, default=3
        Minimum duration of a syllable instance to be included in the
        trajectory average.

    bodyparts: list of str, default=None
        List of bodypart names in `coordinates`.

    use_bodyparts: list of str, default=None
        Ordered list of bodyparts to include in each trajectory. If None, all
        bodyparts will be included.

    density_sample : bool, default=True
        Whether to use density sampling when generating trajectories. If True,
        the trajectory is based on the most exemplary syllable instances,
        rather than being average across all instances.

    sampling_options: dict, default={'n_neighbors':50}
        Dictionary of options for sampling syllable instances (see
        :py:func:`keypoint_moseq.util.sample_instances`). Only used when
        `density_sample` is True.

    Returns
    -------
    representative_trajectories: dict
        Dictionary mapping syllable indexes to representative trajectories
        as arrays of shape (pre+pose, n_bodyparts, [2 or 3]).
    """
    if bodyparts is not None and use_bodyparts is not None:
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)

    syllables = {k: v["syllable"] for k, v in results.items()}
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    min_instances = sampling_options["n_neighbors"] if density_sample else 1
    syllable_instances = get_syllable_instances(
        syllables,
        pre=pre,
        post=post,
        min_duration=min_duration,
        min_frequency=min_frequency,
        min_instances=min_instances,
    )

    if len(syllable_instances) == 0:
        raise ValueError(
            fill(
                "No syllables with sufficient instances to generate a trajectory. "
                "This usually occurs when there is not enough inut data or when "
                "all frames have the same syllable label (use "
                "`plot_syllable_frequencies` to check if this is the case)"
            )
        )
        return

    if density_sample:
        sampling_options["mode"] = "density"
        sampled_instances = sample_instances(
            syllable_instances,
            sampling_options["n_neighbors"],
            coordinates=coordinates,
            centroids=centroids,
            headings=headings,
            **sampling_options,
        )
    else:
        sampled_instances = syllable_instances

    trajectories = {
        syllable: get_instance_trajectories(
            instances,
            coordinates,
            pre=pre,
            post=post,
            centroids=centroids,
            headings=headings,
        )
        for syllable, instances in sampled_instances.items()
    }

    return {s: np.nanmedian(ts, axis=0) for s, ts in trajectories.items()}


def syllable_similarity(
    coordinates,
    results,
    metric="cosine",
    pre=5,
    post=15,
    min_frequency=0.005,
    min_duration=3,
    bodyparts=None,
    use_bodyparts=None,
    density_sample=False,
    sampling_options={"n_neighbors": 50},
    **kwargs,
):
    """Generate a distance matrix over syllable trajectories.

    See :py:func:`keypoint_moseq.util.get_typical_trajectories` for a
    description of the parameters not listed below.

    Parameters
    ----------
    metric: str, default='cosine'
        Distance metric to use. See :py:func:`scipy.spatial.pdist` for options.

    Returns
    -------
    distances : ndarray of shape (n_syllables, n_syllables)
        Pairwise distances between the typical trajectories associated with
        each syllable. Only syllables with sufficient frequency of occurence
        are included.

    syllable_ixs : array of int
        Syllable indexes corresponding to the rows and columns of `distances`.
    """
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
    Xs = np.stack([typical_trajectories[s] for s in syllable_ixs])
    distances = squareform(pdist(Xs.reshape(Xs.shape[0], -1), metric))
    return distances, syllable_ixs


def downsample_timepoints(data, downsample_rate):
    """
    Downsample timepoints, e.g. of coordinates or confidences.

    Parameters
    ----------
    data: ndarray or dict
        Array of shape (n_frames, ...) or a dictionary with such arrays as values.

    downsample_rate: int
        The downsampling rate (e.g., `downsample_rate=2` keeps every other frame).

    Returns
    -------
    downsampled_data: ndarray or dict
        Downsampled array or dictionary of arrays.

    indexes: ndarray or dict
        Downsampled timepoints (in the original numbering)
    """
    if isinstance(data, dict):
        downsampled_data = {}
        indexes = {}
        for k, v in data.items():
            downsampled_data[k], indexes[k] = downsample_timepoints(v, downsample_rate)
    else:
        downsampled_data = data[::downsample_rate]
        indexes = np.arange(len(downsampled_data)) * downsample_rate
    return downsampled_data, indexes


def check_video_paths(video_paths, keys):
    """
    Check if video paths are valid and match the keys.

    Parameters
    ----------
    video_paths: dict
        Dictionary mapping keys to video paths.

    keys: list
        List of keys that require a video path.

    Raises
    ------
    ValueError
        If any of the following are true:
        - a video path is not provided for a key in `keys`
        - a video isn't readable.
        - a video path does not exist.
    """
    missing_keys = set(keys) - set(video_paths.keys())

    nonexistent_videos = []
    unreadable_videos = []
    for path in video_paths.values():
        if not os.path.exists(path):
            nonexistent_videos.append(path)
        else:
            try:
                OpenCVReader(path)[0]
            except:
                unreadable_videos.append(path)

    error_messages = []

    if len(missing_keys) > 0:
        error_messages.append(
            "The following keys require a video path: {}".format(missing_keys)
        )
    if len(nonexistent_videos) > 0:
        error_messages.append(
            "The following videos do not exist: {}".format(nonexistent_videos)
        )
    if len(unreadable_videos) > 0:
        error_messages.append(
            "The following videos are not readable and must be reencoded: {}".format(
                unreadable_videos
            )
        )

    if len(error_messages) > 0:
        raise ValueError("\n\n".join(error_messages))
