import jax.numpy as jnp
import jax
import re
import commentjson
import json
import numpy as np
import h5py
import joblib
import tqdm
import yaml
import os
import pandas as pd
from textwrap import fill
import sleap_io
from pynwb import NWBHDF5IO
from ndx_pose import PoseEstimation
from itertools import islice

from keypoint_moseq.util import list_files_with_exts, check_nan_proportions
from jax_moseq.utils import get_frequencies, unbatch


def _build_yaml(sections, comments):
    text_blocks = []
    for title, data in sections:
        centered_title = f" {title} ".center(50, "=")
        text_blocks.append(f"\n\n{'#'}{centered_title}{'#'}")
        for key, value in data.items():
            text = yaml.dump({key: value}).strip("\n")
            if key in comments:
                text = f"\n{'#'} {comments[key]}\n{text}"
            text_blocks.append(text)
    return "\n".join(text_blocks)


def _get_path(project_dir, model_name, path, filename, pathname_for_error_msg="path"):
    if path is None:
        assert project_dir is not None and model_name is not None, fill(
            f"`model_name` and `project_dir` are required if `{pathname_for_error_msg}` is None."
        )
        path = os.path.join(project_dir, model_name, filename)
    return path


def generate_config(project_dir, **kwargs):
    """Generate a `config.yml` file with project settings. Default settings
    will be used unless overriden by a keyword argument.

    Parameters
    ----------
    project_dir: str
        A file `config.yml` will be generated in this directory.

    kwargs
        Custom project settings.
    """

    def _update_dict(new, original):
        return {k: new[k] if k in new else v for k, v in original.items()}

    hypperams = _update_dict(
        kwargs,
        {
            "error_estimator": {"slope": -0.5, "intercept": 0.25},
            "obs_hypparams": {
                "sigmasq_0": 0.1,
                "sigmasq_C": 0.1,
                "nu_sigma": 1e5,
                "nu_s": 5,
            },
            "ar_hypparams": {
                "latent_dim": 10,
                "nlags": 3,
                "S_0_scale": 0.01,
                "K_0_scale": 10.0,
            },
            "trans_hypparams": {
                "num_states": 100,
                "gamma": 1e3,
                "alpha": 5.7,
                "kappa": 1e6,
            },
            "cen_hypparams": {"sigmasq_loc": 0.5},
        },
    )

    hypperams = {k: _update_dict(kwargs, v) for k, v in hypperams.items()}

    anatomy = _update_dict(
        kwargs,
        {
            "bodyparts": ["BODYPART1", "BODYPART2", "BODYPART3"],
            "use_bodyparts": ["BODYPART1", "BODYPART2", "BODYPART3"],
            "skeleton": [
                ["BODYPART1", "BODYPART2"],
                ["BODYPART2", "BODYPART3"],
            ],
            "anterior_bodyparts": ["BODYPART1"],
            "posterior_bodyparts": ["BODYPART3"],
        },
    )

    other = _update_dict(
        kwargs,
        {
            "recording_name_suffix": "",
            "verbose": False,
            "conf_pseudocount": 1e-3,
            "video_dir": "",
            "keypoint_colormap": "autumn",
            "whiten": True,
            "fix_heading": False,
            "seg_length": 10000,
        },
    )

    fitting = _update_dict(
        kwargs,
        {
            "added_noise_level": 0.1,
            "PCA_fitting_num_frames": 1000000,
            "conf_threshold": 0.5,
            #         'kappa_scan_target_duration': 12,
            #         'kappa_scan_min': 1e2,
            #         'kappa_scan_max': 1e12,
            #         'num_arhmm_scan_iters': 50,
            #         'num_arhmm_final_iters': 200,
            #         'num_kpslds_scan_iters': 50,
            #         'num_kpslds_final_iters': 500
        },
    )

    comments = {
        "verbose": "whether to print progress messages during fitting",
        "keypoint_colormap": "colormap used for visualization; see `matplotlib.cm.get_cmap` for options",
        "added_noise_level": "upper bound of uniform noise added to the data during initial AR-HMM fitting; this is used to regularize the model",
        "PCA_fitting_num_frames": "number of frames used to fit the PCA model during initialization",
        "video_dir": "directory with videos from which keypoints were derived (used for crowd movies)",
        "recording_name_suffix": "suffix used to match videos to recording names; this can usually be left empty (see `util.find_matching_videos` for details)",
        "bodyparts": "used to access columns in the keypoint data",
        "skeleton": "used for visualization only",
        "use_bodyparts": "determines the subset of bodyparts to use for modeling and the order in which they are represented",
        "anterior_bodyparts": "used to initialize heading",
        "posterior_bodyparts": "used to initialize heading",
        "seg_length": "data are broken up into segments to parallelize fitting",
        "trans_hypparams": "transition hyperparameters",
        "ar_hypparams": "autoregressive hyperparameters",
        "obs_hypparams": "keypoint observation hyperparameters",
        "cen_hypparams": "centroid movement hyperparameters",
        "error_estimator": "parameters to convert neural net likelihoods to error size priors",
        "save_every_n_iters": "frequency for saving model snapshots during fitting; if 0 only final state is saved",
        "kappa_scan_target_duration": "target median syllable duration (in frames) for choosing kappa",
        "whiten": "whether to whiten principal components; used to initialize the latent pose trajectory `x`",
        "conf_threshold": "used to define outliers for interpolation when the model is initialized",
        "conf_pseudocount": "pseudocount used regularize neural network confidences",
        "fix_heading": "whether to keep the heading angle fixed; this should only be True if the pose is constrained to a narrow range of angles, e.g. a headfixed mouse.",
    }

    sections = [
        ("ANATOMY", anatomy),
        ("FITTING", fitting),
        ("HYPER PARAMS", hypperams),
        ("OTHER", other),
    ]

    with open(os.path.join(project_dir, "config.yml"), "w") as f:
        f.write(_build_yaml(sections, comments))


def check_config_validity(config):
    """Check if the config is valid.

    To be valid, the config must satisfy the following criteria:
        - All the elements of `config["use_bodyparts"]` are also in
          `config["bodyparts"]`
        - All the elements of `config["anterior_bodyparts"]` are also in
          `config["use_bodyparts"]`
        - All the elements of `config["anterior_bodyparts"]` are also in
          `config["use_bodyparts"]`
        - For each pair in `config["skeleton"]`, both elements also in
          `config["bodyparts"]`

    Parameters
    ----------
    config: dict

    Returns
    -------
    validity: bool
    """
    error_messages = []

    # check anatomy
    for bodypart in config["use_bodyparts"]:
        if not bodypart in config["bodyparts"]:
            error_messages.append(
                f"ACTION REQUIRED: `use_bodyparts` contains {bodypart} "
                "which is not one of the options in `bodyparts`."
            )

    for bodypart in sum(config["skeleton"], []):
        if not bodypart in config["bodyparts"]:
            error_messages.append(
                f"ACTION REQUIRED: `skeleton` contains {bodypart} "
                "which is not one of the options in `bodyparts`."
            )

    for bodypart in config["anterior_bodyparts"]:
        if not bodypart in config["use_bodyparts"]:
            error_messages.append(
                f"ACTION REQUIRED: `anterior_bodyparts` contains {bodypart} "
                "which is not one of the options in `use_bodyparts`."
            )

    for bodypart in config["posterior_bodyparts"]:
        if not bodypart in config["use_bodyparts"]:
            error_messages.append(
                f"ACTION REQUIRED: `posterior_bodyparts` contains {bodypart} "
                "which is not one of the options in `use_bodyparts`."
            )

    if len(error_messages) == 0:
        return True
    for msg in error_messages:
        print(fill(msg, width=70, subsequent_indent="  "), end="\n\n")
    return False


def load_config(project_dir, check_if_valid=True, build_indexes=True):
    """Load a project config file.

    Parameters
    ----------
    project_dir: str
        Directory containing the config file

    check_if_valid: bool, default=True
        Check if the config is valid using
        :py:func:`keypoint_moseq.io.check_config_validity`

    build_indexes: bool, default=True
        Add keys `"anterior_idxs"` and `"posterior_idxs"` to the config. Each
        maps to a jax array indexing the elements of
        `config["anterior_bodyparts"]` and `config["posterior_bodyparts"]` by
        their order in `config["use_bodyparts"]`

    Returns
    -------
    config: dict
    """
    config_path = os.path.join(project_dir, "config.yml")

    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    if check_if_valid:
        check_config_validity(config)

    if build_indexes:
        config["anterior_idxs"] = jnp.array(
            [config["use_bodyparts"].index(bp) for bp in config["anterior_bodyparts"]]
        )
        config["posterior_idxs"] = jnp.array(
            [config["use_bodyparts"].index(bp) for bp in config["posterior_bodyparts"]]
        )

    if not "skeleton" in config or config["skeleton"] is None:
        config["skeleton"] = []

    return config


def update_config(project_dir, **kwargs):
    """Update the config file stored at `project_dir/config.yml`.

    Use keyword arguments to update key/value pairs in the config. To update
    model hyperparameters, just use the name of the hyperparameter as the
    keyword argument.

    Examples
    --------
    To update `video_dir` to `/path/to/videos`::

      >>> update_config(project_dir, video_dir='/path/to/videos')
      >>> print(load_config(project_dir)['video_dir'])
      /path/to/videos

    To update `trans_hypparams['kappa']` to `100`::

      >>> update_config(project_dir, kappa=100)
      >>> print(load_config(project_dir)['trans_hypparams']['kappa'])
      100
    """
    config = load_config(project_dir, check_if_valid=False, build_indexes=False)
    config.update(kwargs)
    generate_config(project_dir, **config)


def setup_project(
    project_dir,
    deeplabcut_config=None,
    sleap_file=None,
    nwb_file=None,
    freipose_config=None,
    overwrite=False,
    **options,
):
    """
    Setup a project directory with the following structure::

        project_dir
        └── config.yml

    Parameters
    ----------
    project_dir: str
        Path to the project directory (relative or absolute).

    deeplabcut_config: str, default=None
        Path to a deeplabcut config file. Will be used to initialize
        `bodyparts`, `skeleton`, `use_bodyparts` and `video_dir` in the
        keypoint MoSeq config (overrided by kwargs).

    sleap_file: str, default=None
        Path to a .hdf5 or .slp file containing predictions for one video. Will
        be used to initialize `bodyparts`, `skeleton`, and `use_bodyparts` in
        the keypoint MoSeq config (overrided by kwargs).

    nwb_file: str, default=None
        Path to a .nwb file containing predictions for one video. Will be used
        to initialize `bodyparts`, `skeleton`, and `use_bodyparts` in the
        keypoint MoSeq config. (overrided by kwargs).

    freipose_config: str, default=None
        Path to a freipose skeleton config file. Will be used to initialize
        `bodyparts`, `skeleton`, and `use_bodyparts` in the keypoint MoSeq config
        (overrided by kwargs).

    overwrite: bool, default=False
        Overwrite any config.yml that already exists at the path
        `{project_dir}/config.yml`.

    options
        Used to initialize config file. Overrides default settings.
    """

    if os.path.exists(project_dir) and not overwrite:
        print(
            fill(
                f"The directory `{project_dir}` already exists. Use "
                "`overwrite=True` or pick a different name"
            )
        )
        return

    if deeplabcut_config is not None:
        dlc_options = {}
        with open(deeplabcut_config, "r") as stream:
            dlc_config = yaml.safe_load(stream)
            if dlc_config is None:
                raise RuntimeError(
                    f"{deeplabcut_config} does not exists or is not a"
                    " valid yaml file"
                )
            if "multianimalproject" in dlc_config and dlc_config["multianimalproject"]:
                dlc_options["bodyparts"] = dlc_config["multianimalbodyparts"]
                dlc_options["use_bodyparts"] = dlc_config["multianimalbodyparts"]
            else:
                dlc_options["bodyparts"] = dlc_config["bodyparts"]
                dlc_options["use_bodyparts"] = dlc_config["bodyparts"]
            dlc_options["skeleton"] = dlc_config["skeleton"]
            dlc_options["video_dir"] = os.path.join(
                dlc_config["project_path"], "videos"
            )
        options = {**dlc_options, **options}

    elif sleap_file is not None:
        sleap_options = {}
        if os.path.splitext(sleap_file)[1] == ".slp":
            slp_file = sleap_io.load_slp(sleap_file)
            assert len(slp_file.skeletons) == 1, fill(
                f"{sleap_file} contains more than one skeleton. "
                "This is not currently supported. Please "
                "open a github issue or email calebsw@gmail.com"
            )
            skeleton = slp_file.skeletons[0]
            node_names = skeleton.node_names
            edge_names = [[e.source.name, e.destination.name] for e in skeleton.edges]
        else:
            with h5py.File(sleap_file, "r") as f:
                node_names = [n.decode("utf-8") for n in f["node_names"]]
                edge_names = [
                    [n.decode("utf-8") for n in edge] for edge in f["edge_names"]
                ]
        sleap_options["bodyparts"] = node_names
        sleap_options["use_bodyparts"] = node_names
        sleap_options["skeleton"] = edge_names
        options = {**sleap_options, **options}

    elif nwb_file is not None:
        nwb_options = {}
        with NWBHDF5IO(nwb_file, mode="r", load_namespaces=True) as io:
            pose_obj = _load_nwb_pose_obj(io, nwb_file)
            bodyparts = list(pose_obj.nodes[:])
            nwb_options["bodyparts"] = bodyparts
            nwb_options["use_bodyparts"] = bodyparts
            if "edges" in pose_obj.fields:
                edges = pose_obj.edges[:]
                skeleton = [[bodyparts[i], bodyparts[j]] for i, j in edges]
                nwb_options["skeleton"] = skeleton
        options = {**nwb_options, **options}

    elif freipose_config is not None:
        freipose_options = {}
        with open(freipose_config, "r") as stream:
            freipose_config = commentjson.load(stream)
            bodyparts = [kp for kp, color in freipose_config["keypoints"]]
            skeleton = []
            for [bp1, bp2], color in freipose_config["limbs"]:
                if isinstance(bp1, list):
                    bp1 = bp1[0]
                if isinstance(bp2, list):
                    bp2 = bp2[0]
                skeleton.append(tuple(sorted([bodyparts[bp1], bodyparts[bp2]])))
            freipose_options["bodyparts"] = bodyparts
            freipose_options["use_bodyparts"] = bodyparts
            freipose_options["skeleton"] = list(map(list, sorted(set(skeleton))))
        options = {**freipose_options, **options}

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    generate_config(project_dir, **options)


def save_pca(pca, project_dir, pca_path=None):
    """Save a PCA model to disk.

    The model is saved to `pca_path` or else to `{project_dir}/pca.p`.

    Parameters
    ----------
    pca: :py:class:`sklearn.decomposition.PCA`
    project_dir: str
    pca_path: str, default=None
    """
    if pca_path is None:
        pca_path = os.path.join(project_dir, "pca.p")
    joblib.dump(pca, pca_path)


def load_pca(project_dir, pca_path=None):
    """Load a PCA model from disk.

    The model is loaded from `pca_path` or else from `{project_dir}/pca.p`.

    Parameters
    ----------
    project_dir: str
    pca_path: str, default=None

    Returns
    -------
    pca: :py:class:`sklearn.decomposition.PCA`
    """
    if pca_path is None:
        pca_path = os.path.join(project_dir, "pca.p")
        assert os.path.exists(pca_path), fill(f"No PCA model found at {pca_path}")
    return joblib.load(pca_path)


def load_checkpoint(project_dir=None, model_name=None, path=None, iteration=None):
    """Load data and model snapshot from a saved checkpoint.

    The checkpoint path can be specified directly via `path` or else it is
    assumed to be `{project_dir}/{model_name}/checkpoint.h5`.

    Parameters
    ----------
    project_dir: str, default=None
        Project directory; used in conjunction with `model_name` to determine the
        checkpoint path if `path` is not specified.

    model_name: str, default=None
        Model name; used in conjunction with `project_dir` to determine the
        checkpoint path if `path` is not specified.

    path: str, default=None
        Checkpoint path; if not specified, the checkpoint path is set to
        `{project_dir}/{model_name}/checkpoint.h5`.

    iteration: int, default=None
        Determines which model snapshot to load. If None, the last snapshot is
        loaded.

    Returns
    -------
    model: dict
        Model dictionary containing states, parameters, hyperparameters,
        noise prior, and random seed.

    data: dict
        Data dictionary containing observations, confidences, mask and
        associated metadata (see :py:func:`keypoint_moseq.util.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.util.format_data`).

    iteration: int
        Iteration of model fitting corresponding to the loaded snapshot.
    """
    path = _get_path(project_dir, model_name, path, "checkpoint.h5")

    with h5py.File(path, "r") as f:
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])

    if iteration is None:
        iteration = saved_iterations[-1]
    else:
        assert iteration in saved_iterations, fill(
            f"No snapshot found for iteration {iteration}. "
            f"Available iterations are {saved_iterations}"
        )

    model = load_hdf5(path, f"model_snapshots/{iteration}")
    metadata = load_hdf5(path, "metadata")
    data = load_hdf5(path, "data")
    return model, data, metadata, iteration


def reindex_syllables_in_checkpoint(
    project_dir=None, model_name=None, path=None, index=None, runlength=True
):
    """Reindex syllable labels by their frequency in the most recent model
    snapshot in a checkpoint file.

    This is an in-place operation: the checkpoint is loaded from disk, modified
    and saved to disk again. The label permutation is applied to all model
    snapshots in the checkpoint.

    The checkpoint path can be specified directly via `path` or else it is
    assumed to be `{project_dir}/{model_name}/checkpoint.h5`.

    Parameters
    ----------
    project_dir: str, default=None
    model_name: str, default=None
    path: str, default=None

    index: array of shape (num_states,), default=None
        Permutation for syllable labels, where `index[i]` is relabled as `i`.
        If None, syllables are relabled by frequency, with the most frequent
        syllable relabled as 0, and so on.

    runlength: bool, default=True
        If True, frequencies are quantified using the number of non-consecutive
        occurrences of each syllable. If False, frequency is quantified by
        total number of frames.

    Returns
    -------
    index: array of shape (num_states,)
        The index used for permuting syllable labels. If `index[i] = j`, then
        the syllable formerly labeled `j` is now labeled `i`.
    """
    path = _get_path(project_dir, model_name, path, "checkpoint.h5")

    with h5py.File(path, "r") as f:
        saved_iterations = [int(i) for i in f["model_snapshots"]]

    if index is None:
        with h5py.File(path, "r") as f:
            last_iter = np.max(saved_iterations)
            num_states = f[f"model_snapshots/{last_iter}/params/pi"].shape[0]
            z = f[f"model_snapshots/{last_iter}/states/z"][()]
            mask = f["data/mask"][()]
        index = np.argsort(get_frequencies(z, mask, num_states, runlength))[::-1]

    def _reindex(model):
        model["params"]["betas"] = model["params"]["betas"][index]
        model["params"]["pi"] = model["params"]["pi"][index, :][:, index]
        model["params"]["Ab"] = model["params"]["Ab"][index]
        model["params"]["Q"] = model["params"]["Q"][index]
        model["states"]["z"] = np.argsort(index)[model["states"]["z"]]
        return model

    for iteration in tqdm.tqdm(
        saved_iterations, desc="Reindexing", unit="model snapshot", ncols=72
    ):
        model = load_hdf5(path, f"model_snapshots/{iteration}")
        save_hdf5(path, _reindex(model), f"model_snapshots/{iteration}")
    return index


def extract_results(
    model,
    metadata,
    project_dir=None,
    model_name=None,
    save_results=True,
    path=None,
):
    """Extract model outputs and [optionally] save them to disk.

    Model outputs are saved to disk as a .h5 file, either at `path`
    if it is specified, or at `{project_dir}/{model_name}/results.h5` if it is not.
    If a .h5 file with the given path already exists, the outputs will be added
    to it. The results have the following structure::

        results.h5
        ├──recording_name1
        │  ├──syllable      # model state sequence (z), shape=(num_timepoints,)
        │  ├──latent_state  # model latent state (x), shape=(num_timepoints,latent_dim)
        │  ├──centroid      # model centroid (v), shape=(num_timepoints,keypoint_dim)
        │  └──heading       # model heading (h), shape=(num_timepoints,)
        ⋮

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters,
        noise prior, and random seed.

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.util.format_data`).

    save_results : bool, default=True
        If True, the model outputs will be saved to disk.

    project_dir : str, default=None
        Path to the project directory. Required if `save_results=True` and
        `results_path=None`.

    model_name : str, default=None
        Name of the model. Required if `save_results=True` and
        `results_path=None`.

    path : str, default=None
        Optional path for saving model outputs.

    Returns
    -------
    results_dict : dict
        Dictionary of model outputs with the same structure as the results
        `.h5` file.
    """
    if save_results:
        path = _get_path(project_dir, model_name, path, "results.h5")

    states = jax.device_get(model["states"])

    # extract syllables; repeat first syllable an extra `nlags` times
    nlags = states["x"].shape[1] - states["z"].shape[1]
    z = np.pad(states["z"], ((0, 0), (nlags, 0)), mode="edge")
    syllables = unbatch(z, *metadata)

    # extract latent state, centroid, and heading
    latent_state = unbatch(states["x"], *metadata)
    centroid = unbatch(states["v"], *metadata)
    heading = unbatch(states["h"], *metadata)

    results_dict = {
        recording_name: {
            "syllable": syllables[recording_name],
            "latent_state": latent_state[recording_name],
            "centroid": centroid[recording_name],
            "heading": heading[recording_name],
        }
        for recording_name in syllables.keys()
    }

    if save_results:
        save_hdf5(path, results_dict)
        print(fill(f"Saved results to {path}"))

    return results_dict


def load_results(project_dir=None, model_name=None, path=None):
    """Load the results from a modeled dataset.

    The results path can be specified directly via `path`. Otherwise it is
    assumed to be `{project_dir}/{model_name}/results.h5`.

    Parameters
    ----------
    project_dir: str, default=None
    model_name: str, default=None
    path: str, default=None

    Returns
    -------
    results: dict
        See :py:func:`keypoint_moseq.fitting.apply_model`
    """
    path = _get_path(project_dir, model_name, path, "results.h5")
    return load_hdf5(path)


def save_results_as_csv(
    results, project_dir=None, model_name=None, save_dir=None, path_sep="-"
):
    """Save modeling results to csv format.

    This function creates a directory and then saves a separate csv file for
    each recording. The directory is created at `save_dir` if provided,
    otherwise at `{project_dir}/{model_name}/results`.

    Parameters
    ----------
    results: dict
        See :py:func:`keypoint_moseq.io.extract_results`.

    project_dir: str, default=None
        Project directory; required if `save_dir` is not provided.

    model_name: str, default=None
        Name of the model; required if `save_dir` is not provided.

    save_dir: str, default=None
        Optional path to the directory where the csv files will be saved.

    path_sep: str, default='-'
        If a path separator ("/" or "\") is present in the recording name, it
        will be replaced with `path_sep` when saving the csv file.
    """
    save_dir = _get_path(project_dir, model_name, save_dir, "results", "save_dir")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key in tqdm.tqdm(results.keys(), desc="Saving to csv", ncols=72):
        column_names, data = [], []

        if "syllable" in results[key].keys():
            column_names.append(["syllable"])
            data.append(results[key]["syllable"].reshape(-1, 1))

        if "centroid" in results[key].keys():
            d = results[key]["centroid"].shape[1]
            column_names.append(["centroid x", "centroid y", "centroid z"][:d])
            data.append(results[key]["centroid"])

        if "heading" in results[key].keys():
            column_names.append(["heading"])
            data.append(results[key]["heading"].reshape(-1, 1))

        if "latent_state" in results[key].keys():
            latent_dim = results[key]["latent_state"].shape[1]
            column_names.append([f"latent_state {i}" for i in range(latent_dim)])
            data.append(results[key]["latent_state"])

        dfs = [pd.DataFrame(arr, columns=cols) for arr, cols in zip(data, column_names)]
        df = pd.concat(dfs, axis=1)

        for col in df.select_dtypes(include=[np.floating]).columns:
            df[col] = df[col].astype(float).round(4)

        save_name = key.replace(os.path.sep, path_sep)
        save_path = os.path.join(save_dir, save_name)
        df.to_csv(f"{save_path}.csv", index=False)


def _name_from_path(filepath, path_in_name, path_sep, remove_extension):
    """Create a name from a filepath.

    Either return the name of the file (with the extension removed) or return
    the full filepath, where the path separators are replaced with `path_sep`.
    """
    if remove_extension:
        filepath = os.path.splitext(filepath)[0]
    if path_in_name:
        return filepath.replace(os.path.sep, path_sep)
    else:
        return os.path.basename(filepath)


def save_keypoints(
    save_dir, coordinates, confidences=None, bodyparts=None, path_sep="-"
):
    """Convenience function for saving keypoint detections to csv files.

    One csv file is saved for each recording in `coordinates`. Each row in the
    csv corresponds to one frame and the columns are named

        "BODYPART1_x", "BODYPART1_y", "BODYPART1_conf", "BODYPART2_x", ...

    Columns with confidence scores are ommitted if `confidences` is not provided.
    Besides confidences, there can be 2 or 3 columns for each bodypart, depending
    on whether the keypoints are 2D or 3D.

    Parameters
    ----------
    save_dir: str
        Directory to save the results. A separate csv file will be saved for
        each recording in `coordinates`.

    coordinates: dict
        Dictionary mapping recording names to numpy arrays of shape
        (n_frames, n_keypoints, 2[or 3]) that contain the x and y (and z)
        coordinates of the keypoints. If any keys contain a path separator
        (such as "/"), it will be replaced with `path_sep` when naming the
        csv file.

    confidences: dict, default=None
        Dictionary mapping recording names to numpy arrays of shape
        (n_frames, n_keypoints) with the confidence scores of the keypoints.
        Must have the same keys as `coordinates`.

    bodyparts: list, default=None
        List of bodypart names, in the same order as the keypoints in the
        `coordinates` and `confidences` arrays. If None, the bodypart names
        will be set to ["bodypart1", "bodypart2", ...].

    path_sep: str, default='-'
        If a path separator ("/" or "\") is present in the recording name, it
        will be replaced with `path_sep` when saving the csv file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if confidences is not None:
        assert set(coordinates.keys()) == set(confidences.keys()), fill(
            "The keys in `coordinates` and `confidences` must be the same."
        )

    # get number of keypoints and dimensions
    _, num_keypoints, num_dims = next(iter(coordinates.values())).shape

    # generate bodypart names if not provided
    if bodyparts is None:
        bodyparts = [f"bodypart{i}" for i in range(num_keypoints)]

    # create column names
    suffixes = ["x", "y", "z"][:num_keypoints]
    if confidences is not None:
        suffixes += ["conf"]
    columns = [f"{bp}_{suffix}" for bp in bodyparts for suffix in suffixes]

    # save data to csv
    for recording_name, coords in coordinates.items():
        if confidences is not None:
            data = np.concatenate(
                [coords, confidences[recording_name][..., np.newaxis]], axis=-1
            ).reshape(-1, (num_dims + 1) * num_keypoints)
        else:
            data = coords.reshape(-1, num_dims * num_keypoints)
        save_name = recording_name.replace(os.path.sep, path_sep)
        save_path = os.path.join(save_dir, save_name)
        pd.DataFrame(data, columns=columns).to_csv(f"{save_path}.csv", index=False)


def load_keypoints(
    filepath_pattern,
    format,
    extension=None,
    recursive=True,
    path_sep="-",
    path_in_name=False,
    remove_extension=True,
    exclude_individuals=["single"],
):
    """
    Load keypoint tracking results from one or more files. Several file
    formats are supported:

    - deeplabcut
        .csv and .h5/.hdf5 files generated by deeplabcut. For single-animal
        tracking, each file yields a single key/value pair in the returned
        `coordinates` and `confidences` dictionaries. For multi-animal tracking,
        a key/vaue pair will be generated for each tracked individual. For
        example the file `two_mice.h5` with individuals "mouseA" and "mouseB"
        will yield the pair of keys `'two_mice_mouseA', 'two_mice_mouseB'`.

    - sleap
        .slp and .h5/.hdf5 files generated by sleap. For single-animal tracking,
        each file yields a single key/value pair in the returned `coordinates`
        and `confidences` dictionaries. For multi-animal tracking, a key/vaue
        pair will be generated for each track. For example a single file called
        `two_mice.h5` will yield the pair of keys `'two_mice_track0',
        'two_mice_track1'`.

    - anipose
        .csv files generated by anipose. Each file should contain five columns
        per keypoint (x,y,z,error,score), plus a last column with the frame
        number. The `score` column is used as the keypoint confidence.

    - sleap-anipose
        .h5/.hdf5 files generated by sleap-anipose. Each file should contain a
        dataset called `'tracks'` with shape (n_frames, 1, n_keypoints, 3). If
        there is also a `'point_scores'` dataset, it will be used as the
        keypoint confidence. Otherwise, the confidence will be set to 1.

    - nwb
        .nwb files (Neurodata Without Borders). Each file should contain
        exactly one `PoseEstimation` object (for multi-animal tracking, each
        animal should be stored in its own .nwb file). The `PoseEstimation`
        object should contain one `PoseEstimationSeries` object for each
        bodypart. Confidence values are optional and will be set to 1 if not
        present.

    - facemap
        .h5 files saved by Facemap. See Facemap documentation for details:
        https://facemap.readthedocs.io/en/latest/outputs.html#keypoints-processing
        The files should have the format::

            [filename].h5
            └──Facemap
                ├──keypoint1
                │  ├──x
                │  ├──y
                │  └──likelihood
                ⋮

    - freipose
        .json files saved by FreiPose. Each file should contain a list of dicts
        that each include a "kp_xyz" key with the 3D coordinates for one frame.
        Keypoint scores (saved under "kp_score") are not loaded because they are
        not bounded between 0 and 1, which is required for modeling. Since
        FreiPose does not save the bodypart names, the `bodyparts` return
        value is set to None.


    Parameters
    ----------
    filepath_pattern: str or list of str
        Filepath pattern for a set of deeplabcut csv or hdf5 files, or a list
        of such patterns. Filepath patterns can be:

        - single file (e.g. `/path/to/file.csv`)
        - single directory (e.g. `/path/to/dir/`)
        - set of files (e.g. `/path/to/fileprefix*`)
        - set of directories (e.g. `/path/to/dirprefix*`)

    format: str
        Format of the files to load. Must be one of `deeplabcut`, `sleap`,
        `anipose`, or `sleap-anipose`.

    extension: str, default=None
        File extension to use when searching for files. If None, then the
        extension will be inferred from the `format` argument:

        - sleap: 'h5' or 'slp'
        - deeplabcut: 'csv' or 'h5'
        - anipose: 'csv'
        - sleap-anipose: 'h5'
        - nwb: 'nwb'
        - facemap: 'h5'
        - freipose: 'json'

    recursive: bool, default=True
        Whether to search recursively for deeplabcut csv or hdf5 files.

    path_in_name: bool, default=False
        Whether to name the tracking results from each file by the path to the
        file (True) or just the filename (False). If True, the `path_sep`
        argument is used to separate the path components.

    path_sep: str, default='-'
        Separator to use when `path_in_name` is True. For example, if
        `path_sep` is `'-'`, then the tracking results from the file
        `/path/to/file.csv` will be named `path-to-file`. Using `'/'` as the
        separator is discouraged, as it will cause problems saving/loading the
        modeling results to/from hdf5 files.

    remove_extension: bool, default=True
        Whether to remove the file extension when naming the tracking results
        from each file.

    exclude_individuals: list of str, default=["single"]
        List of individuals to exclude from the results. This is only used for
        multi-animal tracking with deeplabcut.

    Returns
    -------
    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays of
        shape (n_frames, n_bodyparts, 2[or 3])

    confidences: dict
        Dictionary mapping filenames to `likelihood` scores as ndarrays of
        shape (n_frames, n_bodyparts)

    bodyparts: list of str
        List of bodypart names. The order of the names matches the order of the
        bodyparts in `coordinates` and `confidences`.
    """
    formats = {
        "deeplabcut": (_deeplabcut_loader, [".csv", ".h5", ".hdf5"]),
        "sleap": (_sleap_loader, [".h5", ".hdf5", ".slp"]),
        "anipose": (_anipose_loader, [".csv"]),
        "sleap-anipose": (_sleap_anipose_loader, [".h5", ".hdf5"]),
        "nwb": (_nwb_loader, [".nwb"]),
        "facemap": (_facemap_loader, [".h5", ".hdf5"]),
        "freipose": (_freipose_loader, [".json"]),
    }

    # get format-specific loader and extensions
    assert format in formats, fill(
        f"Unrecognized format {format}. Must be one of {list(formats.keys())}"
    )
    loader, extensions = formats[format]

    # optionally override default extension list
    if extension is not None:
        extensions = [extension]

    # optionally add format-specific arguments
    if format == "deeplabcut":
        additional_args = {"exclude_individuals": exclude_individuals}
    else:
        additional_args = {}

    # get list of filepaths
    filepaths = list_files_with_exts(filepath_pattern, extensions, recursive=recursive)
    assert len(filepaths) > 0, fill(
        f"No files with extensions {extensions} found for {filepath_pattern}"
    )

    # load keypoints from each file
    coordinates, confidences, bodyparts = {}, {}, None
    for filepath in tqdm.tqdm(filepaths, desc=f"Loading keypoints", ncols=72):
        try:
            name = _name_from_path(filepath, path_in_name, path_sep, remove_extension)
            new_coordinates, new_confidences, bodyparts = loader(
                filepath, name, **additional_args
            )

            if set(new_coordinates.keys()) & set(coordinates.keys()):
                raise ValueError(
                    f"Duplicate names found in {filepath_pattern}:\n\n"
                    f"{set(new_coordinates.keys()) & set(coordinates.keys())}"
                    f"\n\nThis may be caused by repeated filenames with "
                    "different extensions. If so, please set the extension "
                    "explicitly via the `extension` argument. Another possible"
                    " cause is commonly-named files in different directories. "
                    "if that is the case, then set `path_in_name=True`."
                )

        except Exception as e:
            print(fill(f"Error loading {filepath}: {e}"))

        coordinates.update(new_coordinates)
        confidences.update(new_confidences)

    # check for valid results
    assert len(coordinates) > 0, fill(f"No valid results found for {filepath_pattern}")
    check_nan_proportions(coordinates, bodyparts)
    return coordinates, confidences, bodyparts


def _deeplabcut_loader(filepath, name, exclude_individuals=["single"]):
    """Load tracking results from deeplabcut csv or hdf5 files."""
    ext = os.path.splitext(filepath)[1]
    if ext == ".h5":
        df = pd.read_hdf(filepath)
    if ext == ".csv":
        with open(filepath) as f:
            head = list(islice(f, 0, 5))
            if "individuals" in head[1]:
                header = [0, 1, 2, 3]
            else:
                header = [0, 1, 2]
        df = pd.read_csv(filepath, header=header, index_col=0)

    coordinates, confidences = {}, {}
    if "individuals" in df.columns.names:
        ind_bodyparts = {}
        for ind in df.columns.get_level_values("individuals").unique():
            if ind in exclude_individuals:
                print(
                    f'Excluding individual: "{ind}". Set `exclude_individuals=[]` to include.'
                )
            else:
                ind_df = df.xs(ind, axis=1, level="individuals")
                bps = ind_df.columns.get_level_values("bodyparts").unique().tolist()
                ind_bodyparts[ind] = bps

                arr = ind_df.to_numpy().reshape(len(ind_df), -1, 3)
                coordinates[f"{name}_{ind}"] = arr[:, :, :-1]
                confidences[f"{name}_{ind}"] = arr[:, :, -1]

        bodyparts = set(ind_bodyparts[list(ind_bodyparts.keys())[0]])
        assert all([set(bps) == bodyparts for bps in ind_bodyparts.values()]), (
            f"Bodyparts are not consistent across individuals. The following bodyparts "
            f"were found for each individual: {ind_bodyparts}. Use `exclude_individuals`"
            "to exclude specific individuals."
        )
    else:
        bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
        arr = df.to_numpy().reshape(len(df), -1, 3)
        coordinates[name] = arr[:, :, :-1]
        confidences[name] = arr[:, :, -1]

    return coordinates, confidences, bodyparts


def _sleap_loader(filepath, name):
    """Load keypoints from sleap hdf5 or slp files."""
    if os.path.splitext(filepath)[1] == ".slp":
        slp_file = sleap_io.load_slp(filepath)

        assert len(slp_file.skeletons) == 1, fill(
            f"{filepath} contains more than one skeleton. "
            "This is not currently supported. Please "
            "open a github issue or email calebsw@gmail.com"
        )

        bodyparts = slp_file.skeletons[0].node_names
        arr = slp_file.numpy(return_confidence=True)
        coords = arr[:, :, :-1]
        confs = arr[:, :, -1]
    else:
        with h5py.File(filepath, "r") as f:
            coords = f["tracks"][()]
            confs = f["point_scores"][()]
            bodyparts = [name.decode("utf-8") for name in f["node_names"]]

    if coords.shape[0] == 1:
        coordinates = {name: coords[0].T}
        confidences = {name: confs[0].T}
    else:
        coordinates = {f"{name}_track{i}": coords[i].T for i in range(coords.shape[0])}
        confidences = {f"{name}_track{i}": confs[i].T for i in range(coords.shape[0])}
    return coordinates, confidences, bodyparts


def _anipose_loader(filepath, name):
    """Load keypoints from anipose csv files."""
    with open(filepath, "r") as f:
        header = f.readline()

    pattern = r"(?P<string>\w+)_x,(?P=string)_y,(?P=string)_z"
    bodyparts = list(re.findall(pattern, header))

    df = pd.read_csv(filepath)
    coordinates = {
        name: np.stack(
            [df[[f"{bp}_x", f"{bp}_y", f"{bp}_z"]].to_numpy() for bp in bodyparts],
            axis=1,
        )
    }
    confidences = {name: df[[f"{bp}_score" for bp in bodyparts]].to_numpy()}
    return coordinates, confidences, bodyparts


def _sleap_anipose_loader(filepath, name):
    """Load keypoints from sleap-anipose hdf5 files."""
    with h5py.File(filepath, "r") as f:
        coords = f["tracks"][()]
        if "point_scores" in f.keys():
            confs = f["point_scores"][()]
        else:
            confs = np.ones_like(coords[..., 0])
        bodyparts = ["bodypart{}".format(i) for i in range(coords.shape[2])]
        if coords.shape[1] == 1:
            coordinates = {name: coords[:, 0]}
            confidences = {name: confs[:, 0]}
        else:
            coordinates = {
                f"{name}_track{i}": coords[:, i] for i in range(coords.shape[1])
            }
            confidences = {
                f"{name}_track{i}": confs[:, i] for i in range(coords.shape[1])
            }
    return coordinates, confidences, bodyparts


def _load_nwb_pose_obj(io, filepath):
    """Grab PoseEstimation object from an opened .nwb file."""
    all_objs = io.read().all_children()
    pose_objs = [o for o in all_objs if isinstance(o, PoseEstimation)]
    assert len(pose_objs) > 0, fill(f"No PoseEstimation objects found in {filepath}")
    assert len(pose_objs) == 1, fill(
        f"Found multiple PoseEstimation objects in {filepath}. "
        "This is not currently supported. Please open a github "
        "issue to request this feature."
    )
    pose_obj = pose_objs[0]
    return pose_obj


def _nwb_loader(filepath, name):
    """Load keypoints from nwb files."""
    with NWBHDF5IO(filepath, mode="r", load_namespaces=True) as io:
        pose_obj = _load_nwb_pose_obj(io, filepath)
        bodyparts = list(pose_obj.nodes[:])
        coords = np.stack(
            [pose_obj.pose_estimation_series[bp].data[()] for bp in bodyparts],
            axis=1,
        )
        if "confidence" in pose_obj.pose_estimation_series[bodyparts[0]].fields:
            confs = np.stack(
                [
                    pose_obj.pose_estimation_series[bp].confidence[()]
                    for bp in bodyparts
                ],
                axis=1,
            )
        else:
            confs = np.ones_like(coords[..., 0])
        coordinates = {name: coords}
        confidences = {name: confs}
    return coordinates, confidences, bodyparts


def _facemap_loader(filepath, name):
    """Load keypoints from facemap h5 files."""
    with h5py.File(filepath, "r") as h5:
        dset = h5["Facemap"]
        bodyparts = sorted(dset.keys())
        coords, confs = [], []
        for bp in bodyparts:
            coords.append(np.stack([dset[bp]["x"], dset[bp]["y"]], axis=1))
            confs.append(dset[bp]["likelihood"])
        coordinates = {name: np.stack(coords, axis=1)}
        confidences = {name: np.stack(confs, axis=1)}
    return coordinates, confidences, bodyparts


def _freipose_loader(filepath, name):
    """Load keypoints from freipose json files."""
    with open(filepath, "r") as f:
        data = json.load(f)
    coords = np.concatenate([d["kp_xyz"] for d in data], axis=0)
    coordinates = {name: coords}
    confidences = {name: np.ones_like(coords[..., 0])}
    return coordinates, confidences, None


def save_hdf5(filepath, save_dict, datapath=None):
    """Save a dict of pytrees to an hdf5 file. The leaves of the pytrees must
    be numpy arrays, scalars, or strings.

    Parameters
    ----------
    filepath: str
        Path of the hdf5 file to create.

    save_dict: dict
        Dictionary where the values are pytrees, i.e. recursive collections of
        tuples, lists, dicts, and numpy arrays.

    datapath: str, default=None
        Path within the hdf5 file to save the data. If None, the data is saved
        at the root of the hdf5 file.
    """
    with h5py.File(filepath, "a") as f:
        if datapath is not None:
            _savetree_hdf5(jax.device_get(save_dict), f, datapath)
        else:
            for k, tree in save_dict.items():
                _savetree_hdf5(jax.device_get(tree), f, k)


def load_hdf5(filepath, datapath=None):
    """Load a dict of pytrees from an hdf5 file.

    Parameters
    ----------
    filepath: str
        Path of the hdf5 file to load.

    datapath: str, default=None
        Path within the hdf5 file to load the data from. If None, the data is
        loaded from the root of the hdf5 file.

    Returns
    -------
    save_dict: dict
        Dictionary where the values are pytrees, i.e. recursive collections of
        tuples, lists, dicts, and numpy arrays.
    """
    with h5py.File(filepath, "r") as f:
        if datapath is None:
            return {k: _loadtree_hdf5(f[k]) for k in f}
        else:
            return _loadtree_hdf5(f[datapath])


def _savetree_hdf5(tree, group, name):
    """Recursively save a pytree to an h5 file group."""
    if name in group:
        del group[name]
    if isinstance(tree, np.ndarray):
        if tree.dtype.kind == "U":
            dt = h5py.special_dtype(vlen=str)
            group.create_dataset(name, data=tree.astype(object), dtype=dt)
        else:
            group.create_dataset(name, data=tree)
    elif isinstance(tree, (float, int, str)):
        group.create_dataset(name, data=tree)
    else:
        subgroup = group.create_group(name)
        subgroup.attrs["type"] = type(tree).__name__

        if isinstance(tree, (tuple, list)):
            for k, subtree in enumerate(tree):
                _savetree_hdf5(subtree, subgroup, f"arr{k}")
        elif isinstance(tree, dict):
            for k, subtree in tree.items():
                _savetree_hdf5(subtree, subgroup, k)
        else:
            raise ValueError(f"Unrecognized type {type(tree)}")


def _loadtree_hdf5(leaf):
    """Recursively load a pytree from an h5 file group."""
    if isinstance(leaf, h5py.Dataset):
        data = np.array(leaf[()])
        if h5py.check_dtype(vlen=data.dtype) == str:
            data = np.array([item.decode("utf-8") for item in data])
        elif data.dtype.kind == "S":
            data = data.item().decode("utf-8")
        elif data.shape == ():
            data = data.item()
        return data
    else:
        leaf_type = leaf.attrs["type"]
        values = map(_loadtree_hdf5, leaf.values())
        if leaf_type == "dict":
            return dict(zip(leaf.keys(), values))
        elif leaf_type == "list":
            return list(values)
        elif leaf_type == "tuple":
            return tuple(values)
        else:
            raise ValueError(f"Unrecognized type {leaf_type}")
