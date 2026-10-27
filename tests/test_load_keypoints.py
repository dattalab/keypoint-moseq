import os

import numpy as np
import pandas as pd

import keypoint_moseq as kpms
from keypoint_moseq.util import list_files_with_exts


def test_directory_listing_is_sorted(dlc_project_dir, tmp_path):
    filenames = sorted(
        name
        for name in os.listdir(os.path.join(dlc_project_dir, "videos"))
        if name.endswith(".h5")
    )
    # Local filesystems often list a small directory in creation order, so
    # create the files in reverse.
    for name in reversed(filenames):
        (tmp_path / name).touch()

    files = list_files_with_exts(str(tmp_path), ["h5"])

    assert [os.path.basename(path) for path in files] == filenames


def test_dlc_bodyparts_and_recordings(tutorial_keypoints, data_regression):
    coordinates, confidences, bodyparts = tutorial_keypoints

    assert sorted(confidences) == sorted(coordinates)
    data_regression.check({"bodyparts": bodyparts, "recordings": sorted(coordinates)})


def test_dlc_keypoints_for_one_recording(
    tutorial_keypoints, snapshot_recording, ndarrays_regression
):
    coordinates, confidences, _ = tutorial_keypoints

    ndarrays_regression.check(
        {
            "coordinates": coordinates[snapshot_recording],
            "confidences": confidences[snapshot_recording],
        }
    )


def test_float32_deeplabcut_file_reaches_init_model(tmp_path):
    # DeepLabCut 3 writes its predictions as float32. keypoint-MoSeq runs JAX
    # in 64-bit mode, and jax-moseq's init_model rejects 32-bit data.
    bodyparts = ["nose", "head", "spine", "tail"]
    n_frames = 500
    rng = np.random.default_rng(0)
    centroid = 200 + np.cumsum(rng.normal(0, 1, (n_frames, 1, 2)), axis=0)
    coordinates = centroid + rng.normal(0, 5, (n_frames, len(bodyparts), 2))
    likelihood = rng.uniform(0.5, 1, (n_frames, len(bodyparts), 1))
    values = np.concatenate([coordinates, likelihood], axis=-1).astype(np.float32)

    # Column levels, HDF5 key and table format as written by DeepLabCut 3's
    # analyze_videos.
    scorer = "DLC_Resnet50_testOct27shuffle1_snapshot_best-10"
    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    path = tmp_path / f"recording{scorer}.h5"
    pd.DataFrame(values.reshape(n_frames, -1), columns=columns).to_hdf(
        path, key="df_with_missing", format="table", mode="w"
    )

    project_dir = str(tmp_path / "project")
    kpms.setup_project(project_dir, bodyparts=bodyparts)
    kpms.update_config(
        project_dir,
        use_bodyparts=bodyparts,
        anterior_bodyparts=["nose"],
        posterior_bodyparts=["tail"],
        latent_dim=4,
    )
    config = kpms.load_config(project_dir)

    coordinates, confidences, _ = kpms.load_keypoints(str(path), "deeplabcut")
    dtypes = {str(a.dtype) for a in [*coordinates.values(), *confidences.values()]}
    assert dtypes == {"float32"}

    data, metadata = kpms.format_data(coordinates, confidences, **config)
    pca = kpms.fit_pca(**data, **config)
    kpms.init_model(data, metadata=metadata, pca=pca, **config)
