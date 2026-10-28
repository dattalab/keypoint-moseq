import os
import shutil
from itertools import islice

import numpy as np
import pandas as pd
import pytest
import sleap_io

import keypoint_moseq as kpms
from keypoint_moseq.util import list_files_with_exts

# Files in the tracker output archive (`tracker_outputs_dir`) that DeepLabCut,
# Lightning Pose and Lightning Pose's EKS smoother write in DeepLabCut's
# layout.
DLC3_SINGLE = "generated/dlc-3.0.2/single-animal/analyzed/one_mouse_clip300DLC_Resnet50_syntheticNov11shuffle1_snapshot_1"
DLC3_MULTI = "generated/dlc-3.0.2/multi-animal/analyzed/one_mouse_clip300DLC_Resnet50_syntheticNov11shuffle1_snapshot_1_el"
SUPERANIMAL = "generated/dlc-3.0.2/superanimal/one_mouse_clip300_superanimal_topviewmouse_hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5"
DLC2_MULTI = "generated/dlc-2.3.11/kpms_fixture_ma-kpms-2026-10-27/videos/one_mouse_clip300DLC_dlcrnetms5_kpms_fixture_maOct27shuffle1_5_el"
DEEPLABCUT_LAYOUT_FILES = [
    # DeepLabCut 3.0.2
    *[f"{stem}{suffix}" for stem in [DLC3_SINGLE, DLC3_MULTI] for suffix in [".h5", ".csv", "_filtered.h5", "_filtered.csv"]],
    SUPERANIMAL,
    # DeepLabCut 2.x
    *[f"{DLC2_MULTI}{suffix}" for suffix in [".h5", ".csv", "_filtered.h5", "_filtered.csv"]],
    "public/movement/poses/DLC_single-wasp.predictions.h5",
    "public/movement/poses/DLC_single-wasp.predictions.csv",
    # Lightning Pose and EKS
    "generated/lightning-pose-2.4.3/models/seed0/video_preds/test_vid.csv",
    "public/eks/data/mirror-mouse-separate/180607_004.train_frames=75.rng=0.top.csv",
    "public/eks/data/mirror-mouse-separate/180607_004.train_frames=75.rng=0.bot.csv",
    "generated/lightning-pose-2.4.3/eks/output/eks_singlecam.csv",
    "public/movement/poses/EKS_IBL-paw_multicam_left.predictions.csv",
]
SLEAP_RUNS = ["generated/sleap-1.6.6/single_instance", "generated/sleap-1.6.6/topdown_tracked"]


def read_deeplabcut_columns(filepath, name, exclude_individuals=("single",)):
    """Coordinates, confidences and bodyparts of a file in DeepLabCut's
    layout, taken from the x, y and likelihood columns of each bodypart by
    name, with bodyparts in file order and recordings named the way
    `load_keypoints` names them. Keypoints whose x, y and likelihood are all
    -1, which DeepLabCut 3 writes for undetected animals, are missing, and
    recordings with no keypoints in any frame are left out."""
    if filepath.endswith(".h5"):
        df = pd.read_hdf(filepath)
    else:
        with open(filepath) as f:
            first_cells = [line.split(",")[0] for line in islice(f, 4)]
        n_header = sum(c in ("scorer", "individuals", "bodyparts", "coords") for c in first_cells)
        df = pd.read_csv(filepath, header=list(range(n_header)), index_col=0)

    if "individuals" in df.columns.names:
        parts = {
            f"{name}_{ind}": df.xs(ind, axis=1, level="individuals")
            for ind in df.columns.get_level_values("individuals").unique()
            if ind not in exclude_individuals
        }
    else:
        parts = {name: df}
    coordinates, confidences = {}, {}
    for key, part in parts.items():
        part = part.droplevel("scorer", axis=1)
        bodyparts = part.columns.get_level_values("bodyparts").unique().tolist()
        coordinates[key] = np.stack(
            [part[bp][["x", "y"]].to_numpy() for bp in bodyparts], axis=1
        )
        confidences[key] = np.stack(
            [part[bp]["likelihood"].to_numpy() for bp in bodyparts], axis=1
        )
        padded = (coordinates[key] == -1).all(axis=-1) & (confidences[key] == -1)
        coordinates[key][padded] = np.nan
        confidences[key][padded] = np.nan
        if np.isnan(coordinates[key]).all():
            del coordinates[key], confidences[key]
    return coordinates, confidences, bodyparts


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


@pytest.mark.parametrize("path", DEEPLABCUT_LAYOUT_FILES, ids=os.path.basename)
def test_deeplabcut_layout_file_matches_its_named_columns(tracker_outputs_dir, path):
    # EKS writes nine columns per bodypart (x, y, likelihood, then ensemble
    # medians and variances), so values must be picked by column name.
    filepath = os.path.join(tracker_outputs_dir, path)
    name = os.path.splitext(os.path.basename(filepath))[0]
    expected_coordinates, expected_confidences, expected_bodyparts = (
        read_deeplabcut_columns(filepath, name)
    )

    coordinates, confidences, bodyparts = kpms.load_keypoints(filepath, "deeplabcut")

    assert bodyparts == expected_bodyparts
    assert sorted(coordinates) == sorted(expected_coordinates)
    for key in expected_coordinates:
        np.testing.assert_array_equal(coordinates[key], expected_coordinates[key])
        np.testing.assert_array_equal(confidences[key], expected_confidences[key])


def test_hdf5_extension_loads_like_h5(tracker_outputs_dir, tmp_path):
    source = os.path.join(tracker_outputs_dir, f"{DLC3_SINGLE}.h5")
    shutil.copy(source, tmp_path / "recording.hdf5")
    expected_coordinates, expected_confidences, expected_bodyparts = kpms.load_keypoints(
        source, "deeplabcut"
    )

    coordinates, confidences, bodyparts = kpms.load_keypoints(
        str(tmp_path / "recording.hdf5"), "deeplabcut"
    )

    assert bodyparts == expected_bodyparts
    name = os.path.basename(DLC3_SINGLE)
    np.testing.assert_array_equal(coordinates["recording"], expected_coordinates[name])
    np.testing.assert_array_equal(confidences["recording"], expected_confidences[name])


@pytest.mark.parametrize("extension", [".h5", ".csv"])
def test_deeplabcut_3d_file_loads_as_3d(tmp_path, extension):
    # DeepLabCut's triangulation writes x, y and z for each bodypart and no
    # likelihood.
    bodyparts = ["nose", "spine", "tail"]
    xyz = np.random.default_rng(0).normal(0, 10, (50, len(bodyparts), 3))
    columns = pd.MultiIndex.from_product(
        [["DLC_3D"], bodyparts, ["x", "y", "z"]], names=["scorer", "bodyparts", "coords"]
    )
    df = pd.DataFrame(xyz.reshape(len(xyz), -1), columns=columns)
    path = tmp_path / f"mouseDLC_3D{extension}"
    if extension == ".h5":
        df.to_hdf(path, key="df_with_missing", format="table", mode="w")
    else:
        df.to_csv(path)

    coordinates, confidences, loaded_bodyparts = kpms.load_keypoints(str(path), "deeplabcut")

    assert loaded_bodyparts == bodyparts
    np.testing.assert_allclose(coordinates["mouseDLC_3D"], xyz)
    np.testing.assert_array_equal(confidences["mouseDLC_3D"], np.ones(xyz.shape[:2]))


def test_superanimal_empty_slots_load_as_missing(tracker_outputs_dir):
    # SuperAnimal output has a fixed number of slots per frame, and DeepLabCut
    # 3 fills each slot with no detection with -1 in x, y and likelihood.
    filepath = os.path.join(tracker_outputs_dir, SUPERANIMAL)
    name = os.path.splitext(os.path.basename(filepath))[0]
    likelihood = pd.read_hdf(filepath).xs("likelihood", axis=1, level="coords")

    coordinates, confidences, _ = kpms.load_keypoints(filepath, "deeplabcut")

    n_padded = sum(
        int((likelihood.xs(key.removeprefix(f"{name}_"), axis=1, level="individuals") == -1).to_numpy().sum())
        for key in confidences
    )
    assert n_padded > 0
    assert sum(int(np.isnan(c).sum()) for c in confidences.values()) == n_padded
    assert sum(int(np.isnan(c).all(-1).sum()) for c in coordinates.values()) == n_padded
    assert not any(np.any(c == -1) for c in [*coordinates.values(), *confidences.values()])


def test_recordings_without_keypoints_are_not_loaded(tracker_outputs_dir):
    # Most of the 10 SuperAnimal slots never hold a detection in this video.
    filepath = os.path.join(tracker_outputs_dir, SUPERANIMAL)
    name = os.path.splitext(os.path.basename(filepath))[0]
    likelihood = pd.read_hdf(filepath).xs("likelihood", axis=1, level="coords")
    individuals = likelihood.columns.get_level_values("individuals").unique()
    detected = [
        f"{name}_{ind}"
        for ind in individuals
        if (likelihood.xs(ind, axis=1, level="individuals") != -1).to_numpy().any()
    ]

    coordinates, confidences, _ = kpms.load_keypoints(filepath, "deeplabcut")

    assert 0 < len(detected) < len(individuals)
    assert sorted(coordinates) == sorted(detected)
    assert sorted(confidences) == sorted(detected)


def test_lightning_pose_metric_files_are_not_loaded(tracker_outputs_dir):
    # Lightning Pose writes per-keypoint metrics as CSVs next to each
    # prediction file.
    video_preds = os.path.join(
        tracker_outputs_dir, "generated/lightning-pose-2.4.3/models/seed0/video_preds"
    )

    coordinates, _, _ = kpms.load_keypoints(video_preds, "deeplabcut")

    assert sorted(coordinates) == ["test_vid"]


def test_slp_with_several_videos_loads_every_video(tracker_outputs_dir):
    # Recordings are named after their video's file, so that check_videos and
    # grid movies find the video.
    filepath = os.path.join(tracker_outputs_dir, "public/sleap_io/tests/data/slp/multiview.slp")
    labels = sleap_io.load_slp(filepath)
    expected = {}
    for video in labels.videos:
        video_name = os.path.splitext(os.path.basename(video.filename))[0]
        tracks = labels.numpy(video=video, return_confidence=True).transpose((1, 0, 2, 3))
        if len(tracks) == 1:
            expected[video_name] = tracks[0]
        else:
            for i, track in enumerate(tracks):
                expected[f"{video_name}_track{i}"] = track
    assert len(labels.videos) > 1

    coordinates, confidences, _ = kpms.load_keypoints(filepath, "sleap")

    assert sorted(coordinates) == sorted(expected)
    for key, track in expected.items():
        np.testing.assert_array_equal(coordinates[key], track[..., :2])
        np.testing.assert_array_equal(confidences[key], track[..., 2])


def test_slp_with_repeated_video_names_raises(tmp_path):
    # Two cameras whose videos have the same file name in different folders
    skeleton = sleap_io.Skeleton(["nose", "tail"])
    videos = [
        sleap_io.Video(str(tmp_path / camera / "session.mp4"), open_backend=False)
        for camera in ["cam1", "cam2"]
    ]
    frames = [
        sleap_io.LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[
                sleap_io.PredictedInstance.from_numpy(
                    np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton, point_scores=np.ones(2)
                )
            ],
        )
        for video in videos
    ]
    path = tmp_path / "two_cameras.slp"
    sleap_io.save_slp(sleap_io.Labels(frames, videos=videos, skeletons=[skeleton]), str(path))

    with pytest.raises(ValueError, match="more than one video named session"):
        kpms.load_keypoints(str(path), "sleap")


@pytest.mark.parametrize(
    "export",
    [
        # SLEAP's default analysis export
        "one_mouse_clip300.000_one_mouse_clip300.analysis.h5",
        # sleap_io.save_analysis_h5(..., preset="standard")
        "one_mouse_clip300.predictions.standard.analysis.h5",
    ],
)
@pytest.mark.parametrize("run", SLEAP_RUNS)
def test_sleap_analysis_file_matches_its_slp(tracker_outputs_dir, run, export):
    # Both analysis files were exported from the .slp with sleap-io 0.9.5.
    # The "standard" preset stores `tracks` as (frame, track, node, xy) where
    # the default stores (track, xy, node, frame).
    run_dir = os.path.join(tracker_outputs_dir, run)
    slp_name = "one_mouse_clip300.predictions"
    export_name = os.path.splitext(export)[0]
    slp = kpms.load_keypoints(os.path.join(run_dir, f"{slp_name}.slp"), "sleap")
    analysis = kpms.load_keypoints(os.path.join(run_dir, export), "sleap")

    assert analysis[2] == slp[2]
    slp_tracks = {key.removeprefix(slp_name): key for key in slp[0]}
    analysis_tracks = {key.removeprefix(export_name): key for key in analysis[0]}
    assert sorted(analysis_tracks) == sorted(slp_tracks)
    for track, key in slp_tracks.items():
        np.testing.assert_array_equal(analysis[0][analysis_tracks[track]], slp[0][key])
        np.testing.assert_array_equal(analysis[1][analysis_tracks[track]], slp[1][key])


@pytest.mark.parametrize(
    "path, n_recordings, n_frames, bodyparts",
    [
        (
            # SLEAP 1.2.7
            "public/sleap_io/tests/data/slp/predictions_1.2.7_provenance_and_tracking.slp",
            2,
            101,
            ["head", "thorax", "abdomen", "wingL", "wingR", "forelegL4", "forelegR4",
             "midlegL4", "midlegR4", "hindlegL4", "hindlegR4", "eyeL", "eyeR"],
        ),
        ("public/sleap/tests/data/tracks/clip.predictions.slp", 2, 1500, ["head", "thorax"]),
        (
            # user labels mixed with predictions
            "public/movement/poses/SLEAP_three-mice_Aeon_mixed-labels.predictions.slp",
            3,
            601,
            ["centroid"],
        ),
        (
            # SLEAP 1.3.1's analysis export
            "public/movement/poses/SLEAP_single-mouse_EPM.analysis.h5",
            1,
            18485,
            ["snout", "left_ear", "right_ear", "centre", "tail_base", "tail_end"],
        ),
        (
            "public/sleap/tests/data/hdf5_format_v1/small_robot.000_small_robot_3_frame.analysis.h5",
            1,
            3,
            ["top", "front", "rear"],
        ),
    ],
)
def test_older_sleap_file_loads(tracker_outputs_dir, path, n_recordings, n_frames, bodyparts):
    coordinates, confidences, loaded_bodyparts = kpms.load_keypoints(
        os.path.join(tracker_outputs_dir, path), "sleap"
    )

    assert loaded_bodyparts == bodyparts
    assert len(coordinates) == n_recordings
    assert {v.shape for v in coordinates.values()} == {(n_frames, len(bodyparts), 2)}
    assert {v.shape for v in confidences.values()} == {(n_frames, len(bodyparts))}
