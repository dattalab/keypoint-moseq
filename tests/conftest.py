import os

import pooch
import pytest

OPEN_FIELD_2D_URL = (
    "https://zenodo.org/api/records/10636983/files/open_field_2D.zip/content"
)
OPEN_FIELD_2D_HASH = "md5:22ec317b33039875cb21c76dca027017"


@pytest.fixture(scope="session")
def dlc_project_dir():
    """DeepLabCut project from the tutorial's 2D open field dataset (Zenodo
    record 10636983), downloaded and unpacked once per machine."""
    files = pooch.retrieve(
        OPEN_FIELD_2D_URL,
        known_hash=OPEN_FIELD_2D_HASH,
        fname="open_field_2D.zip",
        path=pooch.os_cache("keypoint-moseq"),
        processor=pooch.Unzip(),
        progressbar=False,
    )
    config_path = next(
        f for f in files if f.endswith(os.path.join("dlc_project", "config.yaml"))
    )
    return os.path.dirname(config_path)


@pytest.fixture(scope="session")
def tutorial_config_edits():
    """Keyword arguments of the modeling notebook's `update_config` call for
    the example DeepLabCut dataset."""
    return dict(
        video_dir="dlc_project/videos/",
        anterior_bodyparts=["nose"],
        posterior_bodyparts=["spine4"],
        use_bodyparts=[
            "spine4",
            "spine3",
            "spine2",
            "spine1",
            "head",
            "nose",
            "right ear",
            "left ear",
        ],
        fps=30,
    )


def setup_tutorial_project(project_dir, dlc_project_dir, config_edits):
    """Set up a project from the example DeepLabCut config and edit it as in
    the modeling notebook."""
    import keypoint_moseq as kpms

    dlc_config = os.path.join(dlc_project_dir, "config.yaml")
    kpms.setup_project(project_dir, deeplabcut_config=dlc_config)
    kpms.update_config(project_dir, **config_edits)


@pytest.fixture
def tutorial_project_dir(tmp_path, dlc_project_dir, tutorial_config_edits):
    """Project set up from the example DeepLabCut config and edited as in the
    modeling notebook."""
    project_dir = str(tmp_path / "demo_project")
    setup_tutorial_project(project_dir, dlc_project_dir, tutorial_config_edits)
    return project_dir


@pytest.fixture(scope="session")
def snapshot_recording():
    """Recording whose arrays are snapshotted; the shortest in the example
    dataset."""
    return "21_12_10_def6a_3.top.irDLC_resnet50_moseq_exampleAug21shuffle1_500000"


@pytest.fixture(scope="session")
def tutorial_keypoints(dlc_project_dir):
    """Coordinates, confidences and bodyparts loaded the way the modeling
    notebook loads the example DeepLabCut dataset."""
    import keypoint_moseq as kpms

    keypoint_data_path = os.path.join(dlc_project_dir, "videos")
    return kpms.load_keypoints(keypoint_data_path, "deeplabcut")


@pytest.fixture(scope="session")
def tutorial_formatted_data(
    tmp_path_factory, dlc_project_dir, tutorial_config_edits, tutorial_keypoints
):
    """Data and metadata from the modeling notebook's outlier removal and
    format_data cells, run on every recording in the example dataset."""
    import keypoint_moseq as kpms

    project_dir = str(tmp_path_factory.mktemp("formatted") / "demo_project")
    setup_tutorial_project(project_dir, dlc_project_dir, tutorial_config_edits)
    config = lambda: kpms.load_config(project_dir)
    coordinates, confidences, _ = tutorial_keypoints

    kpms.update_config(project_dir, outlier_scale_factor=6.0)
    coordinates, confidences = kpms.outlier_removal(
        dict(coordinates), dict(confidences), project_dir, overwrite=False, **config()
    )
    return kpms.format_data(coordinates, confidences, **config())
