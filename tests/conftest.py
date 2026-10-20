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
def tutorial_keypoints(dlc_project_dir):
    """Coordinates, confidences and bodyparts loaded the way the modeling
    notebook loads the example DeepLabCut dataset."""
    import keypoint_moseq as kpms

    keypoint_data_path = os.path.join(dlc_project_dir, "videos")
    return kpms.load_keypoints(keypoint_data_path, "deeplabcut")
