import os

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
