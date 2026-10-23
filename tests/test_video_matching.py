"""Matching recordings to video and keypoint files: when a match fails, the
error says what was searched and what was found there."""

import os
import warnings

import pytest

import keypoint_moseq as kpms


def touch(directory, *names):
    for name in names:
        open(os.path.join(directory, name), "w").close()


def matching_error(keys, video_dir, **kwargs):
    with pytest.raises(ValueError) as error:
        kpms.find_matching_videos(keys, video_dir, **kwargs)
    return str(error.value)


def test_unset_video_dir_raises():
    message = matching_error(["mouseA_dlc"], None)

    assert "video_dir" in message
    assert "not set" in message


def test_missing_video_dir_shows_the_resolved_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    message = matching_error(["mouseA_dlc"], "videos_typo")

    assert "does not exist" in message
    assert os.path.join(str(tmp_path), "videos_typo") in message


def test_tab_in_video_dir_suggests_a_raw_string():
    # "D:\data\testcsv" written without r"" turns "\t" into a tab.
    message = matching_error(["mouseA_dlc"], "D:\\data\testcsv")

    assert "raw string" in message


def test_no_videos_with_the_searched_extensions_lists_what_is_there(tmp_path):
    touch(tmp_path, "mouseA.m4v", "mouseB.m4v", "notes.txt")

    message = matching_error(["mouseA_dlc"], str(tmp_path))

    assert ".m4v" in message
    assert "video_extension" in message


def test_every_unmatched_recording_is_listed_with_its_closest_video(tmp_path):
    touch(tmp_path, "mouseA.mp4", "mouseC_labeled.mp4")
    keys = ["mouseA_dlc", "mouseB_dlc", "mouseCDLC_resnet50"]

    message = matching_error(keys, str(tmp_path))

    assert "2 of 3 recordings" in message
    assert "mouseB_dlc" in message
    assert "mouseCDLC_resnet50" in message
    assert "closest video: mouseC_labeled.mp4" in message
    assert str(tmp_path) in message


def test_mkv_videos_are_matched_by_default(tmp_path):
    touch(tmp_path, "mouseA.mkv")

    videos = kpms.find_matching_videos(["mouseA_dlc"], str(tmp_path), as_dict=True)

    assert videos == {"mouseA_dlc": os.path.join(str(tmp_path), "mouseA.mkv")}


def test_videos_sharing_a_name_warn_and_name_the_one_used(tmp_path):
    touch(tmp_path, "mouseA.avi", "mouseA.mp4")

    with pytest.warns(UserWarning) as record:
        videos = kpms.find_matching_videos(["mouseA_dlc"], str(tmp_path), as_dict=True)

    message = "\n".join(str(w.message) for w in record)
    assert os.path.join(str(tmp_path), "mouseA.avi") in message
    assert os.path.join(str(tmp_path), "mouseA.mp4") in message
    assert f"Using {videos['mouseA_dlc']}" in message


def test_unique_video_names_do_not_warn(tmp_path):
    touch(tmp_path, "mouseA.mp4", "mouseB.avi")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kpms.find_matching_videos(["mouseA_dlc", "mouseB_dlc"], str(tmp_path))


def test_load_keypoints_without_matching_files_lists_what_is_there(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    touch(videos, "mouseA.mp4", "mouseB.mp4")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as error:
        kpms.load_keypoints("videos", "deeplabcut")

    message = str(error.value)
    assert str(videos) in message
    assert ".mp4" in message


def test_load_keypoints_with_a_missing_path_says_so(tmp_path):
    with pytest.raises(ValueError) as error:
        kpms.load_keypoints(str(tmp_path / "no_such_dir"), "deeplabcut")

    assert "does not exist" in str(error.value)
