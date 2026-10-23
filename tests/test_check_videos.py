import os
import warnings

import cv2
import numpy as np
import pytest

import keypoint_moseq as kpms

WIDTH, HEIGHT = 64, 48
N_FRAMES = 10


def write_video(path, n_frames=N_FRAMES, width=WIDTH, height=HEIGHT, fourcc="MJPG"):
    """Write a video of `n_frames` random-noise frames of size `width` x
    `height`. Noise keeps each frame large enough that truncating the file
    cuts through frames rather than the header."""
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), 30, (width, height)
    )
    for _ in range(n_frames):
        writer.write(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def keypoints_inside(n_frames, n_keypoints=4, seed=0):
    """Keypoints spread over the middle 60% of a WIDTH x HEIGHT frame."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2 * WIDTH, 0.8 * WIDTH, (n_frames, n_keypoints))
    y = rng.uniform(0.2 * HEIGHT, 0.8 * HEIGHT, (n_frames, n_keypoints))
    return np.stack([x, y], axis=-1)


@pytest.fixture
def video_dir(tmp_path):
    """Directory with a 10-frame video for each of mouseA and mouseB."""
    directory = tmp_path / "videos"
    directory.mkdir()
    write_video(directory / "mouseA.avi")
    write_video(directory / "mouseB.avi")
    return str(directory)


def video(video_dir, name):
    return os.path.join(video_dir, f"{name}.avi")


def check_silently(*args, **kwargs):
    """Run check_videos, failing the test on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return kpms.check_videos(*args, **kwargs)


def test_matching_videos_pass_and_return_paths(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(10), "mouseB_dlc": keypoints_inside(10)}

    video_paths = check_silently(coordinates, video_dir=video_dir)

    assert video_paths == {
        "mouseA_dlc": video(video_dir, "mouseA"),
        "mouseB_dlc": video(video_dir, "mouseB"),
    }


def test_keypoints_one_frame_longer_than_video_raise(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(10), "mouseB_dlc": keypoints_inside(11)}

    with pytest.raises(ValueError) as error:
        kpms.check_videos(coordinates, video_dir=video_dir)

    message = str(error.value)
    assert "1 of 2 recordings" in message
    assert "mouseB_dlc" in message
    assert video(video_dir, "mouseB") in message
    assert "up to 10" in message
    assert "has 10 frames" in message
    assert "mouseA_dlc" not in message


def test_video_frame_indexes_beyond_video_raise(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(5), "mouseB_dlc": keypoints_inside(10)}
    video_frame_indexes = {
        "mouseA_dlc": np.array([0, 2, 4, 6, 12]),
        "mouseB_dlc": np.arange(10),
    }

    with pytest.raises(ValueError) as error:
        kpms.check_videos(
            coordinates, video_dir=video_dir, video_frame_indexes=video_frame_indexes
        )

    message = str(error.value)
    assert "mouseA_dlc" in message
    assert "up to 12" in message
    assert "mouseB_dlc" not in message


def test_missing_video_file_raises(tmp_path):
    missing = str(tmp_path / "mouseA.avi")

    with pytest.raises(ValueError) as error:
        kpms.check_videos(
            {"mouseA_dlc": keypoints_inside(10)}, video_paths={"mouseA_dlc": missing}
        )

    message = str(error.value)
    assert "mouseA_dlc" in message
    assert missing in message
    assert "does not exist" in message


def test_unreadable_video_raises(tmp_path):
    unreadable = tmp_path / "mouseA.avi"
    unreadable.write_bytes(b"not a video")

    with pytest.raises(ValueError) as error:
        kpms.check_videos(
            {"mouseA_dlc": keypoints_inside(10)},
            video_paths={"mouseA_dlc": str(unreadable)},
        )

    message = str(error.value)
    assert "mouseA_dlc" in message
    assert str(unreadable) in message
    assert "ffmpeg" in message


def truncate(path, fraction):
    """Cut a video file to its first `fraction` of bytes."""
    with open(path, "rb") as f:
        contents = f.read()
    with open(path, "wb") as f:
        f.write(contents[: int(len(contents) * fraction)])


def test_undecodable_last_needed_frame_raises(tmp_path):
    # A truncated MJPG .avi still reports all 10 frames in its header, but
    # the frames past the cut cannot be decoded.
    path = write_video(tmp_path / "mouseA.avi")
    truncate(path, 0.5)

    with pytest.raises(ValueError) as error:
        kpms.check_videos({"mouseA_dlc": keypoints_inside(10)}, video_dir=str(tmp_path))

    message = str(error.value)
    assert "mouseA_dlc" in message
    assert path in message
    assert "frame 9" in message
    assert "ffmpeg" in message


def test_video_reporting_no_frames_raises(tmp_path):
    # A truncated mp4 loses its index, and OpenCV reports -1 frames.
    path = write_video(tmp_path / "mouseA.mp4", fourcc="mp4v")
    truncate(path, 0.9)

    with pytest.raises(ValueError) as error:
        kpms.check_videos({"mouseA_dlc": keypoints_inside(10)}, video_dir=str(tmp_path))

    message = str(error.value)
    assert path in message
    assert "-1 frames" in message
    assert "ffmpeg" in message


def test_every_bad_recording_is_named_in_one_error(video_dir):
    missing = os.path.join(video_dir, "mouseC.avi")
    coordinates = {"mouseA_dlc": keypoints_inside(15), "mouseC_dlc": keypoints_inside(10)}
    video_paths = {"mouseA_dlc": video(video_dir, "mouseA"), "mouseC_dlc": missing}

    with pytest.raises(ValueError) as error:
        kpms.check_videos(coordinates, video_paths=video_paths)

    message = str(error.value)
    assert "2 of 2 recordings" in message
    assert "up to 14" in message
    assert missing in message


def test_video_paths_missing_a_recording_raise(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(10), "mouseB_dlc": keypoints_inside(10)}

    with pytest.raises(ValueError) as error:
        kpms.check_videos(
            coordinates, video_paths={"mouseA_dlc": video(video_dir, "mouseA")}
        )

    message = str(error.value)
    assert "mouseB_dlc" in message
    assert "no video" in message


def test_keypoints_outside_frame_warn(video_dir):
    outside = keypoints_inside(10)
    outside[:2, :, 0] += WIDTH  # 2 of 10 frames shifted one frame-width right
    coordinates = {"mouseA_dlc": outside, "mouseB_dlc": keypoints_inside(10)}

    with pytest.warns(UserWarning) as record:
        kpms.check_videos(coordinates, video_dir=video_dir)

    message = "\n".join(str(w.message) for w in record)
    assert "mouseA_dlc" in message
    assert video(video_dir, "mouseA") in message
    assert f"{WIDTH} x {HEIGHT}" in message
    assert "20.0%" in message
    assert "mouseB_dlc" not in message


def test_keypoints_just_past_the_edge_are_silent(video_dir):
    near_edge = keypoints_inside(10)
    near_edge[:5, :, 0] = WIDTH  # 1 px past the last column, inside the margin
    coordinates = {"mouseA_dlc": near_edge}

    check_silently(coordinates, video_dir=video_dir)


def test_rare_keypoints_far_outside_are_silent(tmp_path):
    write_video(tmp_path / "mouseA.avi", n_frames=200)
    rare = keypoints_inside(200)  # 800 keypoints
    rare[:4, 0, 0] = 3 * WIDTH  # 4 of 800 (0.5%) far outside
    coordinates = {"mouseA_dlc": rare}

    check_silently(coordinates, video_dir=str(tmp_path))


def test_scaled_keypoints_warning_names_the_scaled_frame(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(10) * 2}

    with pytest.warns(UserWarning) as record:
        kpms.check_videos(coordinates, video_dir=video_dir)

    message = "\n".join(str(w.message) for w in record)
    assert f"{2 * WIDTH} x {2 * HEIGHT}" in message


def test_shifted_keypoints_warning_has_no_scale_hint(video_dir):
    shifted = keypoints_inside(10)
    shifted[..., 0] += 0.5 * WIDTH  # shifted right only; y stays inside
    coordinates = {"mouseA_dlc": shifted}

    with pytest.warns(UserWarning) as record:
        kpms.check_videos(coordinates, video_dir=video_dir)

    message = "\n".join(str(w.message) for w in record)
    assert "mouseA_dlc" in message
    assert "copy of this video" not in message


def test_3d_keypoints_skip_the_frame_bounds_check(video_dir):
    keypoints_3d = np.concatenate(
        [keypoints_inside(10) * 10, np.ones((10, 4, 1))], axis=-1
    )
    coordinates = {"mouseA_dlc": keypoints_3d}

    check_silently(coordinates, video_dir=video_dir)


def test_print_table_lists_each_recording_and_video(video_dir, capsys):
    coordinates = {"mouseA_dlc": keypoints_inside(10), "mouseB_dlc": keypoints_inside(10)}

    kpms.check_videos(coordinates, video_dir=video_dir, print_table=True)

    out = capsys.readouterr().out
    for name in ["mouseA", "mouseB"]:
        assert f"{name}_dlc" in out
        assert video(video_dir, name) in out


def test_no_table_when_print_table_is_false(video_dir, capsys):
    coordinates = {"mouseA_dlc": keypoints_inside(10)}

    kpms.check_videos(coordinates, video_dir=video_dir, print_table=False)

    assert video(video_dir, "mouseA") not in capsys.readouterr().out


def test_video_longer_than_keypoints_warns_about_variable_frame_rate(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(8), "mouseB_dlc": keypoints_inside(10)}

    with pytest.warns(UserWarning) as record:
        kpms.check_videos(coordinates, video_dir=video_dir, fps=30)

    message = "\n".join(str(w.message) for w in record)
    assert "mouseA_dlc" in message
    assert video(video_dir, "mouseA") in message
    assert "10 frames" in message
    assert "8 frames" in message
    assert "setpts=N/(30*TB)" in message
    assert "on purpose" in message
    assert "mouseB_dlc" not in message


def test_video_one_frame_longer_than_keypoints_is_silent(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(9)}

    check_silently(coordinates, video_dir=video_dir, fps=30)


def test_frame_rate_warning_is_skipped_with_video_frame_indexes(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(5)}
    video_frame_indexes = {"mouseA_dlc": np.array([0, 2, 4, 6, 8])}

    check_silently(
        coordinates, video_dir=video_dir, video_frame_indexes=video_frame_indexes, fps=30
    )


def test_video_frame_indexes_missing_a_recording_raise(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(10), "mouseB_dlc": keypoints_inside(10)}

    with pytest.raises(ValueError) as error:
        kpms.check_videos(
            coordinates,
            video_dir=video_dir,
            video_frame_indexes={"mouseA_dlc": np.arange(10)},
        )

    message = str(error.value)
    assert "video_frame_indexes" in message
    assert "mouseB_dlc" in message
    assert "mouseA_dlc" not in message


def test_video_frame_indexes_for_another_recording_raise(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(10)}
    video_frame_indexes = {"mouseA_dlc": np.arange(10), "mouseC_dlc": np.arange(10)}

    with pytest.raises(ValueError) as error:
        kpms.check_videos(
            coordinates, video_dir=video_dir, video_frame_indexes=video_frame_indexes
        )

    assert "mouseC_dlc" in str(error.value)


def test_video_frame_indexes_of_the_wrong_length_raise(video_dir):
    coordinates = {"mouseA_dlc": keypoints_inside(5), "mouseB_dlc": keypoints_inside(10)}
    video_frame_indexes = {"mouseA_dlc": np.array([0, 2, 4, 6]), "mouseB_dlc": np.arange(10)}

    with pytest.raises(ValueError) as error:
        kpms.check_videos(
            coordinates, video_dir=video_dir, video_frame_indexes=video_frame_indexes
        )

    message = str(error.value)
    assert "mouseA_dlc" in message
    assert "4 video frame indexes" in message
    assert "5 frames" in message
    assert "mouseB_dlc" not in message


def test_frame_at_the_reported_frame_count_is_used_when_it_decodes(video_dir, monkeypatch):
    # OpenCV's frame count comes from the video header and can be one short.
    # vidio still reads the frame at that index when it decodes.
    class UndercountingReader(kpms.util.OpenCVReader):
        def __init__(self, filename):
            super().__init__(filename)
            self.nframes -= 1

    monkeypatch.setattr(kpms.util, "OpenCVReader", UndercountingReader)

    check_silently({"mouseA_dlc": keypoints_inside(10)}, video_dir=video_dir)


def test_reencode_command_quotes_the_video_path(tmp_path):
    directory = tmp_path / "my videos"
    directory.mkdir()
    unreadable = directory / "mouse A.avi"
    unreadable.write_bytes(b"not a video")

    with pytest.raises(ValueError) as error:
        kpms.check_videos(
            {"mouseA_dlc": keypoints_inside(10)},
            video_paths={"mouseA_dlc": str(unreadable)},
        )

    message = str(error.value)
    assert f'ffmpeg -i "{unreadable}"' in message
    assert f'"{directory / "mouse A_reencoded.mp4"}"' in message


def test_frame_rate_command_quotes_the_video_path(tmp_path):
    directory = tmp_path / "my videos"
    directory.mkdir()
    path = write_video(directory / "mouse A.avi")

    with pytest.warns(UserWarning) as record:
        kpms.check_videos(
            {"mouse A_dlc": keypoints_inside(8)}, video_dir=str(directory), fps=30
        )

    message = "\n".join(str(w.message) for w in record)
    assert f'ffmpeg -i "{path}"' in message
    assert f'"{directory / "mouse A_cfr.mp4"}"' in message


def test_tutorial_videos_pass_check(
    tutorial_project_dir, tutorial_keypoints, dlc_project_dir, monkeypatch, capsys
):
    """The modeling notebook's check-videos cell, run on the freshly loaded
    tutorial keypoints."""
    monkeypatch.chdir(os.path.dirname(dlc_project_dir))
    config = lambda: kpms.load_config(tutorial_project_dir)
    coordinates, _, _ = tutorial_keypoints

    video_paths = check_silently(coordinates, print_table=True, **config())

    assert set(video_paths) == set(coordinates)
    out = capsys.readouterr().out
    for name in coordinates:
        assert name in out
