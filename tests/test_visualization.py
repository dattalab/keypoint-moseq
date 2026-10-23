import os
import struct

import imageio
import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

import keypoint_moseq as kpms
from keypoint_moseq.util import get_syllable_instances
from analysis_utils import synthetic_results

PDF_SIGNATURE = b"%PDF"
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
PNG_SIGNATURE = b"\x89PNG"
TRAJECTORY_GIF_FRAMES = 10  # default `num_timesteps` of `plot_trajectories`
GRID_MOVIE_MIN_FREQUENCY = 0.06  # passed by the tutorial_grid_movies fixture
GRID_ROWS, GRID_COLS = 4, 6  # defaults of `generate_grid_movies`
GRID_PRE, GRID_POST = 1.0, 2.0  # seconds; defaults of `generate_grid_movies`


def selected_syllables(
    results, config, pre, post, min_frequency, min_instances, min_duration=3
):
    """Syllables that kpms's plotting functions select with
    `get_syllable_instances`, given their window (in seconds) and filters."""
    instances = get_syllable_instances(
        {name: recording["syllable"] for name, recording in results.items()},
        pre=round(pre * config["fps"]),
        post=round(post * config["fps"]),
        min_duration=min_duration,
        min_frequency=min_frequency,
        min_instances=min_instances,
    )
    return sorted(instances)


def plotted_syllables(results, config):
    """Syllables that `generate_trajectory_plots` draws with its defaults:
    those with enough instances for density sampling with 50 neighbors."""
    return selected_syllables(
        results, config, pre=0.167, post=0.5, min_frequency=0.005, min_instances=50
    )


def top_level_mp4_boxes(path):
    """Types of the top-level boxes of an MP4 file, in file order."""
    boxes = []
    with open(path, "rb") as f:
        while header := f.read(8):
            size, kind = struct.unpack(">I4s", header)
            boxes.append(kind.decode("latin-1"))
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
                f.seek(size - 16, 1)
            elif size == 0:
                break
            else:
                f.seek(size - 8, 1)
    return boxes


def test_trajectory_plot_files(tutorial_trajectory_plots, tutorial_outlier_removed):
    results, output_dir = tutorial_trajectory_plots
    project_dir, _, _ = tutorial_outlier_removed
    syllables = plotted_syllables(results, kpms.load_config(project_dir))

    expected = {
        f"{title}.{extension}"
        for title in ["all_trajectories", *[f"Syllable{s}" for s in syllables]]
        for extension in ["pdf", "gif"]
    }
    assert len(syllables) > 0
    assert set(os.listdir(output_dir)) == expected


def test_trajectory_plot_file_formats(tutorial_trajectory_plots):
    _, output_dir = tutorial_trajectory_plots

    for filename in os.listdir(output_dir):
        path = os.path.join(output_dir, filename)
        with open(path, "rb") as f:
            header = f.read(6)
        if filename.endswith(".pdf"):
            assert header.startswith(PDF_SIGNATURE), filename
        else:
            assert header in GIF_SIGNATURES, filename
            with Image.open(path) as gif:
                assert gif.n_frames == TRAJECTORY_GIF_FRAMES, filename


def test_grid_movie_files(tutorial_grid_movies, tutorial_outlier_removed):
    results, sampled_instances, output_dir = tutorial_grid_movies
    project_dir, _, _ = tutorial_outlier_removed
    syllables = selected_syllables(
        results,
        kpms.load_config(project_dir),
        pre=GRID_PRE,
        post=GRID_POST,
        min_frequency=GRID_MOVIE_MIN_FREQUENCY,
        min_instances=GRID_ROWS * GRID_COLS,
    )

    assert len(syllables) > 0
    assert sorted(sampled_instances) == syllables
    for instances in sampled_instances.values():
        assert len(instances) == GRID_ROWS * GRID_COLS
    assert set(os.listdir(output_dir)) == {f"syllable{s}.mp4" for s in syllables}


def test_grid_movie_video_format(tutorial_grid_movies, tutorial_outlier_removed):
    _, _, output_dir = tutorial_grid_movies
    project_dir, _, _ = tutorial_outlier_removed
    fps = kpms.load_config(project_dir)["fps"]

    for filename in os.listdir(output_dir):
        path = os.path.join(output_dir, filename)
        with imageio.get_reader(path, "ffmpeg") as reader:
            meta = reader.get_meta_data()
        width, height = meta["size"]
        num_frames, _ = imageio_ffmpeg.count_frames_and_secs(path)
        boxes = top_level_mp4_boxes(path)

        assert meta["codec"] == "h264", filename
        assert meta["pix_fmt"].startswith("yuv420p"), filename
        assert meta["fps"] == fps, filename
        assert num_frames == round(GRID_PRE * fps) + round(GRID_POST * fps), filename
        assert width / GRID_COLS == height / GRID_ROWS, filename
        # Grid movies are written without `+faststart`: the index (moov)
        # follows the video data (mdat).
        assert boxes.index("mdat") < boxes.index("moov"), filename


def test_similarity_dendrogram(tutorial_dendrogram, tutorial_outlier_removed):
    results, labels, save_path = tutorial_dendrogram
    project_dir, _, _ = tutorial_outlier_removed
    syllables = selected_syllables(
        results,
        kpms.load_config(project_dir),
        pre=0.167,
        post=0.5,
        min_frequency=0.005,
        min_instances=1,
    )

    assert sorted(labels) == sorted(f"Syllable {s}" for s in syllables)
    for extension, signature in [("pdf", PDF_SIGNATURE), ("png", PNG_SIGNATURE)]:
        with open(f"{save_path}.{extension}", "rb") as f:
            assert f.read(len(signature)) == signature, extension


def test_grid_movies_reject_frame_indexes_beyond_video(
    dlc_project_dir,
    tutorial_outlier_removed,
    tutorial_reindexed,
    tutorial_results,
    tmp_path,
    monkeypatch,
):
    project_dir, coordinates, _ = tutorial_outlier_removed
    model_name, _ = tutorial_reindexed
    config = lambda: kpms.load_config(project_dir)
    results = kpms.load_results(project_dir, model_name)
    name = sorted(results)[0]
    video_frame_indexes = {k: np.arange(len(v["syllable"])) for k, v in results.items()}
    video_frame_indexes[name] = video_frame_indexes[name] + 1  # last index is past the video's end
    monkeypatch.chdir(os.path.dirname(dlc_project_dir))

    with pytest.raises(ValueError) as error:
        kpms.generate_grid_movies(
            results,
            project_dir,
            model_name,
            output_dir=str(tmp_path),
            coordinates=coordinates,
            video_frame_indexes=video_frame_indexes,
            min_frequency=GRID_MOVIE_MIN_FREQUENCY,
            **config(),
        )

    assert name in str(error.value)
    assert "up to" in str(error.value)


def test_keypoints_only_grid_movies_do_not_need_videos(
    tutorial_outlier_removed, tutorial_reindexed, tutorial_results, tmp_path
):
    """Run from the test's working directory, where the config's relative
    `video_dir` does not resolve, so any attempt to open a video fails."""
    project_dir, coordinates, _ = tutorial_outlier_removed
    model_name, _ = tutorial_reindexed
    config = lambda: kpms.load_config(project_dir)
    results = kpms.load_results(project_dir, model_name)

    kpms.generate_grid_movies(
        results,
        project_dir,
        model_name,
        output_dir=str(tmp_path),
        coordinates=coordinates,
        keypoints_only=True,
        min_frequency=GRID_MOVIE_MIN_FREQUENCY,
        **config(),
    )

    assert any(f.endswith(".mp4") for f in os.listdir(tmp_path))


def synthetic_coordinates(results, n_keypoints=4):
    """Keypoints with one row per frame of each recording in `results`."""
    return {
        name: np.zeros((len(r["syllable"]), n_keypoints, 2)) for name, r in results.items()
    }


def test_trajectory_plots_reject_coordinates_missing_recordings(tmp_path):
    results = synthetic_results(["rec_a", "rec_b", "rec_c"])
    coordinates = synthetic_coordinates({"rec_a": results["rec_a"]})

    with pytest.raises(ValueError) as error:
        kpms.generate_trajectory_plots(
            coordinates, results, output_dir=str(tmp_path), save_gifs=False, fps=30
        )

    message = str(error.value)
    assert "2 of 3 recordings" in message
    assert "rec_b" in message
    assert "rec_c" in message


def test_dendrogram_rejects_coordinates_of_a_different_length(tmp_path):
    results = synthetic_results(["rec_a", "rec_b"])
    coordinates = synthetic_coordinates(results)
    coordinates["rec_b"] = coordinates["rec_b"][:-1]

    with pytest.raises(ValueError) as error:
        kpms.plot_similarity_dendrogram(
            coordinates, results, save_path=str(tmp_path / "dendrogram"), fps=30
        )

    message = str(error.value)
    assert "rec_b" in message
    assert "299" in message
    assert "300" in message
    assert "rec_a" not in message
