import os
import struct

import imageio
import imageio_ffmpeg
import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image
from sklearn.decomposition import PCA

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
        before_onset_frames=round(pre * config["fps"]),
        after_onset_frames=round(post * config["fps"]),
        min_instance_duration_frames=min_duration,
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


def test_grid_movies_reject_frame_indexes_of_the_wrong_length(
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
    video_frame_indexes[name] = video_frame_indexes[name][:-1]
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
    assert "video_frame_indexes" in str(error.value)


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


def moving_keypoints(results, n_keypoints=4, seed=0):
    """Keypoints scattered around each recording's centroid, so grid-movie
    crops have a real extent."""
    rng = np.random.default_rng(seed)
    return {
        name: r["centroid"][:, None, :]
        + rng.normal(scale=5, size=(len(r["centroid"]), n_keypoints, r["centroid"].shape[1]))
        for name, r in results.items()
    }


def test_min_instance_duration_is_in_seconds(tmp_path):
    # Syllable runs in the synthetic results last 3 to 11 frames; at 100 fps,
    # a 0.1 s minimum excludes every run shorter than 10 frames.
    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)

    sampled = kpms.generate_grid_movies(
        results,
        output_dir=str(tmp_path),
        coordinates=moving_keypoints(results),
        keypoints_only=True,
        fps=100,
        before_onset_sec=0.05,
        after_onset_sec=0.1,
        min_instance_duration_sec=0.1,
        min_frequency=0.0,
        rows=1,
        cols=2,
    )

    durations = [end - start for instances in sampled.values() for _, start, end in instances]
    assert durations
    assert min(durations) >= 10


@pytest.mark.parametrize("save_gifs", [False, True], ids=["mp4_only", "mp4_and_gif"])
def test_trajectory_plot_mp4s_span_the_syllable_window(tmp_path, save_gifs):
    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)
    fps, pre, post = 30, 0.167, 0.5  # the default window: 5 frames before onset, 15 after

    kpms.generate_trajectory_plots(
        moving_keypoints(results),
        results,
        output_dir=str(tmp_path),
        save_gifs=save_gifs,
        save_mp4s=True,
        fps=fps,
        before_onset_sec=pre,
        after_onset_sec=post,
        min_frequency=0.0,
    )

    mp4s = sorted(p for p in os.listdir(tmp_path) if p.endswith(".mp4"))
    assert mp4s
    assert any(p.endswith(".gif") for p in os.listdir(tmp_path)) == save_gifs
    window = (round(pre * fps) + round(post * fps)) / fps
    for name in mp4s:
        _, seconds = imageio_ffmpeg.count_frames_and_secs(str(tmp_path / name))
        assert seconds == pytest.approx(window, abs=0.05), name


@pytest.mark.parametrize("copy", [False, True])
def test_overlay_keypoints_on_image_returns_the_drawn_image(copy):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    coordinates = np.array([[10.0, 10.0], [30.0, 30.0]])

    overlaid = kpms.overlay_keypoints_on_image(image, coordinates, copy=copy)

    assert overlaid[10, 10].any() and overlaid[30, 30].any()
    if copy:
        assert not image.any()


@pytest.fixture
def shown_plotly_figures(monkeypatch):
    """Plotly figures that kpms shows, captured instead of rendered."""
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig: shown.append(fig))
    return shown


def keypoint_colorscales(fig):
    """Colorscale of each keypoint trace, as lists like those
    `matplotlib_colormap_to_plotly` returns."""
    return [
        [list(entry) for entry in trace.marker.colorscale]
        for trace in fig.data
        if trace.mode == "markers"
    ]


def test_plotly_colorscale_matches_the_matplotlib_colormap():
    colorscale = kpms.viz.matplotlib_colormap_to_plotly("autumn")

    assert colorscale[0] == [0.0, "rgb(255, 0, 0)"]
    assert colorscale[-1][1] == "rgb(255, 255, 0)"


def test_3D_pc_plot(tmp_path, shown_plotly_figures):
    use_bodyparts = ["nose", "head", "body", "tail"]
    skeleton = [["nose", "head"], ["head", "body"], ["body", "tail"]]
    rng = np.random.default_rng(0)
    pca = PCA().fit(rng.normal(size=(100, (len(use_bodyparts) - 1) * 3)))

    kpms.plot_pcs(
        pca,
        use_bodyparts=use_bodyparts,
        skeleton=skeleton,
        project_dir=str(tmp_path),
        plot_n_pcs=3,
    )

    [fig] = shown_plotly_figures
    colorscale = kpms.viz.matplotlib_colormap_to_plotly("autumn")
    assert [step.label for step in fig.layout.sliders[0].steps] == ["PC 1", "PC 2", "PC 3"]
    # a pose for each PC, and the mean pose
    assert keypoint_colorscales(fig) == [colorscale] * 4
    assert (tmp_path / "pcs.html").exists()


def test_3D_trajectory_plot(tmp_path, shown_plotly_figures):
    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)
    for recording in results.values():
        recording["centroid"] = np.pad(recording["centroid"], ((0, 0), (0, 1)))
    fps = 30

    kpms.generate_trajectory_plots(
        moving_keypoints(results),
        results,
        output_dir=str(tmp_path),
        save_gifs=False,
        fps=fps,
    )

    [fig] = shown_plotly_figures
    titles = [f"Syllable{s}" for s in plotted_syllables(results, {"fps": fps})]
    colorscale = kpms.viz.matplotlib_colormap_to_plotly("autumn")
    assert titles
    assert [step.label for step in fig.layout.sliders[0].steps] == titles
    assert len(keypoint_colorscales(fig)) > 0
    assert all(c == colorscale for c in keypoint_colorscales(fig))
    assert (tmp_path / "all_trajectories.html").exists()


@pytest.mark.parametrize("n_syllables", [1, 3], ids=["one_syllable", "three_syllables"])
def test_3D_trajectory_slider_shows_one_syllable_at_a_time(
    n_syllables, tmp_path, shown_plotly_figures
):
    # generate_trajectory_plots draws one syllable when only one passes its
    # frequency and instance filters.
    n_frames, edges = 20, [[0, 1], [1, 2], [2, 3]]
    Xs = np.random.default_rng(0).normal(size=(n_syllables, n_frames, 4, 3))
    titles = [f"Syllable{s}" for s in range(n_syllables)]

    kpms.plot_trajectories_3D(Xs, titles, edges, str(tmp_path))

    [fig] = shown_plotly_figures
    steps = fig.layout.sliders[0].steps
    traces = n_frames * (len(edges) + 1)  # each frame's keypoints, then its edges
    assert len(fig.data) == n_syllables * traces
    assert [step.label for step in steps] == titles
    for i, step in enumerate(steps):
        visible = np.zeros(len(fig.data), dtype=bool)
        visible[i * traces : (i + 1) * traces] = True
        np.testing.assert_array_equal(step.args[0]["visible"], visible, err_msg=step.label)
    assert (tmp_path / "all_trajectories.html").exists()
