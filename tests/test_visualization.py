import os
import struct
import subprocess

import imageio
import imageio_ffmpeg
import matplotlib.pyplot as plt
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


def test_grid_movies_with_a_window_size_do_not_need_coordinates(tmp_path):
    results = synthetic_results(["rec_a", "rec_b"])
    for recording in results.values():
        recording["centroid"] = np.clip(recording["centroid"] + 64, 16, 112)
    video_paths = {}
    for name, recording in results.items():
        video_paths[name] = str(tmp_path / f"{name}.mp4")
        with imageio.get_writer(video_paths[name], fps=30) as writer:
            for _ in range(len(recording["syllable"])):
                writer.append_data(np.zeros((128, 128, 3), dtype=np.uint8))
    output_dir = tmp_path / "grid_movies"

    sampled = kpms.generate_grid_movies(
        results,
        output_dir=str(output_dir),
        video_paths=video_paths,
        window_size=32,
        rows=1,
        cols=2,
        fps=30,
        before_onset_sec=0.1,
        after_onset_sec=0.2,
    )

    assert sampled
    assert set(os.listdir(output_dir)) == {f"syllable{s}.mp4" for s in sampled}
    for name in os.listdir(output_dir):
        with imageio.get_reader(str(output_dir / name), "ffmpeg") as reader:
            assert reader.get_meta_data()["size"] == (64, 32), name


def test_density_sampled_grid_movies_of_syllables_with_few_instances(tmp_path):
    results = synthetic_results(["rec_a", "rec_b"])
    # At least the 5 frames before onset and 15 after that density sampling
    # uses for each trajectory.
    fps, pre, post = 30, 0.2, 0.5
    instances = get_syllable_instances(
        {name: recording["syllable"] for name, recording in results.items()},
        before_onset_frames=round(pre * fps),
        after_onset_frames=round(post * fps),
        min_instance_duration_frames=3,
        min_frequency=0.005,
        min_instances=2,
    )
    # enough instances for a 1 x 2 grid, but fewer than the 50 neighbors
    # density sampling uses by default
    assert instances and all(len(v) < 50 for v in instances.values())

    sampled = kpms.generate_grid_movies(
        results,
        output_dir=str(tmp_path),
        coordinates=moving_keypoints(results),
        keypoints_only=True,
        sampling_options={"mode": "density"},
        rows=1,
        cols=2,
        fps=fps,
        before_onset_sec=pre,
        after_onset_sec=post,
    )

    assert sorted(sampled) == sorted(instances)
    for syllable, sampled_instances in sampled.items():
        assert len(set(sampled_instances)) == 2, syllable
        assert set(sampled_instances) <= set(instances[syllable]), syllable
    assert set(os.listdir(tmp_path)) == {f"syllable{s}.mp4" for s in sampled}


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


@pytest.mark.parametrize("num_kappas", [1, 3], ids=["one_kappa", "three_kappas"])
def test_kappa_scan_plot(tmp_path, num_kappas):
    kappas = [10.0**k for k in range(4, 4 + num_kappas)]
    mask = np.ones((1, 12))
    for i, kappa in enumerate(kappas):
        # syllables last 2, 3, 4 frames for successive kappas
        z = (np.arange(12) // (i + 2))[None]
        os.makedirs(tmp_path / f"scan-{kappa}")
        kpms.save_hdf5(
            str(tmp_path / f"scan-{kappa}" / "checkpoint.h5"),
            {
                "data": {"mask": mask},
                "model_snapshots": {"0": {"states": {"z": z}}, "10": {"states": {"z": z}}},
            },
        )

    fig, final_median_durations = kpms.plot_kappa_scan(kappas, str(tmp_path), "scan")

    assert list(final_median_durations) == [2, 3, 4][:num_kappas]
    assert len(fig.axes[0].lines) == num_kappas
    plt.close(fig)


@pytest.mark.parametrize("with_video", [False, True], ids=["keypoints_only", "video"])
def test_grid_movie_dot_shows_on_the_syllable_frames_only(tmp_path, with_video):
    """A syllable on frames 10 to 12 (`end` is exclusive), shown from 2
    frames before its onset to 6 after."""
    num_frames = 40
    if with_video:
        video_paths = {"r": str(tmp_path / "r.mp4")}
        with imageio.get_writer(video_paths["r"], fps=30) as writer:
            for _ in range(num_frames):
                writer.append_data(np.zeros((128, 128, 3), dtype=np.uint8))
        video_frame_indexes = {"r": np.arange(num_frames)}
    else:
        video_paths = video_frame_indexes = None
    coordinates = {"r": np.tile([[64.0, 64.0], [69.0, 64.0], [59.0, 64.0]], (num_frames, 1, 1))}

    frames = kpms.grid_movie(
        [("r", 10, 13)],
        1,
        1,
        video_paths,
        {"r": np.full((num_frames, 2), 64.0)},
        {"r": np.zeros(num_frames)},
        32,
        video_frame_indexes,
        dot_color=(0, 255, 0),
        dot_radius=2,
        before_onset_frames=2,
        after_onset_frames=6,
        overlay_keypoints=not with_video,
        coordinates=coordinates,
        plot_options={"keypoint_colors": [[1.0, 0.0, 0.0]] * 3, "node_size": 1},
    )

    frames_with_dot = [8 + i for i, frame in enumerate(frames) if (frame[..., 1] > 200).any()]
    assert frames_with_dot == [10, 11, 12]


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


def frequency_results():
    # frequencies count instances: 2 of syllable 0, 5 of syllable 1, 3 of syllable 2
    return {"rec": {"syllable": np.array([1, 2, 1, 2, 1, 2, 1, 0, 1, 0])}}


def test_syllable_frequencies_are_plotted_by_rank():
    fig, ax = kpms.plot_syllable_frequencies(
        results=frequency_results(), min_frequency=0.0
    )
    heights = [bar.get_height() for bar in ax.patches]
    np.testing.assert_allclose(heights, [0.5, 0.3, 0.2])
    plt.close(fig)


def test_syllable_frequencies_with_no_syllable_above_the_threshold():
    fig, ax = kpms.plot_syllable_frequencies(
        results=frequency_results(), min_frequency=0.9
    )
    assert len(ax.patches) == 0
    plt.close(fig)


class Sampled(Exception):
    pass


def test_density_sampling_window_is_in_seconds(tmp_path, monkeypatch):
    windows = []

    def record(*args, **kwargs):
        windows.append((kwargs["before_onset_frames"], kwargs["after_onset_frames"]))
        raise Sampled

    monkeypatch.setattr(kpms.viz, "sample_instances", record)
    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)

    with pytest.raises(Sampled):
        kpms.generate_grid_movies(
            results,
            output_dir=str(tmp_path),
            coordinates=moving_keypoints(results),
            keypoints_only=True,
            fps=120,
            min_frequency=0.0,
            rows=1,
            cols=2,
            sampling_options={"mode": "density"},
        )

    # 0.167 s before onset and 0.5 s after, at 120 fps
    assert windows == [(20, 60)]


@pytest.mark.parametrize("before_onset_sec, after_onset_sec", [(0.1, 2.0), (1.0, 0.3)])
def test_density_sampling_with_a_short_grid_window(tmp_path, before_onset_sec, after_onset_sec):
    # at 120 fps, density sampling compares instances over 20 frames before
    # onset and 60 after, more than these grid windows hold
    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)

    sampled = kpms.generate_grid_movies(
        results,
        output_dir=str(tmp_path),
        coordinates=moving_keypoints(results),
        keypoints_only=True,
        fps=120,
        before_onset_sec=before_onset_sec,
        after_onset_sec=after_onset_sec,
        min_instance_duration_sec=0.0,
        min_frequency=0.0,
        rows=1,
        cols=2,
        sampling_options={"mode": "density", "n_neighbors": 5},
    )

    assert len(sampled) > 0


def drifting_blocks_video(path, n_frames, seed):
    """A 96 x 128 color video of random blocks drifting one pixel a frame,
    so every frame differs, with a keyframe every 50 frames."""
    rng = np.random.default_rng(seed)
    blocks = np.kron(rng.integers(0, 256, (12, 32, 3)), np.ones((8, 8, 1))).astype(np.uint8)
    with imageio.get_writer(path, fps=30, ffmpeg_params=["-g", "50"]) as writer:
        for k in range(n_frames):
            writer.append_data(np.ascontiguousarray(blocks[:, k % 128 : k % 128 + 128]))


def reference_grid_movie(
    instances, rows, cols, video_paths, centroids, headings, window_size, video_frame_indexes,
    dot_color, dot_radius, before_onset_frames, after_onset_frames, scaled_window_size,
    edges, overlay_keypoints, coordinates, plot_options,
):
    """grid_movie for 2-D keypoints as written before it read the tiles on
    threads: each tile opens its video with vidio, converts every frame to
    RGB and warps it, and the tiles are stacked and concatenated."""
    from vidio.read import OpenCVReader
    import cv2

    tiles = []
    for key, start, end in instances:
        scale_factor = scaled_window_size / window_size
        clip_frames = slice(start - before_onset_frames, start + after_onset_frames)
        cs = centroids[key][clip_frames]
        h, c = headings[key][start], cs[before_onset_frames]
        r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
        if overlay_keypoints:
            syllable_coordinates = coordinates[key][clip_frames].copy()
        tile = []
        if video_paths is not None:
            reader = OpenCVReader(video_paths[key])
            frames = [reader[ix] for ix in video_frame_indexes[key][clip_frames]]
            reader.close()
            c = r @ c - window_size // 2
            M = [[np.cos(h), np.sin(h), -c[0]], [-np.sin(h), np.cos(h), -c[1]]]
            for ii, (frame, c) in enumerate(zip(frames, cs)):
                if overlay_keypoints:
                    frame = kpms.overlay_keypoints_on_image(
                        frame, syllable_coordinates[ii], edges=edges, **plot_options
                    )
                frame = cv2.warpAffine(frame, np.float32(M), (window_size, window_size))
                frame = cv2.resize(frame, (scaled_window_size, scaled_window_size))
                if 0 <= ii - before_onset_frames < end - start and dot_radius > 0:
                    pos = tuple([int(x) for x in M @ np.append(c, 1) * scale_factor])
                    cv2.circle(frame, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
                tile.append(frame)
        else:
            syllable_coordinates = (
                syllable_coordinates - c
            ) @ r.T * scale_factor + scaled_window_size // 2
            cs = (cs - c) @ r.T * scale_factor + scaled_window_size // 2
            background = np.zeros((scaled_window_size, scaled_window_size, 3))
            for ii, (uvs, c) in enumerate(zip(syllable_coordinates, cs)):
                frame = kpms.overlay_keypoints_on_image(
                    background.copy(), uvs, edges=edges, **plot_options
                )
                if 0 <= ii - before_onset_frames < end - start and dot_radius > 0:
                    cv2.circle(frame, (int(c[0]), int(c[1])), dot_radius, dot_color, -1, cv2.LINE_AA)
                tile.append(frame)
        tiles.append(np.stack(tile))
    tiles = np.stack(tiles).reshape(
        rows, cols, after_onset_frames + before_onset_frames, scaled_window_size, scaled_window_size, 3
    )
    return np.concatenate(np.concatenate(tiles, axis=2), axis=2)


@pytest.mark.parametrize(
    "case", ["video", "video_with_keypoints", "skipping_frame_indexes", "keypoints_only"]
)
def test_grid_movie_frames_are_unchanged(tmp_path, case):
    """grid_movie draws the same frames as the implementation it replaced:
    tiles from two videos, one starting at the first frame, two from the same
    video one after the other and overlapping, scaled down, with the onset dot,
    optionally with the keypoint overlay or without video."""
    n_frames = 220
    rng = np.random.default_rng(1)
    names = ["a", "b"]
    video_paths = {name: str(tmp_path / f"{name}.mp4") for name in names}
    for seed, name in enumerate(names):
        drifting_blocks_video(video_paths[name], n_frames, seed)
    video_frame_indexes = {name: np.arange(n_frames) for name in names}
    clip_frames = n_frames
    if case == "skipping_frame_indexes":
        clip_frames = n_frames // 2
        video_frame_indexes["a"] = np.arange(0, n_frames, 2)
        video_frame_indexes["b"] = np.arange(clip_frames)
    centroids = {name: 48 + rng.uniform(-6, 6, (clip_frames, 2)) * [1, 2] + [0, 16] for name in names}
    headings = {name: rng.uniform(-np.pi, np.pi, clip_frames) for name in names}
    coordinates = {
        name: centroids[name][:, None, :] + rng.normal(scale=6, size=(clip_frames, 3, 2)) for name in names
    }
    instances = [("a", 4, 8), ("b", 40, 52), ("a", 60, 66), ("a", 62, 70), ("b", 90, 96), ("a", 30, 34)]
    options = dict(
        dot_color=(0, 255, 0), dot_radius=3, before_onset_frames=4, after_onset_frames=12,
        scaled_window_size=32, edges=[(0, 1), (1, 2)],
        overlay_keypoints=case in ("video_with_keypoints", "keypoints_only"),
        coordinates=coordinates, plot_options={"node_size": 2, "keypoint_colormap": "autumn"},
    )
    if case == "keypoints_only":
        video_paths = video_frame_indexes = None

    frames = kpms.grid_movie(
        instances, 2, 3, video_paths, centroids, headings, 48, video_frame_indexes, **options
    )
    expected = reference_grid_movie(
        instances, 2, 3, video_paths, centroids, headings, 48, video_frame_indexes, **options
    )

    assert frames.shape == expected.shape == (16, 64, 96, 3)
    np.testing.assert_array_equal(frames, expected)


def test_write_video_clip_encodes_with_the_fastest_preset(tmp_path):
    """The clip is H.264 in the Constrained Baseline profile, which x264
    writes with its ultrafast preset."""
    path = str(tmp_path / "clip.mp4")
    frames = np.random.default_rng(0).integers(0, 256, (10, 64, 96, 3), dtype=np.uint8)

    kpms.write_video_clip(frames, path)

    info = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True
    ).stderr
    assert "h264 (Constrained Baseline)" in info


def test_grid_movies_raise_an_error_from_writing_a_movie(tmp_path, monkeypatch):
    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)

    def fail(*args, **kwargs):
        raise RuntimeError("the encoder failed")

    monkeypatch.setattr(kpms.viz, "write_video_clip", fail)
    with pytest.raises(RuntimeError, match="the encoder failed"):
        kpms.generate_grid_movies(
            results,
            output_dir=str(tmp_path),
            coordinates=moving_keypoints(results),
            keypoints_only=True,
            fps=100,
            before_onset_sec=0.05,
            after_onset_sec=0.1,
            min_frequency=0.0,
            rows=1,
            cols=2,
        )


def gray_ramp_video(path, n_frames=20, size=64):
    """A video whose frame k is uniformly gray at level 10 * k."""
    with imageio.get_writer(path, fps=30) as writer:
        for k in range(n_frames):
            writer.append_data(np.full((size, size, 3), 10 * k, dtype=np.uint8))


@pytest.mark.parametrize("crop_size", [None, 32])
def test_overlay_draws_each_keypoint_row_on_its_video_frame(tmp_path, crop_size):
    video = str(tmp_path / "video.mp4")
    gray_ramp_video(video)
    video_frame_indexes = np.arange(0, 20, 2)
    # two keypoints in opposite corners, so the frame center (and the crop
    # center, their median) shows only the video
    coordinates = np.tile([[10.0, 10.0], [54.0, 54.0]], (10, 1, 1))
    output = str(tmp_path / "overlay.mp4")

    kpms.overlay_keypoints_on_video(
        video,
        coordinates,
        output_path=output,
        show_frame_numbers=False,
        crop_size=crop_size,
        video_frame_indexes=video_frame_indexes,
    )

    with imageio.get_reader(output, "ffmpeg") as reader:
        frames = [frame for frame in reader]
    assert len(frames) == 10
    for i, frame in enumerate(frames):
        center = frame[frame.shape[0] // 2, frame.shape[1] // 2].mean()
        assert abs(center - 10 * video_frame_indexes[i]) <= 4, i


def test_overlay_refuses_the_dict_of_every_recording(tmp_path):
    video = str(tmp_path / "video.mp4")
    gray_ramp_video(video)
    coordinates = np.tile([[10.0, 10.0], [54.0, 54.0]], (10, 1, 1))

    with pytest.raises(TypeError, match=r"video_frame_indexes\[name\]"):
        kpms.overlay_keypoints_on_video(
            video,
            coordinates,
            output_path=str(tmp_path / "overlay.mp4"),
            video_frame_indexes={"rec": np.arange(10)},
        )


def test_overlay_refuses_keypoints_past_the_end_of_the_video(tmp_path):
    video = str(tmp_path / "video.mp4")
    gray_ramp_video(video, n_frames=20)
    coordinates = np.tile([[10.0, 10.0], [54.0, 54.0]], (22, 1, 1))
    output = tmp_path / "overlay.mp4"

    with pytest.raises(ValueError, match=r"need video frames up to 21, but .* has 20 frames"):
        kpms.overlay_keypoints_on_video(video, coordinates, output_path=str(output))

    assert not output.exists()


def count_calls(monkeypatch, module, name):
    """Replace `module.name` with a wrapper that records the arguments of each
    call, and return the list it records into."""
    calls = []
    function = getattr(module, name)

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    return calls


@pytest.mark.parametrize("density_sample", [True, False], ids=["density_sampled", "all_instances"])
def test_typical_trajectories_filter_centroids_and_headings_once(monkeypatch, density_sample):
    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)
    calls = count_calls(monkeypatch, kpms.util, "filter_centroids_headings")

    typical = kpms.util.get_typical_trajectories(
        moving_keypoints(results),
        results,
        min_frequency=0.0,
        density_sample=density_sample,
        sampling_options={"n_neighbors": 10},
    )

    assert len(typical) > 1
    assert len(calls) == 1


def test_typical_trajectories_are_the_median_instance_trajectories():
    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)
    coordinates = moving_keypoints(results)
    instances = get_syllable_instances(
        {name: r["syllable"] for name, r in results.items()},
        before_onset_frames=5,
        after_onset_frames=15,
        min_instance_duration_frames=3,
        min_frequency=0.0,
        min_instances=1,
    )

    typical = kpms.util.get_typical_trajectories(
        coordinates, results, min_frequency=0.0, density_sample=False
    )

    assert sorted(typical) == sorted(instances)
    for syllable, syllable_instances in instances.items():
        trajectories = kpms.util.get_instance_trajectories(
            syllable_instances,
            coordinates,
            before_onset_frames=5,
            after_onset_frames=15,
            centroids={name: r["centroid"] for name, r in results.items()},
            headings={name: r["heading"] for name, r in results.items()},
        )
        np.testing.assert_array_equal(typical[syllable], np.nanmedian(trajectories, axis=0))


def small_trajectories(n_trajectories=2, n_frames=20, n_keypoints=4, seed=0):
    """Titles, trajectories, axis limits and edges for `plot_trajectories`."""
    rng = np.random.default_rng(seed)
    Xs = np.cumsum(rng.normal(size=(n_trajectories, n_frames, n_keypoints, 2)), axis=1)
    lims = kpms.get_limits(Xs, pctl=0, left=0.1, right=0.1, top=0.2, bottom=0.2)
    titles = [f"Syllable{i}" for i in range(n_trajectories)]
    edges = [(i, i + 1) for i in range(n_keypoints - 1)]
    return titles, Xs, lims, edges


def test_trajectory_frames_show_each_step_over_the_earlier_ones():
    """Frame i of the GIF shows the steps up to i with their fade layers and
    the titles, as the finished figure does with the later steps hidden."""
    titles, Xs, lims, edges = small_trajectories()

    fig, ax, rasters = kpms.plot_trajectories(
        titles, Xs, lims, edges=edges, plot_width=2, return_rasters=True
    )

    artists = [a for a in ax.get_children() if a.get_zorder() > 0 and not hasattr(a, "get_text")]
    assert len(rasters) == 10
    for i, raster in enumerate(rasters):
        for artist in artists:
            artist.set_visible(artist.get_zorder() <= i * 4 + 3)
        fig.canvas.draw()
        expected = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        np.testing.assert_array_equal(raster, expected, err_msg=f"frame {i}")
    plt.close(fig)


def test_trajectory_frames_render_the_figure_once(monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    titles, Xs, lims, edges = small_trajectories()
    draws = count_calls(monkeypatch, FigureCanvasAgg, "draw")

    fig, _, rasters = kpms.plot_trajectories(
        titles, Xs, lims, edges=edges, plot_width=2, return_rasters=True
    )

    assert len(rasters) == 10
    assert len(draws) == 1
    plt.close(fig)


def test_trajectory_edges_are_drawn_as_one_line_collection_per_step_and_color():
    from matplotlib.collections import LineCollection

    titles, Xs, lims, edges = small_trajectories()

    fig, ax, _ = kpms.plot_trajectories(titles, Xs, lims, edges=edges, plot_width=2)

    collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert ax.lines == []
    assert len(collections) == 2 * 10
    assert all(len(c.get_segments()) == len(Xs) * len(edges) for c in collections)
    plt.close(fig)


def test_gif_frames_are_quantized_with_fast_octree(tmp_path, monkeypatch):
    # more than 256 colors per frame, so each frame needs an adaptive palette
    frames = np.random.default_rng(0).integers(0, 256, (3, 32, 48, 3), dtype=np.uint8)
    path = str(tmp_path / "frames.gif")
    quantized = count_calls(monkeypatch, Image.Image, "quantize")

    kpms.viz.save_gif(frames, path)
    methods = [kwargs.get("method") for _, kwargs in quantized]

    with Image.open(path) as gif:
        decoded = []
        for i in range(gif.n_frames):
            gif.seek(i)
            decoded.append(np.asarray(gif.convert("RGB")))
    for frame, decoded_frame in zip(frames, decoded):
        expected = Image.fromarray(frame).quantize(256, method=Image.Quantize.FASTOCTREE)
        np.testing.assert_array_equal(decoded_frame, np.asarray(expected.convert("RGB")))
    assert methods == [Image.Quantize.FASTOCTREE] * 3


def test_saving_trajectory_plots_without_animations_renders_nothing_with_agg(
    tmp_path, monkeypatch
):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)
    draws = count_calls(monkeypatch, FigureCanvasAgg, "draw")

    kpms.generate_trajectory_plots(
        moving_keypoints(results),
        results,
        output_dir=str(tmp_path),
        save_gifs=False,
        fps=30,
        min_frequency=0.0,
    )
    plt.close("all")

    assert any(name.endswith(".pdf") for name in os.listdir(tmp_path))
    assert draws == []


def test_saving_the_dendrogram_renders_it_with_agg_once(tmp_path, monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    results = synthetic_results(["rec_a", "rec_b"], n_frames=3000)
    draws = count_calls(monkeypatch, FigureCanvasAgg, "draw")

    kpms.plot_similarity_dendrogram(
        moving_keypoints(results),
        results,
        save_path=str(tmp_path / "dendrogram"),
        fps=30,
        min_frequency=0.0,
    )
    plt.close("all")

    assert sorted(os.listdir(tmp_path)) == ["dendrogram.pdf", "dendrogram.png"]
    assert len(draws) == 1  # for the PNG


def test_instance_trajectories_apply_the_rigid_transform_with_one_shape(monkeypatch):
    """The jitted transform sees the same array shape for any number of
    instances, so it compiles once."""
    results = synthetic_results(["rec_a"], n_frames=300)
    coordinates = moving_keypoints(results)
    transforms = count_calls(monkeypatch, kpms.util, "inverse_rigid_transform")

    for n_instances in [3, 7]:
        instances = [("rec_a", start, start + 5) for start in range(20, 20 + 10 * n_instances, 10)]
        trajectories = kpms.util.get_instance_trajectories(
            instances,
            coordinates,
            before_onset_frames=5,
            after_onset_frames=15,
            centroids={"rec_a": results["rec_a"]["centroid"]},
            headings={"rec_a": results["rec_a"]["heading"]},
        )
        assert trajectories.shape == (n_instances, 20, 4, 2)

    assert len({args[0].shape for args, _ in transforms}) == 1
