import os
import threading
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import FigureCanvasBase, MouseButton, MouseEvent

import keypoint_moseq as kpms
from keypoint_moseq import calibration

NUM_ANNOTATIONS = 25


@pytest.fixture
def calibration_inputs(
    tutorial_project_dir, tutorial_outlier_removed, dlc_project_dir, monkeypatch
):
    """Project directory and outlier-removed keypoints for the calibration
    cell, run from the directory containing `dlc_project` so the config's
    relative `video_dir` resolves the way it does in the notebook."""
    monkeypatch.chdir(os.path.dirname(dlc_project_dir))
    _, coordinates, confidences = tutorial_outlier_removed
    yield tutorial_project_dir, coordinates, confidences
    plt.close("all")


@pytest.fixture
def frames_loaded_up_front(monkeypatch):
    """noise_calibration waits for its background thread to read every
    frame, so a test can click through them all; the sample keys and frames
    are recorded."""
    loaded = {}
    start_loading_frames = calibration._start_loading_frames

    def start_and_wait(sample_keys, *args, **kwargs):
        frames, thread = start_loading_frames(sample_keys, *args, **kwargs)
        thread.join()
        loaded.update(keys=list(sample_keys), images=frames)
        return frames, thread

    monkeypatch.setattr(calibration, "_start_loading_frames", start_and_wait)
    return loaded


def fake_reader(reads, failing=(), gate=None):
    """A stand-in for OpenCVReader whose frame i is a 4 x 6 RGB image filled
    with i % 256. Each read is recorded in `reads` as (path, frame index);
    reads of the pairs in `failing` raise as OpenCVReader does on a frame it
    cannot decode; and with a `gate`, every read after the first waits for
    the gate to be set."""

    class FakeReader:
        def __init__(self, path):
            self.path = path

        def __getitem__(self, ix):
            if gate is not None and reads:
                gate.wait(timeout=60)
            reads.append((self.path, ix))
            if (self.path, ix) in failing:
                raise ValueError(f"error decoding frame {ix} from video {self.path}")
            return np.full((4, 6, 3), ix % 256, np.uint8)

    return FakeReader


class FakeTimer:
    """A stand-in for the figure canvas's timer: tick() runs its callbacks
    if it was started and not stopped since."""

    def __init__(self):
        self.callbacks, self.running = [], False

    def add_callback(self, func, *args, **kwargs):
        self.callbacks.append((func, args, kwargs))

    def start(self, interval=None):
        self.running = True

    def stop(self):
        self.running = False

    def tick(self):
        if self.running:
            for func, args, kwargs in self.callbacks:
                func(*args, **kwargs)


def fake_timers(monkeypatch):
    """Make every canvas's new_timer return a FakeTimer, recorded in the
    returned list."""
    timers = []

    def new_timer(self, *args, **kwargs):
        timers.append(FakeTimer())
        return timers[-1]

    monkeypatch.setattr(FigureCanvasBase, "new_timer", new_timer)
    return timers


def click(fig, x, y):
    """Left-click the figure's axes at data coordinates (x, y)."""
    x_display, y_display = fig.axes[0].transData.transform((x, y))
    event = MouseEvent("button_press_event", fig.canvas, x_display, y_display, button=MouseButton.LEFT)
    fig.canvas.callbacks.process("button_press_event", event)


def annotate(widget, num_annotations):
    """Click a fixed offset from the detected keypoint on successive frames,
    pressing Next after each click."""
    fig = plt.gcf()
    next_button = widget.children[0].children[1]
    for i in range(num_annotations):
        detected_x, detected_y = fig.axes[0].collections[0].get_offsets()[0]
        click(fig, detected_x + 1 + 3 * (i % 5), detected_y)
        next_button.click()


def annotations_path(project_dir):
    return os.path.join(project_dir, "error_annotations.csv")


@pytest.mark.usefixtures("frames_loaded_up_front")
def test_noise_calibration(calibration_inputs, dataframe_regression, num_regression):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)

    np.random.seed(0)
    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config())
    annotate(widget, NUM_ANNOTATIONS)

    info_label, usr_msg = widget.children[1].children
    assert info_label.value.endswith(f"Completed annotations = {NUM_ANNOTATIONS}")
    assert usr_msg.value == f"Annotations saved to {annotations_path(project_dir)}"

    annotations = pd.read_csv(annotations_path(project_dir), comment="#")
    assert len(annotations) == NUM_ANNOTATIONS
    dataframe_regression.check(annotations)
    num_regression.check(
        {
            "slope": [config()["error_estimator"]["slope"]],
            "intercept": [config()["error_estimator"]["intercept"]],
            "conf_threshold": [config()["conf_threshold"]],
        },
        basename="test_noise_calibration_config",
    )


@pytest.mark.usefixtures("frames_loaded_up_front")
def test_noise_calibration_updates_a_held_config_to_match_config_yml(
    calibration_inputs, tutorial_pca
):
    """A config held in a variable, rather than reloaded with `config()`, is
    passed to calibration, then to `init_model`, and its `error_estimator` is
    saved back to config.yml."""
    project_dir, coordinates, confidences = calibration_inputs
    config = kpms.load_config(project_dir)
    original_error_estimator = dict(config["error_estimator"])
    pca, _ = tutorial_pca

    np.random.seed(0)
    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config)
    annotate(widget, NUM_ANNOTATIONS)

    saved_error_estimator = kpms.load_config(project_dir)["error_estimator"]
    assert saved_error_estimator != original_error_estimator
    assert config["error_estimator"] == saved_error_estimator

    name = sorted(coordinates)[0]
    data, metadata = kpms.format_data(
        {name: coordinates[name]}, {name: confidences[name]}, **config
    )
    kpms.init_model(data, metadata=metadata, pca=pca, **config)

    kpms.update_config(project_dir, error_estimator=config["error_estimator"])
    assert kpms.load_config(project_dir)["error_estimator"] == saved_error_estimator


@pytest.mark.usefixtures("frames_loaded_up_front")
def test_noise_calibration_does_not_save_before_twenty_annotations(
    calibration_inputs,
):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    original_config = config()

    np.random.seed(0)
    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config())
    annotate(widget, 19)

    _, usr_msg = widget.children[1].children
    assert usr_msg.value.startswith("Annotations not saved: complete 1 more")
    assert not os.path.exists(annotations_path(project_dir))
    assert config()["error_estimator"] == original_config["error_estimator"]


def test_noise_calibration_existing_annotations_declined(
    calibration_inputs, monkeypatch
):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    with open(annotations_path(project_dir), "w") as f:
        f.write("existing annotations")
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config())

    assert widget is None
    with open(annotations_path(project_dir)) as f:
        assert f.read() == "existing annotations"


def test_noise_calibration_existing_annotations_overwritten(
    calibration_inputs, monkeypatch
):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    with open(annotations_path(project_dir), "w") as f:
        f.write("existing annotations")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config())

    assert widget is not None
    assert not os.path.exists(annotations_path(project_dir))


def test_noise_calibration_rejects_keypoints_longer_than_video(calibration_inputs):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    name = sorted(coordinates)[0]
    coordinates, confidences = dict(coordinates), dict(confidences)
    coordinates[name] = np.concatenate([coordinates[name], coordinates[name][-1:]])
    confidences[name] = np.concatenate([confidences[name], confidences[name][-1:]])

    with pytest.raises(ValueError) as error:
        kpms.noise_calibration(project_dir, coordinates, confidences, **config())

    assert name in str(error.value)
    assert "up to" in str(error.value)


def test_noise_calibration_rejects_frame_indexes_missing_a_recording(calibration_inputs):
    project_dir, coordinates, confidences = calibration_inputs
    config = lambda: kpms.load_config(project_dir)
    first, *rest = sorted(coordinates)
    video_frame_indexes = {k: np.arange(len(coordinates[k])) for k in rest}

    with pytest.raises(ValueError) as error:
        kpms.noise_calibration(
            project_dir,
            coordinates,
            confidences,
            video_frame_indexes=video_frame_indexes,
            **config(),
        )

    assert first in str(error.value)
    assert "video_frame_indexes" in str(error.value)


class Checked(Exception):
    """Raised by the stand-in check_videos, which noise_calibration calls
    right after checking the matplotlib backend."""


def start_calibration(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise Checked

    monkeypatch.setattr(kpms.calibration, "check_videos", stop)
    coordinates = {"mouseA": np.zeros((10, 4, 2))}
    confidences = {"mouseA": np.ones((10, 4))}
    with pytest.raises(Checked):
        kpms.noise_calibration(
            str(tmp_path),
            coordinates,
            confidences,
            bodyparts=list("abcd"),
            use_bodyparts=list("abcd"),
            video_dir=str(tmp_path),
        )


def test_calibration_warns_without_the_ipympl_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")

    with pytest.warns(UserWarning) as record:
        start_calibration(tmp_path, monkeypatch)

    messages = [" ".join(str(w.message).split()) for w in record]
    message = next(m for m in messages if "ipympl" in m)
    assert "'agg'" in message
    assert "%matplotlib widget" in message


@pytest.mark.parametrize("backend", ["widget", "ipympl", "module://ipympl.backend_nbagg"])
def test_calibration_is_silent_with_the_ipympl_backend(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        start_calibration(tmp_path, monkeypatch)


def confidence_bins(confidences, bodyparts, use_bodyparts, num_bins=10, conf_pseudocount=1e-3):
    """The (recording, frame, bodypart) observations in each of
    sample_error_frames' log-spaced confidence bins."""
    confidences = {k: v + conf_pseudocount for k, v in confidences.items()}
    all_confs = np.concatenate([v.flatten() for v in confidences.values()])
    min_conf, max_conf = np.nanmin(all_confs), np.nanmax(all_confs)
    thresholds = np.logspace(np.log10(min_conf), np.log10(max_conf), num_bins + 1)
    mask = np.array([bp in use_bodyparts for bp in bodyparts])[None, :]
    return [
        {
            (key, t, bodyparts[k])
            for key, confs in confidences.items()
            for t, k in zip(*np.nonzero((confs >= low) * (confs < high) * mask))
        }
        for low, high in zip(thresholds[:-1], thresholds[1:])
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_error_frames_draws_evenly_from_the_confidence_bins(seed):
    """Three recordings (one of a single frame), a bodypart left out of
    use_bodyparts, NaN confidences, and confidences clustered near 1 so that
    the low bins hold fewer entries than they would sample. Each bin gives 10
    distinct observations, or all of them when it has fewer, and the same
    seed draws the same frames."""
    rng = np.random.default_rng(seed)
    bodyparts = ["a", "b", "c", "d", "e"]
    confidences = {
        "rec0": rng.uniform(0, 1, (400, 5)) ** 0.25,
        "rec1": rng.uniform(0.9, 1, (1, 5)),
        "rec2": np.concatenate([rng.uniform(0.95, 1, (60, 5)), np.full((3, 5), np.nan)]),
    }
    use_bodyparts = ["a", "c", "d", "e"]
    bins = confidence_bins(confidences, bodyparts, use_bodyparts)
    assert min(len(b) for b in bins) < 10 < max(len(b) for b in bins)

    np.random.seed(seed)
    sampled = calibration.sample_error_frames(confidences, bodyparts, use_bodyparts)

    assert len(set(sampled)) == len(sampled)
    assert [len(b.intersection(sampled)) for b in bins] == [min(10, len(b)) for b in bins]
    assert len(sampled) == sum(min(10, len(b)) for b in bins)
    np.random.seed(seed)
    assert calibration.sample_error_frames(confidences, bodyparts, use_bodyparts) == sampled


def open_widget(calibration_inputs, frames_loaded_up_front):
    """The calibration widget on the tutorial data, with every frame read and
    the frames recorded in their display order."""
    project_dir, coordinates, confidences = calibration_inputs
    config = kpms.load_config(project_dir)
    np.random.seed(0)
    widget = kpms.noise_calibration(project_dir, coordinates, confidences, **config)
    return widget, plt.gcf(), frames_loaded_up_front, coordinates, config["bodyparts"]


def assert_shows(widget, fig, shown, coordinates, bodyparts, index, mark=None, completed=0):
    """The figure shows the index'th sampled frame in full, with the circle on
    its detected keypoint and, when `mark` is given, one red cross there;
    the label names its bodypart and the number of annotations."""
    ax = fig.axes[0]
    key = shown["keys"][index]
    recording, frame, bodypart = key
    image = shown["images"][key]
    height, width = image.shape[:2]

    images = [im for im in ax.images if im.get_visible()]
    assert len(images) == 1
    np.testing.assert_array_equal(images[0].get_array(), image)
    assert ax.get_xlim() == (-0.5, width - 0.5)
    assert ax.get_ylim() == (height - 0.5, -0.5)

    circle, *others = ax.collections
    np.testing.assert_allclose(
        circle.get_offsets(), [coordinates[recording][frame, bodyparts.index(bodypart)]]
    )
    marks = [xy for c in others if c.get_visible() for xy in c.get_offsets()]
    if mark is None:
        assert len(marks) == 0
    else:
        assert len(marks) == 1
        np.testing.assert_allclose(marks[0], mark)

    info_label, _ = widget.children[1].children
    assert info_label.value == (
        f"Target bodypart = {bodypart} | Completed annotations = {completed}"
    )


def detected_point(fig):
    return fig.axes[0].collections[0].get_offsets()[0]


def test_widget_shows_each_frame_with_its_annotation_through_every_flow(
    calibration_inputs, frames_loaded_up_front
):
    widget, fig, shown, coordinates, bodyparts = open_widget(calibration_inputs, frames_loaded_up_front)
    prev_button, next_button = widget.children[0].children
    check = lambda index, mark=None, completed=0: assert_shows(  # noqa: E731
        widget, fig, shown, coordinates, bodyparts, index, mark, completed
    )
    check(0)

    x, y = detected_point(fig)
    click(fig, x + 3, y)
    check(0, (x + 3, y), 1)
    click(fig, x - 2, y + 4)  # re-annotating replaces the cross
    check(0, (x - 2, y + 4), 1)
    prev_button.click()  # already on the first frame
    check(0, (x - 2, y + 4), 1)

    next_button.click()
    check(1, None, 1)
    x1, y1 = detected_point(fig)
    click(fig, x1 + 5, y1)
    check(1, (x1 + 5, y1), 2)
    next_button.click()
    check(2, None, 2)
    prev_button.click()
    check(1, (x1 + 5, y1), 2)
    prev_button.click()
    check(0, (x - 2, y + 4), 2)
    click(fig, x + 1, y + 1)  # re-annotating after coming back
    check(0, (x + 1, y + 1), 2)

    last = len(shown["keys"]) - 1
    for _ in range(last + 1):  # one press more than there are frames
        next_button.click()
    check(last, None, 2)


def test_widget_shows_the_whole_frame_again_after_zooming(
    calibration_inputs, frames_loaded_up_front
):
    widget, fig, shown, coordinates, bodyparts = open_widget(calibration_inputs, frames_loaded_up_front)
    prev_button, next_button = widget.children[0].children
    ax = fig.axes[0]
    x, y = detected_point(fig)
    ax.set_xlim(x - 40, x + 40)  # as the toolbar's zoom does
    ax.set_ylim(y + 40, y - 40)
    fig.canvas.draw()

    click(fig, x + 3, y + 3)
    assert ax.get_xlim() == (x - 40, x + 40)
    next_button.click()
    assert_shows(widget, fig, shown, coordinates, bodyparts, 1, None, 1)
    prev_button.click()
    assert_shows(widget, fig, shown, coordinates, bodyparts, 0, (x + 3, y + 3), 1)


def pixels(fig):
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def reference_drawing(image, detected, mark=None):
    """The widget's figure drawn from scratch, as the widget drew every frame
    before it reused its image and markers: the same figure and axes layout,
    the axes cleared, the frame shown with imshow, a yellow circle on the
    detected keypoint and a red cross on the annotation."""
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplots_adjust(top=1, bottom=0.01, left=0.07, right=0.99)
    ax.margins(y=0)
    ax.set_frame_on(False)
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.1, pos.y0, pos.width - 0.1, pos.height + pos.y0])
    ax.clear()
    ax.imshow(image)
    ax.scatter(*detected, color="yellow", marker="o", facecolor="none")
    if mark is not None:
        ax.scatter(*mark, color="red", marker="x")
    fig.canvas.draw()
    result = pixels(fig)
    plt.close(fig)
    return result


def test_widget_draws_what_a_fresh_drawing_shows(calibration_inputs, frames_loaded_up_front):
    """After each kind of update, the figure's pixels equal the frame drawn
    from scratch with the same circle and cross; a cross that was moved or
    shown with a blitted update leaves nothing behind."""
    widget, fig, shown, coordinates, bodyparts = open_widget(calibration_inputs, frames_loaded_up_front)
    prev_button, next_button = widget.children[0].children

    def expected(index, mark=None):
        key = shown["keys"][index]
        return reference_drawing(shown["images"][key], detected_point(fig), mark)

    np.testing.assert_array_equal(pixels(fig), expected(0))
    x, y = detected_point(fig)
    click(fig, x + 6, y - 4)
    np.testing.assert_array_equal(pixels(fig), expected(0, (x + 6, y - 4)))
    click(fig, x - 5, y + 2)
    np.testing.assert_array_equal(pixels(fig), expected(0, (x - 5, y + 2)))
    next_button.click()
    np.testing.assert_array_equal(pixels(fig), expected(1))
    prev_button.click()
    np.testing.assert_array_equal(pixels(fig), expected(0, (x - 5, y + 2)))


def test_widget_reuses_its_image_and_redraws_only_the_cross_on_a_click(
    calibration_inputs, frames_loaded_up_front
):
    widget, fig, shown, coordinates, bodyparts = open_widget(calibration_inputs, frames_loaded_up_front)
    prev_button, next_button = widget.children[0].children
    ax = fig.axes[0]
    image = ax.images[0]
    full_draws = []
    fig.canvas.mpl_connect("draw_event", lambda event: full_draws.append(event))

    x, y = detected_point(fig)
    click(fig, x + 3, y)
    click(fig, x + 4, y)
    assert full_draws == []

    next_button.click()
    prev_button.click()
    assert list(ax.images) == [image]
    assert len(full_draws) == 2


def test_noise_calibration_returns_once_the_first_frame_is_read(
    calibration_inputs, monkeypatch
):
    """The first frame is read before noise_calibration returns and shown;
    a background thread reads the rest afterwards, in display order."""
    project_dir, coordinates, confidences = calibration_inputs
    reads, gate = [], threading.Event()
    monkeypatch.setattr(calibration, "OpenCVReader", fake_reader(reads, gate=gate))
    started = {}
    start_loading_frames = calibration._start_loading_frames

    def record(sample_keys, *args, **kwargs):
        started["keys"] = list(sample_keys)
        started["frames"], started["thread"] = start_loading_frames(sample_keys, *args, **kwargs)
        return started["frames"], started["thread"]

    monkeypatch.setattr(calibration, "_start_loading_frames", record)
    np.random.seed(0)
    kpms.noise_calibration(project_dir, coordinates, confidences, **kpms.load_config(project_dir))

    keys, frames = started["keys"], started["frames"]
    assert list(frames) == keys[:1]
    np.testing.assert_array_equal(plt.gcf().axes[0].images[0].get_array(), frames[keys[0]])

    gate.set()
    started["thread"].join(timeout=60)
    assert list(frames) == keys
    for recording, frame, bodypart in keys:
        np.testing.assert_array_equal(
            frames[(recording, frame, bodypart)], np.full((4, 6, 3), frame % 256, np.uint8)
        )


def test_a_frame_that_cannot_be_read_is_described_and_the_rest_are_read(monkeypatch):
    """Reading fails for the first frame, read before the thread starts, and
    for one the thread reads; each is replaced by a message naming the
    recording, the video, both frame numbers and the error."""
    reads = []
    failing = {("videos/a.mp4", 2), ("videos/b.mp4", 7)}
    monkeypatch.setattr(calibration, "OpenCVReader", fake_reader(reads, failing=failing))
    sample_keys = [("recA", 2, "nose"), ("recB", 3, "tail"), ("recA", 5, "tail")]
    video_paths = {"recA": "videos/a.mp4", "recB": "videos/b.mp4"}
    video_frame_indexes = {"recA": np.arange(10), "recB": np.arange(10) + 4}

    frames, thread = calibration._start_loading_frames(sample_keys, video_paths, video_frame_indexes)
    thread.join(timeout=60)

    assert list(frames) == sample_keys
    for key, path, frame_ix in [
        (sample_keys[0], "videos/a.mp4", 2),
        (sample_keys[1], "videos/b.mp4", 7),
    ]:
        recording, frame, _ = key
        message = frames[key]
        assert isinstance(message, str)
        assert f"video frame {frame_ix} of {path}" in message
        assert f"frame {frame} of the keypoints for recording {recording}" in message
        assert f"ValueError: error decoding frame {frame_ix} from video {path}" in message
        assert "check this video" in message
    np.testing.assert_array_equal(frames[sample_keys[2]], np.full((4, 6, 3), 5, np.uint8))


SAMPLE_KEYS = [("rec", 0, "a"), ("rec", 1, "b"), ("rec", 2, "a")]


def synthetic_widget(project_dir, frames):
    """The calibration widget for three frames of one recording, showing the
    frames read so far: `frames` maps sample keys to 40 x 60 images or to
    read-error messages."""
    coordinates = {"rec": np.tile([[10.0, 20.0], [30.0, 15.0]], (3, 1, 1))}
    widget = calibration._noise_calibration_widget(
        str(project_dir),
        coordinates,
        {"rec": np.ones((3, 2))},
        SAMPLE_KEYS,
        frames,
        bodyparts=["a", "b"],
        video_frame_indexes={"rec": np.arange(3)},
        error_estimator={"slope": -0.5, "intercept": 0.25},
        conf_threshold=0.5,
    )
    return widget, plt.gcf()


def frame_image(value):
    return np.full((40, 60, 3), value, np.uint8)


def visible_texts(fig):
    """The figure's visible texts without their whitespace, which wrapping
    changes."""
    return ["".join(t.get_text().split()) for t in fig.axes[0].texts if t.get_visible()]


def assert_shows_message(widget, fig, message, bodypart, completed=0):
    """The figure shows a uniform image the size of the frames before it,
    the frame's limits, no markers and `message`, wrapped to at most 60
    characters a line; the label names the bodypart."""
    ax = fig.axes[0]
    (image,) = [im for im in ax.images if im.get_visible()]
    array = np.asarray(image.get_array())
    assert array.shape[:2] == (40, 60)
    assert (array == array.flat[0]).all()
    assert ax.get_xlim() == (-0.5, 59.5)
    assert ax.get_ylim() == (39.5, -0.5)
    assert all(len(c.get_offsets()) == 0 for c in ax.collections if c.get_visible())
    assert visible_texts(fig) == ["".join(message.split())]
    (text,) = [t.get_text() for t in ax.texts if t.get_visible()]
    assert max(len(line) for line in text.split("\n")) <= 60
    info_label, _ = widget.children[1].children
    assert info_label.value == f"Target bodypart = {bodypart} | Completed annotations = {completed}"


def assert_shows_frame(fig, value, detected):
    ax = fig.axes[0]
    (image,) = [im for im in ax.images if im.get_visible()]
    np.testing.assert_array_equal(image.get_array(), frame_image(value))
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [detected])
    assert visible_texts(fig) == []


def test_widget_shows_a_frame_once_it_has_been_read(tmp_path, monkeypatch):
    """A frame still being read shows a placeholder and ignores clicks until
    the timer finds it read. Leaving it stops the timer."""
    timers = fake_timers(monkeypatch)
    frames = {SAMPLE_KEYS[0]: frame_image(50)}
    widget, fig = synthetic_widget(tmp_path, frames)
    (timer,) = timers
    prev_button, next_button = widget.children[0].children
    assert_shows_frame(fig, 50, (10, 20))
    assert not timer.running

    next_button.click()
    assert_shows_message(widget, fig, "Loading frame…", "b")
    assert timer.running
    click(fig, 30, 15)
    assert_shows_message(widget, fig, "Loading frame…", "b")
    timer.tick()
    assert_shows_message(widget, fig, "Loading frame…", "b")

    frames[SAMPLE_KEYS[1]] = frame_image(100)
    timer.tick()
    assert_shows_frame(fig, 100, (30, 15))
    assert not timer.running
    click(fig, 31, 16)
    info_label, _ = widget.children[1].children
    assert info_label.value.endswith("Completed annotations = 1")

    next_button.click()
    assert timer.running
    prev_button.click()
    assert_shows_frame(fig, 100, (30, 15))
    assert not timer.running


def test_widget_shows_a_read_error_and_goes_on_to_the_next_frame(tmp_path, monkeypatch, capsys):
    """The error is shown in the figure, with a path longer than a line
    broken across lines, and printed in full where it can be copied."""
    fake_timers(monkeypatch)
    path = "/data/" + "a_long_directory_name_without_spaces/" * 3 + "rec.mp4"
    error = f"Could not read video frame 1 of {path} (frame 1 of the keypoints ...)"
    frames = {SAMPLE_KEYS[0]: frame_image(50), SAMPLE_KEYS[1]: error, SAMPLE_KEYS[2]: frame_image(150)}
    widget, fig = synthetic_widget(tmp_path, frames)
    prev_button, next_button = widget.children[0].children
    _, usr_msg = widget.children[1].children
    saving = "Annotations not saved: complete 20 more annotations to start auto-saving"
    assert usr_msg.value == saving

    capsys.readouterr()
    next_button.click()
    assert_shows_message(widget, fig, error, "b")
    assert error in capsys.readouterr().out
    assert usr_msg.value == "Could not read this frame: see the message in the figure."
    click(fig, 30, 15)
    assert_shows_message(widget, fig, error, "b")

    next_button.click()
    assert_shows_frame(fig, 150, (10, 20))
    assert usr_msg.value == saving
