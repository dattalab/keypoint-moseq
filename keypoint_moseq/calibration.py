import numpy as np
import tqdm
import os
import threading
import warnings
import matplotlib
from textwrap import fill
from vidio.read import OpenCVReader
from keypoint_moseq.io import update_config
from keypoint_moseq.util import check_videos, find_matching_videos, get_edges


def _draw_distinct(count, n):
    """`n` distinct integers from range(count), drawn uniformly with numpy's
    global random state by Floyd's algorithm. np.random.choice without
    replacement shuffles all of range(count), which is slow for a bin of
    millions of observations."""
    drawn = []
    for j in range(count - n, count):
        i = np.random.randint(j + 1, dtype=np.int64)
        drawn.append(j if i in drawn else i)
    return drawn


def sample_error_frames(
    confidences,
    bodyparts,
    use_bodyparts,
    num_bins=10,
    num_samples=100,
    conf_pseudocount=1e-3,
):
    """Randomly sample frames, enriching for those with low confidence keypoint
    detections.

    Parameters
    ----------
    confidences: dict
        Keypoint detection confidences for a collection of recordings

    bodyparts: list
        Label for each keypoint represented in `confidences`

    use_bodyparts: list
        Ordered subset of keypoint labels to be used for modeling

    num_bins: int, default=10
        Number of bins to use for enriching low-confidence keypoint
        detections. Confidence values for all used keypoints are
        divided into log-spaced bins and an equal number of instances
        are sampled from each bin.

    num_samples: int, default=100
        Total number of frames to sample

    conf_pseudocount: float, default=1e-3
        Pseudocount used to augment keypoint confidences.

    Returns
    -------
    sample_keys: list of tuples
        List of sampled frames as tuples with format
        (key, frame_number, bodypart)
    """
    confidences = {k: v + conf_pseudocount for k, v in confidences.items()}
    all_confs = np.concatenate([v.flatten() for v in confidences.values()])
    min_conf, max_conf = np.nanmin(all_confs), np.nanmax(all_confs)
    thresholds = np.logspace(np.log10(min_conf), np.log10(max_conf), num_bins + 1)
    mask = np.array([bp in use_bodyparts for bp in bodyparts])[None, :]

    # Every observation's bin, in one pass: bin b holds thresholds[b] <= conf
    # < thresholds[b + 1]; num_bins marks none (out of range, NaN or an unused
    # bodypart). Observations are ordered by recording, then frame and
    # bodypart, which is how a position maps back to its keypoint.
    keys = list(confidences.keys())
    num_keypoints = len(bodyparts)
    bins = np.concatenate(
        [
            (np.searchsorted(thresholds, confs, side="right") - 1).ravel()
            for confs in confidences.values()
        ]
    )
    used = np.tile(mask[0], len(bins) // num_keypoints)
    bins = np.where(used & (bins >= 0) & (bins < num_bins), bins, num_bins)
    # bins fit a small integer type, which numpy sorts by radix
    order = np.argsort(bins.astype(np.min_scalar_type(num_bins)), kind="stable")
    bin_starts = np.cumsum(np.bincount(bins, minlength=num_bins + 1))
    recording_starts = np.cumsum([0] + [confs.size for confs in confidences.values()])

    sample_keys = []
    for b in range(num_bins):
        start = bin_starts[b - 1] if b > 0 else 0
        count = bin_starts[b] - start
        if count > 0:
            n = min(num_samples // num_bins, count)
            for i in _draw_distinct(count, n):
                position = order[start + i]
                r = np.searchsorted(recording_starts, position, side="right") - 1
                t, k = divmod(position - recording_starts[r], num_keypoints)
                sample_keys.append((keys[r], t, bodyparts[k]))

    sample_keys = [sample_keys[i] for i in np.random.permutation(len(sample_keys))]
    return sample_keys


def load_sampled_frames(
    sample_keys,
    video_dir,
    video_frame_indexes,
    video_extension=None,
    video_paths=None,
    recording_name_suffix="",
    shared_videos=False,
):
    """Load sampled frames from a directory of videos.

    Parameters
    ----------
    sample_keys: list of tuples
        List of sampled frames as tuples with format
        (key, frame_number, bodypart)

    video_dir: str
        Path to directory containing videos

    video_frame_indexes: dict
        Dictionary mapping recording names to arrays of video frame indexes.
        This is useful when the original keypoint coordinates used for modeling
        corresponded to a subset of frames from each video (i.e. if videos were
        trimmed or coordinates were downsampled).

    video_extension: str, default=None
        Preferred video extension (passed to :py:func:`keypoint_moseq.util.find_matching_videos`)

    video_paths: dict, default=None
        Dictionary mapping recording names to video paths. If None, videos
        are matched to recordings in `video_dir`.

    recording_name_suffix: str, default=""
        See :py:func:`keypoint_moseq.util.find_matching_videos`.

    shared_videos: bool, default=False
        See :py:func:`keypoint_moseq.util.find_matching_videos`.

    Returns
    -------
    sample_images: dict
        Dictionary mapping elements from `sample_keys` to the
        corresponding video frames.
    """
    keys = sorted(set([k[0] for k in sample_keys]))
    if video_paths is None:
        videos = find_matching_videos(
            keys,
            video_dir,
            video_extension=video_extension,
            recording_name_suffix=recording_name_suffix,
            shared_videos=shared_videos,
        )
    else:
        videos = [video_paths[key] for key in keys]
    key_to_video = dict(zip(keys, videos))
    readers = {key: OpenCVReader(video) for key, video in zip(keys, videos)}
    pbar = tqdm.tqdm(
        sample_keys,
        desc="Loading sample frames",
        position=0,
        leave=True,
        ncols=72,
    )
    sampled_keys = {}
    for key, frame, bodypart in pbar:
        frame_ix = video_frame_indexes[key][frame]
        sampled_keys[(key, frame, bodypart)] = readers[key][frame_ix]
    return sampled_keys


def _start_loading_frames(sample_keys, video_paths, video_frame_indexes):
    """Read the first sampled frame, then start a thread that reads the rest
    in order.

    Returns a dict that fills in as frames are read, mapping each sample key
    to its frame, or to a message saying why it could not be read, and the
    thread.
    """
    readers = {}

    def read(sample_key):
        key, frame, _ = sample_key
        frame_ix = video_frame_indexes[key][frame]
        try:
            if key not in readers:
                readers[key] = OpenCVReader(video_paths[key])
            return readers[key][frame_ix]
        except Exception as error:
            return (
                f"Could not read video frame {frame_ix} of {video_paths[key]} "
                f"(frame {frame} of the keypoints for recording {key}). "
                f"{type(error).__name__}: {error}\n\n"
                "This frame is skipped: press Next to go on. Please check this "
                "video: a frame that cannot be read usually means the file is "
                "damaged or was not fully copied, and other frames may be "
                "affected too. Try opening it in a video player at that frame."
            )

    frames = {sample_keys[0]: read(sample_keys[0])}

    def read_the_rest():
        for sample_key in sample_keys[1:]:
            frames[sample_key] = read(sample_key)

    thread = threading.Thread(target=read_the_rest, daemon=True)
    thread.start()
    return frames, thread


def save_annotations(project_dir, annotations, video_frame_indexes):
    """Save calibration annotations to a csv file.

    Parameters
    ----------
    project_dir: str
        Save annotations to `{project_dir}/error_annotations.csv`

    annotations: dict
        Dictionary mapping sample keys to annotated keypoint
        coordinates. (See :py:func:`keypoint_moseq.calibration.sample_error_frames`
        for format of sample keys)

    video_frame_indexes: dict
        Dictionary mapping recording names to arrays of video frame indexes.
        This is useful when the original keypoint coordinates used for modeling
        corresponded to a subset of frames from each video (i.e. if videos were
        trimmed or coordinates were downsampled).
    """
    output = [
        "# key: recording name",
        "# coordinate_index: index of the keypoint data in coordinates (same as video_frame_index if every frame was used)",
        "# video_frame_index: frame number in the video",
        "# bodypart: name of the bodypart that was annotated",
        "# x: x-coordinate of the annotated keypoint",
        "# y: y-coordinate of the annotated keypoint",
        "key,coordinate_index,video_frame_index,bodypart,x,y",
    ]
    for (key, frame, bodypart), (x, y) in annotations.items():
        output.append(
            f"{key},{frame},{video_frame_indexes[key][frame]},{bodypart},{x},{y}"
        )
    path = os.path.join(project_dir, "error_annotations.csv")
    open(path, "w").write("\n".join(output))


def save_params(project_dir, slope, intercept, conf_threshold):
    """Save config parameters learned via calibration.

    Parameters
    ----------
    project_dir: str
        Save parameters to `{project_dir}/config.yml`

    slope: float
        Slope of error vs confidence regression line

    intercept: float
        Intercept of error vs confidence regression line

    conf_threshold: float
        Confidence threshold for outlier detection
    """
    update_config(
        project_dir,
        conf_threshold=float(conf_threshold),
        slope=float(slope),
        intercept=float(intercept),
    )


def _noise_calibration_widget(
    project_dir,
    coordinates,
    confidences,
    sample_keys,
    sample_images,
    *,
    bodyparts,
    video_frame_indexes,
    error_estimator,
    conf_threshold,
    **kwargs,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from ipywidgets import Button, Label, Output, HBox, VBox

    # Constant for required number of annotations before auto-saving
    required_annotations = 20

    num_images = len(sample_keys)
    current_img_idx = [0]
    current_img_key = [sample_keys[current_img_idx[0]]]
    annotations = {}

    next_button = Button(description="Next")
    prev_button = Button(description="Prev")
    info_label = Label(
        f"Target bodypart = {current_img_key[0][2]} | Completed annotations = 0",
        layout={"margin": "0px"},
    )
    usr_msg = Label(layout={"margin": "0px"})
    output = Output(layout={"margin": "0px", "padding": "0px"})

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.canvas.header_visible = False
    plt.subplots_adjust(top=1, bottom=0.01, left=0.07, right=0.99)
    ax.margins(y=0)
    ax.set_frame_on(False)
    ax.margins(y=0)
    pos = ax.get_position()
    right_shift = 0.1  # Avoids the toolbar overlapping the y-axis tick labels
    ax.set_position(
        [pos.x0 + right_shift, pos.y0, pos.width - right_shift, pos.height + pos.y0]
    )

    # One image and two markers, updated for each frame. The annotation cross
    # is animated: full redraws leave it out, save the rest of the figure as
    # the background, then add it, so a click only redraws the cross on top
    # of that background.
    ax.clear()
    frame_image = ax.imshow(np.full((1, 1, 3), 200, np.uint8))
    keypoint_marker = ax.scatter([], [], color="yellow", marker="o", facecolor="none")
    annotation_marker = ax.scatter([], [], color="red", marker="x", animated=True)
    # shown over a gray frame while a frame is being read, or if it can't be
    message = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center")
    background = [None]

    # Frames are read on a background thread. While the shown one is still
    # being read, a timer checks for it on the main thread, which draws.
    timer = fig.canvas.new_timer(interval=100)

    def show_when_read():
        if current_img_key[0] in sample_images:
            show_image(current_img_key[0])

    timer.add_callback(show_when_read)

    def on_draw(event):
        background[0] = fig.canvas.copy_from_bbox(fig.bbox)
        ax.draw_artist(annotation_marker)

    fig.canvas.mpl_connect("draw_event", on_draw)

    def show_annotation(image_key):
        xy = [annotations[image_key]] if image_key in annotations else np.empty((0, 2))
        annotation_marker.set_offsets(xy)

    def update_info_label():
        bodypart = current_img_key[0][2]
        info_label.value = (
            f"Target bodypart = {bodypart} | Completed annotations = {len(annotations)}"
        )

    def show_save_status():
        remaining = required_annotations - len(annotations)
        if remaining > 0:
            usr_msg.value = f"Annotations not saved: complete {remaining} more annotations to start auto-saving"
        else:
            usr_msg.value = f"Annotations saved to {project_dir}/error_annotations.csv"

    def save_annotations_data():
        # Get error and confidence values only for the coordinates that have been annotated
        errors = []
        confidences_annot = []

        for video, frame, bodypart in annotations.keys():
            bodypart_idx = bodyparts.index(bodypart)

            original_coordinates = coordinates[video][frame, bodypart_idx, :]
            annotated_coordinates = annotations[(video, frame, bodypart)]

            error = np.log10(
                np.sqrt(np.sum((original_coordinates - annotated_coordinates) ** 2)) + 1
            )
            confidence = np.log10(confidences[video][frame, bodypart_idx])

            errors.append(error)
            confidences_annot.append(confidence)

        # Fit a line to the annotated data with confidence as the x-axis and error as the y-axis
        # scipy.stats.linregress might be a little more clear but this avoid another import
        slope, intercept = np.polyfit(confidences_annot, errors, 1)
        error_estimator["slope"] = float(slope)
        error_estimator["intercept"] = float(intercept)

        save_annotations(project_dir, annotations, video_frame_indexes)
        usr_msg.value = f"Annotations saved to {project_dir}/error_annotations.csv"
        save_params(project_dir, slope, intercept, conf_threshold)

    def onclick(event):
        shown = isinstance(sample_images.get(current_img_key[0]), np.ndarray)
        if shown and event.xdata is not None and event.ydata is not None:
            annotations[current_img_key[0]] = (event.xdata, event.ydata)
            show_annotation(current_img_key[0])
            if background[0] is not None and fig.canvas.supports_blit:
                fig.canvas.restore_region(background[0])
                ax.draw_artist(annotation_marker)
                fig.canvas.blit(fig.bbox)
            else:
                fig.canvas.draw()
            update_info_label()

            # Check if we have enough annotations to save
            if len(annotations) >= required_annotations:
                save_annotations_data()
            else:
                show_save_status()

    fig.canvas.mpl_connect("button_press_event", onclick)

    def show_image(image_key):
        with output:
            output.clear_output(wait=True)
            if image_key in sample_images:
                timer.stop()
            else:
                timer.start()
            image = sample_images.get(image_key)
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
                frame_image.set_data(image)
                if image.ndim == 2:
                    frame_image.autoscale()
                # the whole frame, as imshow shows a new image, whatever the zoom
                frame_image.set_extent((-0.5, width - 0.5, height - 0.5, -0.5))
                ax.set_xlim(-0.5, width - 0.5)
                ax.set_ylim(height - 0.5, -0.5)

                frame = image_key[1]
                bodypart_idx = bodyparts.index(image_key[2])
                keypoint_marker.set_offsets([coordinates[image_key[0]][frame, bodypart_idx, :2]])
                show_annotation(image_key)
                message.set_visible(False)
                show_save_status()
            else:
                # a gray frame the size of the last one, so nothing moves
                height, width = frame_image.get_array().shape[:2]
                frame_image.set_data(np.full((height, width, 3), 200, np.uint8))
                keypoint_marker.set_offsets(np.empty((0, 2)))
                annotation_marker.set_offsets(np.empty((0, 2)))
                message.set_visible(True)
                if image is None:
                    message.set_text("Loading frame…")
                    show_save_status()
                else:
                    # wrapped to fit, breaking paths, and printed below to copy
                    paragraphs = image.split("\n\n")
                    message.set_text("\n\n".join(fill(p, 60, break_on_hyphens=False) for p in paragraphs))
                    print(image)
                    usr_msg.value = "Could not read this frame: see the message in the figure."

            fig.canvas.draw()

            update_info_label()

    def next_image(_):
        if current_img_idx[0] < num_images - 1:
            current_img_idx[0] += 1
            current_img_key[0] = sample_keys[current_img_idx[0]]
            show_image(current_img_key[0])

    def prev_image(_):
        if current_img_idx[0] > 0:
            current_img_idx[0] -= 1
            current_img_key[0] = sample_keys[current_img_idx[0]]
            show_image(current_img_key[0])

    next_button.on_click(next_image)
    prev_button.on_click(prev_image)

    show_image(current_img_key[0])

    controls = HBox([prev_button, next_button])
    msg_box = VBox([info_label, usr_msg])
    ui = VBox([controls, msg_box, output], layout={"margin": "0px", "padding": "0px"})
    return ui


def noise_calibration(
    project_dir,
    coordinates,
    confidences,
    *,
    bodyparts,
    use_bodyparts,
    video_dir,
    video_extension=None,
    conf_pseudocount=0.001,
    video_frame_indexes=None,
    **kwargs,
):
    """Perform manual annotation to calibrate the relationship between keypoint
    error and neural network confidence.

    This function creates a widget for interactive annotation in jupyter lab.
    Users mark correct keypoint locations for a sequence of frames, and a
    regression line is fit to the `log(confidence), log(error)` pairs obtained
    through annotation. The regression coefficients are used during modeling to
    set a prior on the noise level for each keypoint on each frame.

    Follow these steps to use the widget:
        - Run the cell below. A widget should appear with a video frame.
          The yellow marker denotes the automatically detected location of the bodypart.

        - Annotate each frame with the correct location of the labeled bodypart
            - Left click to specify the correct location - an "X" should appear.
            - Use the prev/next buttons to annotate additional frames.
            - Click and drag the bottom-right shaded corner of the widget to adjust image size.
            - Use the toolbar to the left of the figure to pan and zoom.

        - It is suggested to annotate at least 50 frames, tracked by the 'Completed annotations'
          counter. Each run starts from scratch: an existing `error_annotations.csv` is deleted
          after you confirm.

        - Annotations will be automatically saved once you've completed at least 20 annotations.
          Each new annotation after that will trigger an auto-save of all your work.
          The message at the top of the widget will indicate when your annotations are being saved.


    Parameters
    ----------
    project_dir: str
        Project directory. Must contain a `config.yml` file.

    coordinates: dict
        Keypoint coordinates for a collection of recordings. Values
        must be numpy arrays of shape (T,K,2) where K is the number
        of keypoints. Keys can be any unique str, but must start with
        the name of a videofile in `video_dir`.

    confidences: dict
        Nonnegative confidence values for the keypoints in
        `coordinates` as numpy arrays of shape (T,K).

    bodyparts: list
        Label for each keypoint represented in `coordinates`

    use_bodyparts: list
        Ordered subset of keypoint labels to be used for modeling

    video_dir: str
        Path to directory containing videos. Each video should
        correspond to a key in `coordinates`. The key must
        contain the videoname as a prefix.

    video_extension: str, default=None
        Preferred video extension (used in :py:func:`keypoint_moseq.util.find_matching_videos`)

    conf_pseudocount: float, default=0.001
        Pseudocount added to confidence values to avoid log(0) errors.

    video_frame_indexes: dict, default=None
        Dictionary mapping recording names to arrays of video frame indexes.
        This is useful when the original keypoint coordinates used for modeling
        corresponded to a subset of frames from each video (i.e. if videos were
        trimmed or coordinates were downsampled).

    kwargs
        Other config entries, normally passed with `**config()`.
        `error_estimator` and `conf_threshold` are required. `fps`,
        `recording_name_suffix` and `shared_videos` are passed to
        :py:func:`keypoint_moseq.util.check_videos`, which runs first and
        raises if the videos do not line up with the keypoints.
    """

    dim = list(coordinates.values())[0].shape[-1]
    assert dim == 2, "Calibration is only supported for 2D keypoints."

    # `%matplotlib widget` selects ipympl, which matplotlib reports as
    # "widget" or "ipympl", or by module path in older versions
    backend = matplotlib.get_backend()
    if backend.lower() not in ("widget", "ipympl", "module://ipympl.backend_nbagg"):
        warnings.warn(
            fill(
                "The calibration widget needs matplotlib's interactive ipympl "
                f"backend, but the current backend is '{backend}', so the widget "
                "will probably show a still image that ignores clicks. Put "
                "`%matplotlib widget` at the top of the cell that calls "
                "noise_calibration, as in the modeling notebook, and run the cell "
                "again."
            )
        )

    # Check the videos before the prompt below, which deletes any existing
    # annotations.
    video_paths = check_videos(
        coordinates,
        video_dir=video_dir,
        video_frame_indexes=video_frame_indexes,
        video_extension=video_extension,
        print_table=False,
        fps=kwargs.get("fps"),
        recording_name_suffix=kwargs.get("recording_name_suffix", ""),
        shared_videos=kwargs.get("shared_videos", False),
    )
    if video_frame_indexes is None:
        video_frame_indexes = {k: np.arange(len(v)) for k, v in coordinates.items()}

    if os.path.exists(os.path.join(project_dir, "error_annotations.csv")):
        response = input(
            "error_annotations.csv already exists. Continuing will overwrite the existing file (start noise calibration from scratch). Do you want to continue? (y/n)"
        )
        if response != "y":
            return
        else:
            os.remove(os.path.join(project_dir, "error_annotations.csv"))

    confidences = {k: v + conf_pseudocount for k, v in confidences.items()}
    sample_keys = sample_error_frames(confidences, bodyparts, use_bodyparts)
    sample_images, _ = _start_loading_frames(sample_keys, video_paths, video_frame_indexes)

    return _noise_calibration_widget(
        project_dir,
        coordinates,
        confidences,
        sample_keys,
        sample_images,
        bodyparts=bodyparts,
        video_frame_indexes=video_frame_indexes,
        **kwargs,
    )
