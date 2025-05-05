import numpy as np
import tqdm
import os
from textwrap import fill
from vidio.read import OpenCVReader
from keypoint_moseq.io import update_config
from keypoint_moseq.util import find_matching_videos, get_edges


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

    sample_keys = []
    for low, high in zip(thresholds[:-1], thresholds[1:]):
        samples_in_bin = []
        for key, confs in confidences.items():
            for t, k in zip(*np.nonzero((confs >= low) * (confs < high) * mask)):
                samples_in_bin.append((key, t, bodyparts[k]))

        if len(samples_in_bin) > 0:
            n = min(num_samples // num_bins, len(samples_in_bin))
            for i in np.random.choice(len(samples_in_bin), n, replace=False):
                sample_keys.append(samples_in_bin[i])

    sample_keys = [sample_keys[i] for i in np.random.permutation(len(sample_keys))]
    return sample_keys


def load_sampled_frames(
    sample_keys,
    video_dir,
    video_frame_indexes,
    video_extension=None,
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

    Returns
    -------
    sample_keys: dict
        Dictionary mapping elements from `sample_keys` to the
        corresponding videos frames.
    """
    keys = sorted(set([k[0] for k in sample_keys]))
    videos = find_matching_videos(keys, video_dir, video_extension=video_extension)
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


def save_params(project_dir, estimator):
    """Save config parameters learned via calibration.

    Parameters
    ----------
    project_dir: str
        Save parameters `{project_dir}/config.yml`

    estimator: dict
        Dictionary containing calibration parameters with keys:
        - conf_threshold: float, confidence threshold for outlier detection
        - slope: float, slope of error vs confidence regression line
        - intercept: float, intercept of error vs confidence regression line
    """
    update_config(
        project_dir,
        conf_threshold=float(estimator["conf_threshold"]),
        slope=float(estimator["slope"]),
        intercept=float(estimator["intercept"]),
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
    current_annotation_marker = [None]
    annotations = {}

    next_button = Button(description="Next")
    prev_button = Button(description="Prev")
    info_label = Label(
        f"Target bodypart = {current_img_key[0][2]} | Completed annotations = 0",
        layout={"margin": "0px"},
    )
    usr_msg = Label(
        f"Annotations not saved: complete {required_annotations} more annotations to start auto-saving",
        layout={"margin": "0px"},
    )
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

    def update_info_label():
        bodypart = current_img_key[0][2]
        info_label.value = (
            f"Target bodypart = {bodypart} | Completed annotations = {len(annotations)}"
        )

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
        error_estimator["slope"] = slope
        error_estimator["intercept"] = intercept
        error_estimator["conf_threshold"] = conf_threshold

        save_annotations(project_dir, annotations, video_frame_indexes)
        usr_msg.value = f"Annotations saved to {project_dir}/error_annotations.csv"
        save_params(project_dir, error_estimator)

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            # Check for and remove existing annotation marker
            if current_annotation_marker[0] is not None:
                current_annotation_marker[0].remove()
                current_annotation_marker[0] = None

            annotations[current_img_key[0]] = (event.xdata, event.ydata)
            current_annotation_marker[0] = ax.scatter(
                event.xdata, event.ydata, color="red", marker="x"
            )
            fig.canvas.draw()
            update_info_label()

            # Check if we have enough annotations to save
            if len(annotations) >= required_annotations:
                save_annotations_data()
            else:
                remaining = required_annotations - len(annotations)
                usr_msg.value = f"Annotations not saved: complete {remaining} more annotations to start auto-saving"

    fig.canvas.mpl_connect("button_press_event", onclick)

    def show_image(image_key):
        with output:
            output.clear_output(wait=True)
            ax.clear()
            ax.imshow(sample_images[image_key])

            frame = image_key[1]
            bodypart_idx = bodyparts.index(image_key[2])
            video_coordinates = coordinates[image_key[0]]
            ax.scatter(
                video_coordinates[frame, bodypart_idx, 0],
                video_coordinates[frame, bodypart_idx, 1],
                color="yellow",
                marker="o",
                facecolor="none",
            )

            # If the user has already annotated this keypoint, plot it
            if image_key in annotations:
                current_annotation_marker[0] = ax.scatter(
                    annotations[image_key][0],
                    annotations[image_key][1],
                    color="red",
                    marker="x",
                )

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

        - It is suggested to annotate at least 50 frames, tracked by the 'annotations' counter.
        This counter includes saved annotations from previous sessions if you've run this
        widget on this project before.

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

    video_frame_indexes: dict, default-None
        Dictionary mapping recording names to arrays of video frame indexes.
        This is useful when the original keypoint coordinates used for modeling
        corresponded to a subset of frames from each video (i.e. if videos were
        trimmed or coordinates were downsampled).
    """

    if os.path.exists(os.path.join(project_dir, "error_annotations.csv")):
        response = input(
            "error_annotations.csv already exists. Continuing will overwrite the existing file (start noise calibration from scratch). Do you want to continue? (y/n)"
        )
        if response != "y":
            return
        else:
            os.remove(os.path.join(project_dir, "error_annotations.csv"))

    if video_frame_indexes is None:
        video_frame_indexes = {k: np.arange(len(v)) for k, v in coordinates.items()}
    else:
        assert set(video_frame_indexes.keys()) == set(
            coordinates.keys()
        ), "The keys of `video_frame_indexes` must match the keys of `results`"
        for k, v in coordinates.items():
            assert len(v) == len(video_frame_indexes[k]), (
                "There is a mismatch between the length of `video_frame_indexes` "
                f"and the length of `coordinates` results for key {k}."
                f"\n\tLength of video_frame_indexes = {len(video_frame_indexes[k])}"
                f"\n\tLength of coordinates = {len(v)}"
            )

    dim = list(coordinates.values())[0].shape[-1]
    assert dim == 2, "Calibration is only supported for 2D keypoints."

    confidences = {k: v + conf_pseudocount for k, v in confidences.items()}
    sample_keys = sample_error_frames(confidences, bodyparts, use_bodyparts)

    sample_images = load_sampled_frames(
        sample_keys,
        video_dir,
        video_frame_indexes,
        video_extension,
    )

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
