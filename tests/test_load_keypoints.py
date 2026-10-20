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
