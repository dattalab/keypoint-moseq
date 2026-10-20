SNAPSHOT_RECORDING = (
    "21_12_10_def6a_3.top.irDLC_resnet50_moseq_exampleAug21shuffle1_500000"
)


def test_dlc_bodyparts_and_recordings(tutorial_keypoints, data_regression):
    coordinates, confidences, bodyparts = tutorial_keypoints

    assert sorted(confidences) == sorted(coordinates)
    data_regression.check({"bodyparts": bodyparts, "recordings": sorted(coordinates)})


def test_dlc_keypoints_for_one_recording(tutorial_keypoints, ndarrays_regression):
    coordinates, confidences, bodyparts = tutorial_keypoints

    ndarrays_regression.check(
        {
            "coordinates": coordinates[SNAPSHOT_RECORDING],
            "confidences": confidences[SNAPSHOT_RECORDING],
        }
    )
