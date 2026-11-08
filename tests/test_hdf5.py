import numpy as np

import keypoint_moseq as kpms


def test_lists_and_tuples_keep_their_order(tmp_path):
    """save_hdf5 stores each element of a list or tuple as a child named
    `arr{index}`. Taking the children in name order (arr0, arr1, arr10, ...)
    would load any list or tuple longer than 10 out of order."""
    path = str(tmp_path / "tree.h5")
    tree = {
        "arrays": [np.full(3, i) for i in range(12)],
        "names": tuple(f"recording{i}" for i in range(12)),
        "nested": {"pairs": [[i, i + 1] for i in range(11)]},
    }

    kpms.save_hdf5(path, tree)
    loaded = kpms.load_hdf5(path)

    assert isinstance(loaded["arrays"], list)
    assert len(loaded["arrays"]) == len(tree["arrays"])
    for actual, expected in zip(loaded["arrays"], tree["arrays"]):
        np.testing.assert_array_equal(actual, expected)
    assert loaded["names"] == tree["names"]
    assert loaded["nested"]["pairs"] == tree["nested"]["pairs"]
    assert kpms.load_hdf5(path, "names") == tree["names"]
