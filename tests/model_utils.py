"""Helpers for comparing keypoint-MoSeq model dictionaries in tests."""

import numpy as np


def plain(tree):
    """Nested dicts of arrays and scalars as plain Python values."""
    if isinstance(tree, dict):
        return {key: plain(value) for key, value in tree.items()}
    return np.asarray(tree).tolist()


def structure(tree):
    """Shape and dtype of every array in nested dicts."""
    if isinstance(tree, dict):
        return {str(key): structure(value) for key, value in tree.items()}
    array = np.asarray(tree)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def model_structure(model):
    return {
        "keys": sorted(model),
        "states": structure(model["states"]),
        "params": structure(model["params"]),
    }


def assert_tree_equal(actual, expected, path="model"):
    """Assert that nested dicts have the same keys and identical arrays."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert sorted(actual) == sorted(expected), path
        for key in expected:
            assert_tree_equal(actual[key], expected[key], f"{path}[{key!r}]")
    else:
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected), err_msg=path
        )
