"""The release workflow refuses to publish when a notebook's
`notebook_version` does not match the release tag."""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, ".github", "scripts", "check_notebook_versions.py")
NOTEBOOKS = [
    "docs/source/modeling.ipynb",
    "docs/source/analysis.ipynb",
    "docs/keypoint_moseq_colab.ipynb",
]


def notebook_versions():
    """The `notebook_version` passed in each notebook, read independently of
    the script under test."""
    versions = {}
    for path in NOTEBOOKS:
        with open(os.path.join(REPO, path)) as f:
            cells = json.load(f)["cells"]
        sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        (call,) = [s for s in sources if "check_installation(" in s]
        versions[path] = call.split('notebook_version="')[1].split('"')[0]
    return versions


def run(tag, repo=REPO):
    return subprocess.run(
        [sys.executable, SCRIPT, tag, repo], capture_output=True, text=True
    )


def test_every_notebook_states_the_same_version():
    assert len(set(notebook_versions().values())) == 1


@pytest.mark.parametrize("suffix", ["0", "3", "0beta", "0rc1"])
def test_tag_with_the_notebooks_minor_version_passes(suffix):
    minor = next(iter(notebook_versions().values()))

    result = run(f"{minor}.{suffix}")

    assert result.returncode == 0, result.stdout + result.stderr


def test_tag_with_a_leading_v_passes():
    minor = next(iter(notebook_versions().values()))

    assert run(f"v{minor}.0").returncode == 0


def test_tag_for_another_minor_version_fails():
    result = run("99.1.0")

    assert result.returncode != 0
    output = result.stdout + result.stderr
    for path in NOTEBOOKS:
        assert path in output
    assert "99.1" in output
    assert "delete" in output


def test_notebook_without_the_check_fails(tmp_path):
    for path in NOTEBOOKS:
        os.makedirs(tmp_path / os.path.dirname(path), exist_ok=True)
        with open(tmp_path / path, "w") as f:
            json.dump({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f)

    result = run("0.8.0", repo=str(tmp_path))

    assert result.returncode != 0
    assert "check_installation" in result.stdout + result.stderr
