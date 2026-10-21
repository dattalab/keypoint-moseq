# Contributing to keypoint-MoSeq

This page covers how changes get into the repository and how to run and extend the test suite.

## How changes get merged

All changes reach `main` through a pull request; direct pushes to `main` are blocked. A pull request can be merged once its `tests` check passes, and is merged by a maintainer with write access to the repository. For a first-time contributor's pull request, a maintainer approves the test run before it starts.

Small, focused pull requests are reviewed fastest. For larger changes, open an issue first to discuss the approach.

## Running the tests

The test suite runs the steps of the modeling tutorial on the tutorial dataset. With [uv](https://docs.astral.sh/uv/) installed, run it from the repository root:

```bash
uvx nox
```

This builds a fresh Python 3.10 environment, installs keypoint-MoSeq with its `test` dependencies, and runs pytest. The first run downloads the tutorial dataset (about 300 MB) from Zenodo and caches it in your user cache directory (`~/.cache/keypoint-moseq` on Linux). The full suite takes about 8 minutes on a CPU.

```bash
uvx nox -r                          # reuse the existing environment
uvx nox -r -- tests/test_pca.py     # run one test file
uvx nox -r -- -k checkpoint         # run tests whose names match
```

The same suite runs on every pull request and nightly against the newest allowed version of every dependency.

## What the tests check

Most tests record what keypoint-MoSeq does today, not what it should do. Their job is to catch changes nobody intended, such as a dependency update that shifts a result, so a passing test is not an endorsement of the behavior it checks. A failing test means behavior changed; whether that is a problem depends on whether the change was intended.

Known bugs are recorded the same way. A test marked `@pytest.mark.xfail(strict=True)` reproduces a crash or wrong result that exists today, and its `reason` names the bug. When a change fixes the bug, that test fails because the bug no longer occurs: remove the marker and make the test check the corrected behavior.

## Adding tests

Bug fixes and new features should come with a test. Most tests follow a step of the tutorial notebooks, `docs/source/modeling.ipynb` and `docs/source/analysis.ipynb`. The shared fixtures in `tests/conftest.py` run the notebook's steps in order on the tutorial dataset, so a new test can usually request the fixture for the step it needs and check that step's output.

## When a snapshot test fails

Many tests compare outputs with stored values in the `tests/test_*/` directories. The failure message lists which values differ and by how much. If your change is not meant to alter that output, the test has found something worth investigating. If it is, check that the reported differences are the ones you intended, then regenerate the stored values and run the test again:

```bash
uvx nox -r -- --force-regen tests/test_pca.py
uvx nox -r -- tests/test_pca.py
```

Commit the regenerated files together with the code change.

## Changes that affect the tutorial

`docs/source/modeling.ipynb` defines how keypoint-MoSeq is meant to be run. If a change affects that workflow, update the notebook and its Colab copy, `docs/keypoint_moseq_colab.ipynb`, in the same pull request.

## Reporting problems

Open an issue at https://github.com/dattalab/keypoint-moseq/issues. Include the keypoint-MoSeq version, your operating system, your GPU and CUDA version if you use one, and the full error message.
