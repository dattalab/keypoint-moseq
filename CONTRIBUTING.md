# Contributing to keypoint-MoSeq

This page covers how changes get into the repository and how to run and extend the test suite.

## How changes get merged

All changes reach `main` through a pull request; direct pushes to `main` are blocked. A pull request can be merged once its `tests` check passes (see [Which Python versions are tested](#which-python-versions-are-tested)), and is merged by a maintainer with write access to the repository. For a first-time contributor's pull request, a maintainer approves the test run before it starts.

Small, focused pull requests are reviewed fastest. For larger changes, open an issue first to discuss the approach.

## Running the tests

The test suite runs the steps of the modeling tutorial on the tutorial dataset. With [uv](https://docs.astral.sh/uv/) installed, run it from the repository root for one Python version:

```bash
uvx nox -s tests-3.10
```

This builds a fresh Python 3.10 environment, installs keypoint-MoSeq with its `test` dependencies, and runs pytest. The first run downloads the tutorial dataset (about 300 MB) from Zenodo and caches it in your user cache directory (`~/.cache/keypoint-moseq` on Linux). The full suite takes about 8 to 11 minutes per Python version on a CPU.

```bash
uvx nox -s tests-3.10 tests-3.14              # the versions a pull request is tested on
uvx nox                                       # every supported version, one after another
uvx nox -r -s tests-3.10                      # reuse the existing environment
uvx nox -r -s tests-3.10 -- tests/test_pca.py # run one test file
uvx nox -r -s tests-3.10 -- -k checkpoint     # run tests whose names match
```

nox uses uv to find or download each Python version, so keep uv current (`uv self update`). An old uv may only know a pre-release of the newest Python and silently test against it.

## Which Python versions are tested

keypoint-MoSeq supports Python 3.10 through 3.14. The list lives in `PYTHON_VERSIONS` in `noxfile.py`, and each version has a nox session named `tests-<version>`. Every test run installs the newest release of each dependency that the requirements in `setup.cfg` allow, and the newest releases differ by Python version: Python 3.10, for example, gets pandas 2 and an older NumPy, while later versions get pandas 3 and newer NumPy and matplotlib releases.

| When | Python versions | Required to merge |
|---|---|---|
| Every push to a pull request | 3.10 and 3.14 | Yes |
| Every push to `main` | 3.10, 3.11, 3.12, 3.13, 3.14 | No |
| Nightly, and on demand from the Actions tab | 3.10, 3.11, 3.12, 3.13, 3.14 | No |

A pull request is tested on the oldest and newest supported versions because those bracket the range of dependency releases. Each version runs as its own `pytest (python <version>)` job, in parallel, and the `tests` check passes only when every one of them passes, so a pull request needs the full suite to pass on both 3.10 and 3.14. Before opening one, run those two sessions locally. A failure on `main` or in the nightly run that a pull request did not show usually means a newly released dependency broke something on one Python version; it is fixed in its own pull request.

The test suite runs on Linux. On macOS and Windows, every run instead installs keypoint-MoSeq from the checkout and imports it, on Python 3.10 and 3.14 (the `install-<version>` nox sessions, as `install (<os>, python <version>)` jobs), and the `tests` check requires those jobs to pass too. A requirement with no wheels for macOS or Windows makes `pip install keypoint-moseq` quietly install an older release that it can satisfy there, so these jobs catch it before a release does.

The tests run on a CPU. On a GPU, keypoint-MoSeq fits with parallel message passing, a code path that CPU runs do not exercise, so changes to model fitting should also be checked on a GPU with the `cuda` extra installed.

## What the tests check

Most tests record what keypoint-MoSeq does today, not what it should do. Their job is to catch changes nobody intended, such as a dependency update that shifts a result, so a passing test is not an endorsement of the behavior it checks. A failing test means behavior changed; whether that is a problem depends on whether the change was intended.

Known bugs are recorded the same way. A test marked `@pytest.mark.xfail(strict=True)` reproduces a crash or wrong result that exists today, and its `reason` names the bug. When a change fixes the bug, that test fails because the bug no longer occurs: remove the marker and make the test check the corrected behavior.

## Adding tests

Bug fixes and new features should come with a test. Most tests follow a step of the tutorial notebooks, `docs/source/modeling.ipynb` and `docs/source/analysis.ipynb`. The shared fixtures in `tests/conftest.py` run the notebook's steps in order on the tutorial dataset, so a new test can usually request the fixture for the step it needs and check that step's output.

## When a snapshot test fails

Many tests compare outputs with stored values in the `tests/test_*/` directories. The failure message lists which values differ and by how much. If your change is not meant to alter that output, the test has found something worth investigating. If it is, check that the reported differences are the ones you intended, then regenerate the stored values and run the test again:

```bash
uvx nox -r -s tests-3.10 -- --force-regen tests/test_pca.py
uvx nox -r -s tests-3.10 -- tests/test_pca.py
uvx nox -r -s tests-3.14 -- tests/test_pca.py
```

The stored values have to hold on every supported Python version, so after regenerating them on one version, run the test on the other. Numeric snapshots carry a tolerance for differences in the last few digits between NumPy releases. Where dependency releases legitimately disagree, the test records every accepted form: text columns, for example, are stored as `object or str` because pandas 2 and pandas 3 give them different dtypes.

Commit the regenerated files together with the code change.

## Changes that affect the tutorial

`docs/source/modeling.ipynb` defines how keypoint-MoSeq is meant to be run. If a change affects that workflow, update the notebook and its Colab copy, `docs/keypoint_moseq_colab.ipynb`, in the same pull request.

## Version numbers

Maintainers choose the version number when they publish a release, based on what changes for someone who upgrades. A patch release (`0.7.0` → `0.7.1`) leaves every output unchanged for the same dependency versions: syllables and fitted parameters, PCA, dataframes, saved files and figures. Patch releases hold crash fixes, documentation, performance improvements with identical output, and tightened dependency requirements that exclude a broken upstream release. Anything that changes outputs, including a bug fix that changes results people already have, is a minor release (`0.7.0` → `0.8.0`), as are new features, deprecations and other changes to the dependency requirements. While the version is 0.x, API changes and changes that stop older projects from loading are minor releases too, called out in the release notes.

If your pull request changes outputs, or needed a stored snapshot regenerated, say so in its description.

Each tutorial notebook (`docs/source/modeling.ipynb`, `docs/source/analysis.ipynb` and the Colab copy) calls `kpms.check_installation(notebook_version="X.Y")` near the top, which raises when the installed keypoint-MoSeq is a different minor version than the notebook states. Before publishing a minor release, set `notebook_version` in all three notebooks to the new version in a pull request. The release workflow checks this before uploading to PyPI and stops if any notebook states another version. GitHub has already created the tag by then, so after merging the fix, delete the release and its tag (`git push --delete origin <tag>`) and publish the release again from the updated `main`. Patch releases keep the same `notebook_version`.

The documentation's notebook links and the Colab links point at the `stable` branch, and Read the Docs builds the stable documentation from it. After the PyPI upload, the release workflow moves `stable` to the release commit, so those links always open the notebooks of the newest release on PyPI. Pre-releases such as `0.9.0b1` leave `stable` where it is, because `pip install keypoint-moseq` does not install them. Changes reach `stable` only through a release from `main`: a commit made directly on `stable` stops the workflow from moving it forward.

## Reporting problems

Open an issue at https://github.com/dattalab/keypoint-moseq/issues. Include the keypoint-MoSeq version, your Python version, your operating system, your GPU and CUDA version if you use one, and the full error message.
