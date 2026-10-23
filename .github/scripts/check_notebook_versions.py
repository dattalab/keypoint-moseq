"""Check that every tutorial notebook states the version being released.

Each notebook calls `kpms.check_installation(notebook_version="X.Y")`, which
raises when the installed keypoint-moseq is a different minor version. The
release workflow runs this script before publishing to PyPI, so a release
cannot ship notebooks that state another version.

Usage: python check_notebook_versions.py TAG [REPOSITORY_DIRECTORY]
"""

import json
import os
import re
import sys

NOTEBOOKS = [
    "docs/source/modeling.ipynb",
    "docs/source/analysis.ipynb",
    "docs/keypoint_moseq_colab.ipynb",
]
CALL = re.compile(r'check_installation\(\s*notebook_version\s*=\s*"([^"]+)"')


def release_minor(tag):
    """The major.minor version of a release tag such as "0.8.0" or "v0.8.0b1"."""
    match = re.match(r"v?(\d+)\.(\d+)", tag)
    if match is None:
        sys.exit(f"Cannot read a version from the release tag {tag!r}.")
    return f"{match.group(1)}.{match.group(2)}"


def stated_versions(path):
    """Every notebook_version passed to check_installation in a notebook."""
    with open(path) as f:
        cells = json.load(f)["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    return [m.group(1) for source in sources for m in CALL.finditer(source)]


def main(tag, repository="."):
    expected = release_minor(tag)
    problems = []
    for notebook in NOTEBOOKS:
        versions = stated_versions(os.path.join(repository, notebook))
        if not versions:
            problems.append(f"  {notebook}: no check_installation(notebook_version=...) call")
        elif any(v != expected for v in versions):
            stated = ", ".join(f'notebook_version="{v}"' for v in versions)
            problems.append(f"  {notebook}: {stated}")
        else:
            print(f'{notebook}: notebook_version="{expected}"')
    if problems:
        print(
            f'Release {tag} needs notebook_version="{expected}" in every notebook, but:',
            *problems,
            "",
            "Nothing was published to PyPI. To release: merge a pull request that sets",
            f'notebook_version="{expected}" in each notebook, delete this GitHub release',
            f"and its tag (git push --delete origin {tag}), then publish the release",
            "again from the updated main branch.",
            sep="\n",
        )
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:])
