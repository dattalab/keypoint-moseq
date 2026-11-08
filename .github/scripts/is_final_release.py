"""Report whether a release tag names a final release.

The documentation's notebook links and the Colab links open notebooks from the
stable branch, and `pip install keypoint-moseq` never installs a pre-release.
The release workflow therefore points stable at a release only when it is
final, so those notebooks always match the version users install.

Usage: python is_final_release.py TAG

Prints "final=true" for a plain version number such as "0.8.1" or "v0.8.1",
and "final=false" for anything else, such as "0.9.0b1", "0.9.0rc1" or
"0.9.0.dev0", in the format of a GitHub Actions step output.
"""

import re
import sys


def is_final(tag):
    """Whether a release tag is a version number with no pre-release part."""
    return re.fullmatch(r"v?\d+(\.\d+)*", tag) is not None


if __name__ == "__main__":
    print(f"final={str(is_final(sys.argv[1])).lower()}")
