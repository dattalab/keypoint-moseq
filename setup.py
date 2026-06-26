"""Minimal shim so versioneer can inject the dynamic version at build time.

All package metadata and dependencies live in ``pyproject.toml`` (PEP 621).
This file exists only because versioneer 0.29 has no standalone PEP 517 build
hook; it needs ``setup()`` to call ``versioneer.get_version()`` /
``versioneer.get_cmdclass()`` for the version to land in the built metadata.
"""

import versioneer
from setuptools import setup

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
