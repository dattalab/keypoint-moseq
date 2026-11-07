"""Importing keypoint_moseq leaves JAX's precision as the session has it.

JAX runs in float32 unless 64-bit mode is turned on, and keypoint_moseq does
not change that setting, so a session runs in float64 only if the user turns
on ``jax_enable_x64`` themselves. Each import test imports the keypoint_moseq
under test in a new Python process with ``JAX_ENABLE_X64`` unset.
"""

import json
import os
import subprocess
import sys

import jax
import keypoint_moseq
import pytest

REPORT = """
import json
import jax
{before_import}
before = jax.config.read("jax_enable_x64")
import keypoint_moseq
print(json.dumps({{
    "before": before,
    "after": jax.config.read("jax_enable_x64"),
    "kpms_file": keypoint_moseq.__file__,
}}))
"""


def import_kpms(tmp_path, before_import=""):
    env = dict(os.environ)
    env.pop("JAX_ENABLE_X64", None)
    # the keypoint_moseq under test, wherever it is imported from
    kpms_parent = os.path.dirname(os.path.dirname(keypoint_moseq.__file__))
    env["PYTHONPATH"] = os.pathsep.join([kpms_parent, env.get("PYTHONPATH", "")])
    # programs compiled here stay out of the user's compilation cache
    env["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "jax_cache")
    script = REPORT.format(before_import=before_import)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["kpms_file"] == keypoint_moseq.__file__
    return report


@pytest.mark.parametrize(
    "before_import, x64",
    [("", False), ("jax.config.update('jax_enable_x64', True)", True)],
    ids=["default_float32", "x64_turned_on"],
)
def test_import_leaves_64_bit_mode_as_it_was(tmp_path, before_import, x64):
    report = import_kpms(tmp_path, before_import)

    assert report["before"] is x64
    assert report["after"] is x64


def test_the_test_suite_runs_in_float32():
    assert not jax.config.read("jax_enable_x64")
