"""kpms keeps JAX's compiled programs in a persistent cache across sessions.

Importing keypoint_moseq points JAX's compilation cache at a ``jax`` folder in
the platform's user cache directory (``platformdirs.user_cache_dir
("keypoint_moseq")``) and caches every compiled program, however quick to
compile, unless the user has set a minimum compile time. If the user has
chosen a cache directory, through ``JAX_COMPILATION_CACHE_DIR`` or
``jax.config``, kpms changes none of the cache settings.

Each test imports the keypoint_moseq under test in a new Python process, run
from a temporary directory, with every location platformdirs reads (HOME,
XDG_CACHE_HOME, LOCALAPPDATA) in a temporary directory, so the real cache is
not touched.
"""

import json
import os
import subprocess
import sys

import keypoint_moseq

REPORT = """
import json
import jax
{before_import}
before = (jax.config.jax_compilation_cache_dir,
          jax.config.jax_persistent_cache_min_compile_time_secs)
import keypoint_moseq
import platformdirs
jax.jit(lambda x: x + 1)(1.0)
print(json.dumps({{
    "before": before,
    "cache_dir": jax.config.jax_compilation_cache_dir,
    "min_compile_time": jax.config.jax_persistent_cache_min_compile_time_secs,
    "user_cache_dir": platformdirs.user_cache_dir("keypoint_moseq"),
    "kpms_file": keypoint_moseq.__file__,
}}))
"""


def import_kpms(tmp_path, before_import="", **env_vars):
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        LOCALAPPDATA=str(tmp_path / "localappdata"),
    )
    for name in ["JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"]:
        env.pop(name, None)
    # the keypoint_moseq under test, wherever it is imported from
    kpms_parent = os.path.dirname(os.path.dirname(keypoint_moseq.__file__))
    env["PYTHONPATH"] = os.pathsep.join([kpms_parent, env.get("PYTHONPATH", "")])
    env.update(env_vars)
    script = REPORT.format(before_import=before_import)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["kpms_file"] == keypoint_moseq.__file__
    assert report["user_cache_dir"].startswith(str(tmp_path))
    return report


def test_import_caches_every_compile_in_the_user_cache_directory(tmp_path):
    report = import_kpms(tmp_path)

    expected = os.path.join(report["user_cache_dir"], "jax")
    assert report["cache_dir"] == expected
    assert report["min_compile_time"] == 0
    assert os.listdir(expected)  # the program compiled after the import


def test_a_minimum_compile_time_from_the_environment_is_kept(tmp_path):
    report = import_kpms(tmp_path, JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="5")

    assert report["cache_dir"] == os.path.join(report["user_cache_dir"], "jax")
    assert report["min_compile_time"] == 5


def test_a_minimum_compile_time_set_in_jax_config_is_kept(tmp_path):
    set_time = "jax.config.update('jax_persistent_cache_min_compile_time_secs', 5.0)"
    report = import_kpms(tmp_path, before_import=set_time)

    assert report["cache_dir"] == os.path.join(report["user_cache_dir"], "jax")
    assert report["min_compile_time"] == 5


def test_a_cache_directory_from_the_environment_is_left_alone(tmp_path):
    mine = str(tmp_path / "mine")
    report = import_kpms(tmp_path, JAX_COMPILATION_CACHE_DIR=mine)

    assert report["before"][0] == mine
    assert [report["cache_dir"], report["min_compile_time"]] == report["before"]
    assert not os.path.exists(report["user_cache_dir"])


def test_a_cache_directory_set_in_jax_config_is_left_alone(tmp_path):
    mine = str(tmp_path / "mine")
    set_dir = f"jax.config.update('jax_compilation_cache_dir', {mine!r})"
    report = import_kpms(tmp_path, before_import=set_dir)

    assert report["before"][0] == mine
    assert [report["cache_dir"], report["min_compile_time"]] == report["before"]
    assert not os.path.exists(report["user_cache_dir"])
