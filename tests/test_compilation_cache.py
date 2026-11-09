"""kpms keeps JAX's compiled programs in a persistent cache across sessions.

Importing keypoint_moseq points JAX's compilation cache at a folder named for
the compression JAX applies to cache entries (``zstd`` or ``zlib``), inside a
``jax`` folder in the platform's user cache directory
(``platformdirs.user_cache_dir("keypoint_moseq")``), and caches every compiled
program, however quick to compile, unless the user has set a minimum compile
time. If the user has
chosen a cache directory, through ``JAX_COMPILATION_CACHE_DIR`` or
``jax.config``, kpms changes none of the cache settings.

Each test imports the keypoint_moseq under test in a new Python process, run
from a temporary directory, with every location platformdirs reads (HOME,
XDG_CACHE_HOME, LOCALAPPDATA) in a temporary directory, so the real cache is
not touched. The test suite itself compiles with the persistent cache off
(conftest.py).
"""

import json
import os
import subprocess
import sys

import jax
import keypoint_moseq

REPORT = """
import json
{before_jax}
import jax
from jax._src import compilation_cache
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
    # zstd frames start with the bytes 28 b5 2f fd
    "compression": "zstd" if compilation_cache.compress_executable(b"x")[:4]
                   == bytes.fromhex("28b52ffd") else "zlib",
}}))
"""


# Python without compression.zstd (3.14+) or the zstandard package
WITHOUT_ZSTD = """
import sys
sys.modules["compression.zstd"] = None
sys.modules["zstandard"] = None
"""


def import_kpms(tmp_path, before_import="", before_jax="", **env_vars):
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
    script = REPORT.format(before_import=before_import, before_jax=before_jax)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["kpms_file"] == keypoint_moseq.__file__
    assert report["user_cache_dir"].startswith(str(tmp_path))
    return report


def default_cache_dir(report):
    return os.path.join(report["user_cache_dir"], "jax", report["compression"])


def test_import_caches_every_compile_in_the_user_cache_directory(tmp_path):
    report = import_kpms(tmp_path)

    expected = default_cache_dir(report)
    assert report["cache_dir"] == expected
    assert report["min_compile_time"] == 0
    assert os.listdir(expected)  # the program compiled after the import


def test_without_zstd_the_cache_is_in_a_zlib_folder(tmp_path):
    report = import_kpms(tmp_path, before_jax=WITHOUT_ZSTD)

    assert report["compression"] == "zlib"
    expected = os.path.join(report["user_cache_dir"], "jax", "zlib")
    assert report["cache_dir"] == expected
    assert os.listdir(expected)


def test_a_minimum_compile_time_from_the_environment_is_kept(tmp_path):
    report = import_kpms(tmp_path, JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="5")

    assert report["cache_dir"] == default_cache_dir(report)
    assert report["min_compile_time"] == 5


def test_a_minimum_compile_time_set_in_jax_config_is_kept(tmp_path):
    set_time = "jax.config.update('jax_persistent_cache_min_compile_time_secs', 5.0)"
    report = import_kpms(tmp_path, before_import=set_time)

    assert report["cache_dir"] == default_cache_dir(report)
    assert report["min_compile_time"] == 5


def test_the_test_suite_compiles_without_the_persistent_cache():
    assert not jax.config.jax_enable_compilation_cache


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
