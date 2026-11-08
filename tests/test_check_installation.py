"""`check_installation` reports what is installed and raises when it finds a
problem that issues show users hit: a notebook written for another release,
a stale kernel, packages shadowed from the user site, mismatched JAX
packages, or a GPU that JAX cannot use.

The test suite runs on CPU. Where a check depends on something a CPU test
machine cannot have (an installed CUDA plugin, a released kpms version
rather than a git checkout), the tests change only what the installed
package metadata or `kpms.__version__` report. The GPU checks themselves are
exercised on a GPU.
"""

import os
import site
import sys
from importlib import metadata

import jax
import pytest

import keypoint_moseq as kpms
from keypoint_moseq.util import _explain_gpu_error

INSTALLED = kpms.__version__
ON_GPU = any(d.platform == "gpu" for d in jax.devices())
# The suite runs on CPU in CI; on a GPU machine, JAX sees the GPU, so the
# checks for a missing GPU do not apply.
without_gpu = pytest.mark.skipif(ON_GPU, reason="needs a machine where JAX sees no GPU")
with_gpu = pytest.mark.skipif(not ON_GPU, reason="needs a GPU")


def fake_versions(monkeypatch, **overrides):
    """Make the installed-package metadata report other versions for some
    distributions, e.g. `jaxlib="0.4.0"`; None means not installed."""
    real_version = metadata.version
    overrides = {name.replace("_", "-"): v for name, v in overrides.items()}

    def version(name):
        if name in overrides:
            if overrides[name] is None:
                raise metadata.PackageNotFoundError(name)
            return overrides[name]
        return real_version(name)

    monkeypatch.setattr(metadata, "version", version)


def release_install(monkeypatch, version):
    """Make the running kpms, and the package on disk, look like a release."""
    monkeypatch.setattr(kpms, "__version__", version)
    fake_versions(monkeypatch, keypoint_moseq=version)


def check_error(**kwargs):
    with pytest.raises(RuntimeError) as error:
        kpms.check_installation(**kwargs)
    return str(error.value)


def test_report_lists_versions_locations_and_devices(capsys):
    kpms.check_installation()

    out = capsys.readouterr().out
    assert INSTALLED in out
    assert jax.__version__ in out
    assert os.path.dirname(kpms.__file__) in out
    assert sys.executable in out
    assert f"{jax.devices()[0].platform}:0" in out


def test_notebook_for_the_installed_minor_version_passes(monkeypatch):
    release_install(monkeypatch, "0.8.2")

    kpms.check_installation(notebook_version="0.8")


def test_notebook_for_another_version_raises(monkeypatch):
    release_install(monkeypatch, "0.6.6")

    message = check_error(notebook_version="0.8")

    assert "0.8" in message
    assert "0.6.6" in message
    assert "pip install -U keypoint-moseq" in message
    assert "github.com/dattalab/keypoint-moseq/blob/0.6.6/docs/source/modeling.ipynb" in message


def test_development_install_skips_the_notebook_check(monkeypatch, capsys):
    release_install(monkeypatch, "0.7.1+4.gc871cae")

    kpms.check_installation(notebook_version="0.8")

    assert "development install" in capsys.readouterr().out


def test_stale_kernel_raises(monkeypatch):
    # The package on disk was upgraded after this kernel imported kpms.
    monkeypatch.setattr(kpms, "__version__", "0.8.0")
    fake_versions(monkeypatch, keypoint_moseq="0.8.1")

    message = check_error()

    assert "0.8.1" in message
    assert "0.8.0" in message
    assert "restart the kernel" in message


def test_development_install_skips_the_stale_kernel_check(monkeypatch):
    # An editable install computes its version from git when imported, while
    # the installed metadata keeps the version from install time, so the two
    # differ once a new tag is pulled. Restarting the kernel would not help.
    monkeypatch.setattr(kpms, "__version__", "0.8.0+3.gabc1234")
    fake_versions(monkeypatch, keypoint_moseq="0.7.1+5.gdef5678")

    kpms.check_installation()


def test_packages_shadowed_from_the_user_site_raise(monkeypatch, tmp_path):
    user_site = str(tmp_path / ".local" / "lib" / "site-packages")
    monkeypatch.setattr(site, "getusersitepackages", lambda: user_site)
    monkeypatch.setattr(jax, "__file__", os.path.join(user_site, "jax", "__init__.py"))

    message = check_error()

    assert "jax" in message
    assert user_site in message
    assert "PYTHONNOUSERSITE" in message


def test_mismatched_jax_and_jaxlib_raise(monkeypatch):
    fake_versions(monkeypatch, jaxlib="0.4.0")

    message = check_error()

    assert "jaxlib" in message
    assert "0.4.0" in message
    assert jax.__version__ in message


@without_gpu
def test_cuda_plugin_without_a_visible_gpu_reports_the_cpu(monkeypatch, capsys):
    # A GPU install can be used on a machine without a GPU, e.g. for the
    # analysis notebook, so this is reported but is not a problem.
    fake_versions(monkeypatch, jax_cuda12_plugin=jax.__version__)

    kpms.check_installation()

    out = capsys.readouterr().out
    assert "JAX is using the CPU" in out
    assert "If you expected a GPU here" in out
    assert "nvidia-smi" in out
    assert "CUDA_VISIBLE_DEVICES" in out


@without_gpu
def test_system_cuda_on_the_library_path_is_named_as_the_cause(monkeypatch, capsys):
    # Reproduced on an A100 node: a system CUDA toolkit first on
    # LD_LIBRARY_PATH stops JAX's CUDA plugin from starting, and JAX falls
    # back to the CPU, saying a CUDA-enabled jaxlib is not installed.
    fake_versions(monkeypatch, jax_cuda12_plugin=jax.__version__)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/n/app/cuda/12.8.0-gcc-14.2.0/lib64:/usr/lib")

    kpms.check_installation()

    expected = capsys.readouterr().out.split("If you expected a GPU here")[1]
    assert "/n/app/cuda/12.8.0-gcc-14.2.0/lib64" in expected
    assert "module unload" in expected


@without_gpu
def test_cpu_only_install_without_cuda_plugin_passes(monkeypatch, capsys):
    fake_versions(monkeypatch, jax_cuda12_plugin=None)

    kpms.check_installation()

    out = capsys.readouterr().out
    assert "JAX is using the CPU" in out
    assert "keypoint-moseq[cuda]" in out
    assert "If you expected a GPU here" not in out


@with_gpu
def test_gpu_test_runs_on_a_visible_gpu(capsys):
    kpms.check_installation()

    assert "GPU test: passed" in capsys.readouterr().out


def test_every_problem_is_listed_and_the_report_is_printed_first(monkeypatch, capsys):
    monkeypatch.setattr(kpms, "__version__", "0.8.0")
    fake_versions(monkeypatch, keypoint_moseq="0.8.1", jaxlib="0.4.0")

    message = check_error()

    assert "restart the kernel" in message
    assert "jaxlib" in message
    assert jax.__version__ in capsys.readouterr().out


def test_report_is_printed_when_jax_cannot_list_its_devices(monkeypatch, capsys):
    # e.g. JAX_PLATFORMS=cuda on a machine where the CUDA backend cannot start
    def no_devices():
        raise RuntimeError("Unable to initialize backend 'cuda'")

    monkeypatch.setattr(jax, "devices", no_devices)

    message = check_error()

    assert "Unable to initialize backend 'cuda'" in message
    out = capsys.readouterr().out
    assert jax.__version__ in out
    assert sys.executable in out


@pytest.mark.parametrize(
    "error, expected",
    [
        # #150, #158
        (
            "XlaRuntimeError: INTERNAL: no kernel image is available for execution on the device",
            "driver",
        ),
        # #69, #103, #107
        ("XlaRuntimeError: INTERNAL: cuSolver internal error", "LD_LIBRARY_PATH"),
        # #78
        (
            "XlaRuntimeError: INTERNAL: Unable to initialize cuBLAS: CUBLAS_STATUS_NOT_INITIALIZED",
            "LD_LIBRARY_PATH",
        ),
        # #39, #155
        ("Could not load dynamic library 'libcudart.so.11.0'", "LD_LIBRARY_PATH"),
        (
            "XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4.0KiB.",
            "another",
        ),
    ],
)
def test_known_gpu_errors_are_explained(error, expected):
    assert expected in _explain_gpu_error(error)


def test_unknown_gpu_errors_keep_their_message():
    assert "some new failure" in _explain_gpu_error("INTERNAL: some new failure")
