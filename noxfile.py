import nox

nox.options.default_venv_backend = "uv"
# Plain `nox` runs the test suite; the install sessions are for CI on macOS
# and Windows.
nox.options.sessions = ["tests"]

# Every Python version keypoint-moseq supports. Sessions are named
# tests-<version>, e.g. `nox -s tests-3.12`; plain `nox` runs them all.
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Run the test suite against a fresh, non-editable install."""
    session.install(".[test]")
    session.run("uv", "pip", "freeze", external=True)
    session.run("pytest", *session.posargs, env={"MPLBACKEND": "Agg"})


@nox.session(python=PYTHON_VERSIONS)
def install(session):
    """Install keypoint-moseq without its test dependencies and import it.

    CI runs this on macOS and Windows, where the test suite does not run. A
    requirement with no wheels for one of those platforms then fails here,
    instead of making pip quietly fall back to an older keypoint-moseq.
    """
    session.install(".")
    session.run("uv", "pip", "freeze", external=True)
    session.run(
        "python", "-c", "import keypoint_moseq; print(keypoint_moseq.__version__)"
    )
