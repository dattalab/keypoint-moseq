import nox

nox.options.default_venv_backend = "uv"

# Every Python version keypoint-moseq supports. Sessions are named
# tests-<version>, e.g. `nox -s tests-3.12`; plain `nox` runs them all.
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Run the test suite against a fresh, non-editable install."""
    session.install(".[test]")
    session.run("uv", "pip", "freeze", external=True)
    session.run("pytest", *session.posargs, env={"MPLBACKEND": "Agg"})
