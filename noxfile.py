import nox

nox.options.default_venv_backend = "uv"


@nox.session(python="3.10")
def tests(session):
    """Run the test suite against a fresh, non-editable install."""
    session.install(".[test]")
    session.run("pytest", *session.posargs, env={"MPLBACKEND": "Agg"})
