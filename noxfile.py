import nox

@nox.session(python=['3.10'])
def tests_cpu(session):
    session.install('-e', '.[dev]')
    session.run('pytest', '-q')

@nox.session(python=['3.10'])
def tests_cuda(session):
    session.install('-e', '.[dev,cuda]')
    session.run('pytest', '-q')