# -*- coding: utf-8 -*-
from invoke import task, run

@task
def test(cover=False):
    """Run tests (use --cover for coverage tests)"""
    if cover:
        run('py.test --cov-report term-missing --cov=causalinfo tests', pty=True)
    else:
        run('py.test -v', pty=True)

@task
def clean():
    """Clean all build and cruft files"""
    print("Removing python cruft ...")
    run("find . -name '*.pyc' -exec rm -f {} +")
    run("find . -name '*.pyo' -exec rm -f {} +")
    run("find . -name '*~' -exec rm -f {} +")
    run("find . -name '__pycache__' -exec rm -fr {} +")

    print("Removing build ...")
    run("rm -rf build")
    run("rm -rf dist")
    run("rm -rf *.egg-info")

    print("Removing IPython Notebook checkpoints...")
    run("find . -name '__pynb_checkpoints__' -exec rm -fr {} +")

    print("Removing generated html ...")
    run("rm -f README.html")

@task 
def build():
    """Build the distribution"""
    print("Building sdist ...")
    run('python setup.py sdist', hide='out')
    print("Building bdist_wheel ...")
    run('python setup.py bdist_wheel', hide='out')

@task
def publish(release=False):
    """Publish to the cheeseshop."""
    if release:
        run('python setup.py register')
        run('twine upload dist/*.tar.gz')
        run('twine upload dist/*.whl')
    else:
        run('python setup.py -r test register')
        run('twine upload -r test dist/*.tar.gz')
        run('twine upload -r test dist/*.whl')

@task
def readme(browse=True):
    run('rst2html.py README.rst > README.html')
    if browse:
        run('open README.html')
