# -*- coding: utf-8 -*-
from invoke import task, run

@task
def test(cover=False):
    if cover:
        run('py.test --cov-report term-missing --cov=causalinfo tests')
    else:
        run('py.test -v', pty=True)

@task
def clean():
    """Clean all build and cruft files"""
    print("Cleaning python cruft ...")
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

@task 
def build():
    print("Build sdist ...")
    run('python setup.py sdist', hide='out')
    print("Build bdist_wheel ...")
    run('python setup.py bdist_wheel', hide='out')

@task
def publish(release=False):
    """Publish to the cheeseshop."""
    if release:
        run('twine upload dist/*.tar.gz')
        run('twine upload dist/*.whl')
    else:
        run('twine upload -r test dist/*.tar.gz')
        run('twine upload -r test dist/*.whl')

 	# @twine upload dist/*.tar.gz
 	# @twine upload dist/*.whl

@task
def readme(browse=False):
    run('rst2html.py README.rst > README.html')
