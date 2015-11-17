# -*- coding: utf-8 -*-
from invoke import task, run
import os
import sys

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

@task
def notebook():
    from IPython.terminal.ipapp import launch_new_instance
    from socket import gethostname
    import warnings

    print('Installing in develop mode')
    run('python setup.py develop', hide='out')

    print('Changing to notebooks folder')
    here = os.path.dirname(__file__)
    os.chdir(os.path.join(here, 'notebooks'))
    old_argv = sys.argv[:]

    # Taken from here:
    # http://stackoverflow.com/questions/
    # 26338688/start-ipython-notebook-with-python-file
    try:
        warnings.filterwarnings("ignore", module = "zmq.*")
        sys.argv = ['ipython', 'notebook']
        sys.argv.append("--IPKernelApp.pylab='inline'")
        sys.argv.append("--NotebookApp.ip=" + gethostname())
        sys.argv.append("--NotebookApp.open_browser=True")
        print('Invoking "' + ' '.join(sys.argv) + '"')
        launch_new_instance()
    finally:
        # Not sure this is strictly necessary...
        sys.argv = old_argv
        os.chdir(here)
        print('Removing development package...')
        run('python setup.py develop -u', hide='out')
