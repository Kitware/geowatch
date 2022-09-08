Installing Watch For Development
================================


There are two methods for installing watch in development mode. 
The first is a local install (recommended), and the second is using a docker
image. This document goes over both options.


Option 1: Local Install
-----------------------


Install Python
~~~~~~~~~~~~~~

Python 3.8+ is required for watch. Python versions can be managed with either
conda or pyenv. Working with conda is more beginner friendly, but pyenv has
less commercial restrictions, but requires a compiler certain system libraries
(e.g. openssl, sqlite3, readline, ffi, curses, bz2, etc..) to compile Python.
If you are able to compile Python We recommend using pyenv.

To install pyenv, see the `pyenv installation instructions <../docs/install_python_pyenv.rst>`_.

To install conda, see the `conda installation instructions <../docs/install_python_conda.rst>`_.

NOTE: If using conda, do NOT use ``conda install`` to install Python packages,
we only use conda to install the Python binaries. We exclusively use pip to
manage packages.


Non-Python Requirements
~~~~~~~~~~~~~~~~~~~~~~~

There are several binary libraries that some components of the watch module
might assume exist, but don't have Python distributions. These are:

* ffmpeg - for making animated gifs
* tmux - for the tmux queue (to be replaced by slurm)
* jq - for special kwcoco json queries


On Debian-based systems install these via:

.. code:: bash

   sudo apt install ffmpeg tmux jq


Installing
~~~~~~~~~~

Assuming you have cloned this repo, and you are in a Python virtual
environment, the watch repo can be setup as:

.. code:: bash

   # Update Python build tools
   pip install pip setuptools wheel build -U

   # Install Kitware's gdal wheels
   pip install -r requirements/gdal.txt

   # Install linting tools
   pip install -r requirements/linting.txt

   # Install the main watch package with all development extras
   pip install -e .[development,optional,headless]



Testing
~~~~~~~

You can test that watch is correctly installed by running the run tests script:


.. code:: bash

    ./run_tests.py


Option 2: Docker Image
-----------------------

This repository also includes a ``Dockerfile`` that can be used to
build the WATCH Docker image.  The built Docker image will have the
WATCH Conda environment and WATCH Python module pre-installed.

To build the conda Docker image:

.. code:: bash

   docker build .


To build the pyenv Docker image:

.. code:: bash

    # Requires pulling this file for new docker-buildkit syntax
    docker login
    docker pull docker/dockerfile:1.3.0-labs

    DOCKER_BUILDKIT=1 docker build --progress=plain -t "watch_pyenv310" -f ./dockerfiles/pyenv.Dockerfile .


We will eventually deprecate the usage of conda. Using pyenv is recommended.
