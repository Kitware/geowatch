Installing GeoWATCH For Development
===================================


There are two methods for installing GeoWATCH in development mode.
The first is a local install (recommended), and the second is using a docker
image. This document goes over both options.


Option 1: Local Install
-----------------------


Install Python
~~~~~~~~~~~~~~

Python 3.8+ is required for GeoWATCH. Python versions can be managed with either
conda or pyenv. Working with conda is more beginner friendly, but pyenv has
less commercial restrictions, but requires a compiler certain system libraries
(e.g. openssl, sqlite3, readline, ffi, curses, bz2, etc..) to compile Python.
If you are able to compile Python We recommend using pyenv.

To install pyenv, see the `pyenv installation instructions <install_python_pyenv.rst>`_.

To install conda, see the `conda installation instructions <install_python_conda.rst>`_.

NOTE: If using conda, do NOT use ``conda install`` to install Python packages,
we only use conda to install the Python binaries. We exclusively use pip to
manage packages.


Non-Python Requirements
~~~~~~~~~~~~~~~~~~~~~~~

There are several binary libraries that some components of the ``geowatch``
module might assume exist, but don't have Python distributions. These are:

* ffmpeg - for making animated gifs
* tmux - for the tmux queue (to be replaced by slurm)
* jq - for special kwcoco json queries


On Debian-based systems install these via:

.. code:: bash

   sudo apt install ffmpeg tmux jq


Installing
~~~~~~~~~~

Assuming you have cloned this repo, and you are in a Python virtual
environment, cd into the root of the repo and then geowatch can be setup using
the ``run_developer_setup`` script.

.. code:: bash

   cd $HOME/code/watch
   bash ./run_developer_setup.sh

or more explicitly via:

.. code:: bash

   # Update Python build tools
   pip install pip setuptools wheel build -U

   # Install Kitware's gdal wheels
   pip install -r requirements/gdal.txt

   # Install linting tools
   pip install -r requirements/linting.txt

   # Install the main geowatch package with all development extras
   pip install -e .[development,optional,headless]


Submodules
~~~~~~~~~~

All necessary submodules live in `geowatch_tpl` and are statically "vendored"
into the repo, but if you need to develop these submodules, you can use:

.. code:: bash

    git submodule update --init --recursive


See the geowatch_tpl README and docs for more information.


Testing
~~~~~~~

You can test that geowatch is correctly installed by running the run tests script:


.. code:: bash

    ./run_tests.py


Option 2: Docker Image
-----------------------

This repository also includes a ``Dockerfile`` that can be used to
build the GeoWATCH Docker image.  The built Docker image will have the
GeoWATCH Conda environment and GeoWATCH Python module pre-installed.

To build the conda Docker image:

.. code:: bash

   docker build .


To build the pyenv Docker image:

.. code:: bash

    # Requires pulling this file for new docker-buildkit syntax
    docker login
    docker pull docker/dockerfile:1.3.0-labs

    DOCKER_BUILDKIT=1 docker build --progress=plain -t "watch_pyenv310" -f ./dockerfiles/pyenv.Dockerfile .


The usage of conda is no longer directly supported, but still exists for user
convinience. However, pyenv is strongly recommended.
