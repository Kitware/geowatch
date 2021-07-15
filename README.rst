WATCH
=====

|master-pipeline| |master-coverage|

This repository addresses the algorithmic challenges of the IARPA SMART
program. The goal of this software is analyze space-based imagery to
perform broad-area search for natural and anthropogenic events and
characterize their extent and progression in time and space.


The following table provides links to relevant resources for the SMART WATCH project:

+-------------------------------+----------------------------------------------------------------+
| The SMART WATCH Python Module | https://gitlab.kitware.com/smart/watch/                        |
+-------------------------------+----------------------------------------------------------------+
| The SMART WATCH DVC Repo      | https://gitlab.kitware.com/smart/smart_watch_dvc/              |
+-------------------------------+----------------------------------------------------------------+
| The SMART WATCH Collection    | https://data.kitware.com/#collection/602457272fa25629b95d1718  |
+-------------------------------+----------------------------------------------------------------+

Getting Started
---------------

Install Conda
~~~~~~~~~~~~~

Conda3 is required to set up the environment using Python 3. Follow the
instructions below to install miniconda3 for Linux. For Windows 10
users, the Windows Subsystem for Linux (WSL) allows you to run Linux
within Windows.

.. code:: bash

    # Download the conda install script into a temporary directory
    mkdir -p ~/tmp
    cd ~/tmp

    # To update to a newer version see:
    # https://docs.conda.io/en/latest/miniconda_hashes.html for updating
    CONDA_INSTALL_SCRIPT=Miniconda3-py38_4.9.2-Linux-x86_64.sh
    CONDA_EXPECTED_SHA256=1314b90489f154602fd794accfc90446111514a5a72fe1f71ab83e07de9504a7
    curl https://repo.anaconda.com/miniconda/$CONDA_INSTALL_SCRIPT > $CONDA_INSTALL_SCRIPT
    CONDA_GOT_SHA256=$(sha256sum $CONDA_INSTALL_SCRIPT | cut -d' ' -f1)
    # For security, it is important to verify the hash
    if [[ "$CONDA_GOT_SHA256" != "$CONDA_EXPECTED_SHA256_HASH" ]]; then
        echo "Downloaded file does not match hash! DO NOT CONTINUE!"
        exit 1;
    fi
    chmod +x $CONDA_INSTALL_SCRIPT 

    # Install miniconda to user local directory
    _CONDA_ROOT=$HOME/.local/conda
    sh $CONDA_INSTALL_SCRIPT -b -p $_CONDA_ROOT
    # Activate the basic conda environment
    source $_CONDA_ROOT/etc/profile.d/conda.sh
    # Update the base and create a virtual environment named py38
    conda update --name base conda --yes 


Create WATCH Conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instructions below can be used to create the WATCH Conda
environment.

.. code:: bash

   mkdir WATCH_DIR #(pick a name of your choice)
   cd WATCH_DIR
   git clone https://gitlab.kitware.com/smart/watch.git
   cd watch
   conda env create -f conda_env.yml
   conda activate watch

To deactivate the watch environment, run:

.. code:: bash

   conda deactivate

To remove the watch environment, run:

.. code:: bash

   conda deactivate #(if watch is activated before)
   conda remove --name watch --all

To update the watch environment when new packages have been added, run:

.. code:: bash

   conda activate watch
   conda env update -f deployment/conda/conda_env.yml

Installation
~~~~~~~~~~~~

The WATCH Python module can then be installed with ``pip`` via the following
command, where ``/path/to/watch-repo`` is the absolute path to the directory
containing this README.md file.

NOTE: It is important you install the module with the editable (``-e``) flag,
otherwise changes you make to the module, will not be reflected when you run
your scripts.

.. code:: bash

   pip install -e /path/to/watch-repo


This is more commonly done as

.. code:: bash

   cd /path/to/watch-repo
   pip install -e .


After the ``watch`` module has been installed to your python environment, it
can be imported from anywhere regardless of the current working directory.


NOTE: The ``conda_env.yml`` was written such that all dependencies are
installed via pip. This allows for alternatives to conda such as 
`pyenv <docs/pyenv_alternative.rst>`_ to be used.

Docker Image
~~~~~~~~~~~~

This repository also includes a ``Dockerfile`` that can be used to
build the WATCH Docker image.  The built Docker image will have the
WATCH Conda environment and WATCH Python module pre-installed.

To build the Docker image:

.. code:: bash

   docker build .


Running the Algorithm Toolkit (ATK) example project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure that you have already setup the WATCH Conda enviornment.

Then enter the following commands in your terminal to run the ATK
example project:

.. code:: bash

   cd atk/example
   alg run

Point your browser to http://localhost:5000/. You should see the
development environment welcome page.

Refer to the `development environment`_ portion of the `atk docs`_ for a
crash course on how to use the web-based development environment.

Running tests
-------------

Watch uses the ``pytest`` module for running unit tests. Unit tests
should be added into the ``tests`` directory and files should be
prefixed with ``test_``.

Additionally, code blocks in function docstrings will be interpreted as tests using 
`xdoctest <https://xdoctest.readthedocs.io/en/latest/autoapi/xdoctest/index.html>`_ 
as part of the `Google docstring convention <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

For example here are what doctests look like for a class and for a function:

.. code:: python

    class GdalOpen:
        """
        A simple context manager for friendlier gdal use.

        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> from watch.utils.util_raster import *
            >>> from watch.demo.landsat_demodata import grab_landsat_product
            >>> path = grab_landsat_product()['bands'][0]
            >>> 
            >>> # standard use:
            >>> dataset = gdal.Open(path)
            >>> print(dataset.GetDescription())  # do stuff
            >>> del dataset  # or 'dataset = None'
            >>> 
            >>> # equivalent:
            >>> with GdalOpen(path) as dataset:
            >>>     print(dataset.GetDescription())  # do stuff
        """
        # code goes here

    def my_cool_function(inputs):
        """
        The purpose of this function is to demonstrate how to write a doctest. 

        Example:
            >>> # An example of how to use my cool function
            >>> # The xdoctest module will run this as a test
            >>> inputs = 'construct-demo-data'
            >>> my_cool_function(inputs)
        """
        import this
        print('You input: {}'.format(inputs))


The ``run_tests.py`` script provided here will run all tests in the ``tests``
directory and in docstrings and report coverage. This script is simply a
wrapper around the ``pytest`` command.

Alternatively doctests can be invoked specivially via ``xdoctest -m watch`` to
run all doctests, or ``xdoctest -m <path-to-file>`` to run all doctests in a
file.

How to contribute
-----------------

We follow a `merge requests <https://docs.gitlab.com/ee/user/project/merge_requests/>`_ workflow.

Here is a complete, minimal example of how to add code to this repository, assuming you have followed the instructions above. You should be inside this repo's directory tree on your local machine and have the WATCH Conda environment active.

.. code:: bash

   git checkout -b my_new_branch

   # example commit: change some files
   git commit -am "changed some files"

   # example commit: add a file
   echo "some work" > new_file.py
   git add new_file.py
   git commit -am "added a file"

   # now, integrate other changes that have occurred in this time
   git merge origin/master

   # make sure all tests pass (including ones you wrote!)
   python run_tests.py

   # and add your branch to gitlab.kitware.com
   git push --set-upstream origin my_new_branch


To get your code merged, create an MR from your branch `here <https://gitlab.kitware.com/smart/watch/-/merge_requests>`_ and @ someone from Kitware to take a look at it. It is a good idea to create a `draft MR <https://docs.gitlab.com/ee/user/project/merge_requests/drafts.html>`_ a bit before you are finished, in order to ask and answer questions about your new feature and make sure it is properly tested.

You can use `markdown <https://docs.gitlab.com/ee/user/markdown.html>`_ to write an informative merge message.

Adding submodules
-----------------

Library code can be added to the relevant subdirectory under the
``watch`` directory. The current submodules are as follows:



.. code:: bash


    watch
    ├── cli
    ├── demo
    ├── datacube
    │   ├── atmosphere
    │   ├── cloud
    │   ├── reflectance
    │   └── registration
    ├── sequencing
    ├── validation
    ├── datasets
    ├── tasks
    │   ├── fusion
    │   ├── invariants
    │   ├── landcover
    │   ├── materials
    │   ├── reflectance
    │   ├── semantics
    │   ├── template
    │   └── uky_temporal_prediction
    ├── utils
    ├── gis
    └── validation


Adding command line tools
-------------------------

New Python command line scripts can be added under the ``watch/cli``
directory. New tools can be registered with the ``watch-cli`` tool in the
``watch/cli/__main__.py`` file, or invoked explicitly via ``python -m
watch.cli.<script-name>``.

Scripts that don’t quite belong in the WATCH Python module itself
(e.g. due to a lack of general purpose use, or lack of polish) can be
added to the ``scripts`` or ``dev`` directory. Generally, the ``scripts``
directory is for data processing and ``dev`` is for scripts related to
repository maintenence. 


.. _development environment: https://algorithm-toolkit.readthedocs.io/en/latest/dev-environment.html#
.. _atk docs: https://algorithm-toolkit.readthedocs.io/en/latest/index.html

.. |master-pipeline| image:: https://gitlab.kitware.com/smart/watch/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/smart/watch/-/pipelines/master/latest
.. |master-coverage| image:: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
   :target: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
