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

Install Python
~~~~~~~~~~~~~~

Python 3.8+ is required for watch. Python versions can be managed with either
conda or pyenv. Working with conda is more beginner friendly, but pyenv has
less commercial restrictions.

To install pyenv, see the `pyenv installation instructions <docs/pyenv_alternative.rst>`_.

To install Miniconda3, follow the instructions below for Linux. For Windows 10
users, the Windows Subsystem for Linux (WSL) allows you to run Linux within
Windows.

.. code:: bash

    # Download the conda install script into a temporary directory
    mkdir -p ~/tmp
    cd ~/tmp

    # To update to a newer version see:
    # https://docs.conda.io/en/latest/miniconda_hashes.html for updating
    CONDA_INSTALL_SCRIPT=Miniconda3-py38_4.9.2-Linux-x86_64.sh
    curl https://repo.anaconda.com/miniconda/$CONDA_INSTALL_SCRIPT > $CONDA_INSTALL_SCRIPT

    # For security, it is important to verify the hash
    CONDA_EXPECTED_SHA256=1314b90489f154602fd794accfc90446111514a5a72fe1f71ab83e07de9504a7
    echo "${CONDA_EXPECTED_SHA256}  ${CONDA_INSTALL_SCRIPT}" > conda_expected_hash.sha256 
    if ! sha256sum --status -c conda_expected_hash.sha256; then
        echo "Downloaded file does not match hash! DO NOT CONTINUE!"
    else
        echo "Hash verified, continue with install"
        chmod +x $CONDA_INSTALL_SCRIPT 
        # Install miniconda to user local directory
        _CONDA_ROOT=$HOME/.local/conda
        sh $CONDA_INSTALL_SCRIPT -b -p $_CONDA_ROOT
        # Activate the basic conda environment
        source $_CONDA_ROOT/etc/profile.d/conda.sh
        # Update the base 
        conda update --name base conda --yes 
    fi

NOTE: If using conda, do NOT use ``conda install`` to install Python packages. 


Create WATCH environment with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If using conda, the instructions below can be used to create the WATCH Conda
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


Create WATCH environment with Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First create and activate a new virtual environment (note this could be done
with conda as well).

If using `pyenv installation instructions <docs/pyenv_alternative.rst>`_, then
a virtual environment can be created with the standard ``venv`` module.
Assuming you have installed Python 3.8.5 with pyenv the following will create a
virtual environment.

.. code:: bash

    CHOSEN_PYTHON_VERSION=3.8.5
    # Set your shell to use this pyenv shim
    pyenv shell $CHOSEN_PYTHON_VERSION

    # Create the virtual environment
    python -m venv $(pyenv prefix)/envs/pyenv-watch

    # Activate the virtual environment
    source $(pyenv prefix)/envs/pyenv-watch/bin/activate


Once you are in a virtual environment (managed by either conda or pyenv), the
WATCH Python module can then be installed with ``pip`` via the following
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

This installation process is also scripted in the top-level
``run_developer_setup.sh`` script and takes care of issues that can arise with
opencv-python.

After the ``watch`` module has been installed to your python environment, it
can be imported from anywhere regardless of the current working directory as
long as the virtual environment was installed in is active.


Docker Image
~~~~~~~~~~~~

This repository also includes a ``Dockerfile`` that can be used to
build the WATCH Docker image.  The built Docker image will have the
WATCH Conda environment and WATCH Python module pre-installed.

To build the Docker image:

.. code:: bash

   docker build .


Module Structure
-----------------

The current ``watch`` module struture is as follows:


.. code:: bash

    ├── watch
    │   ├── cli
    │   ├── datacube
    │   │   ├── cloud
    │   │   └── registration
    │   ├── datasets
    │   ├── demo
    │   ├── gis
    │   ├── rc
    │   ├── tasks
    │   │   ├── depth
    │   │   ├── fusion
    │   │   ├── invariants
    │   │   ├── landcover
    │   │   ├── rutgers_material_change_detection
    │   │   ├── rutgers_material_seg
    │   │   ├── template
    │   │   ├── tracking
    │   │   └── uky_temporal_prediction
    │   └── utils


Important WATCH Scripts
~~~~~~~~~~~~~~~~~~~~~~~

The SMART WATCH module comes with a command line interface (CLI). This can be invoked
via ``python -m watch --help`` (note: if the module has been pip installed
``python -m watch`` can be replaced with ``smartwatch`` for primary CLI commands).

The following is a list of the primary CLI commands:

* ``python -m watch find_dvc --help`` - Helper to return the path the the WATCH DVC Repo (if it is a known location)

* ``python -m watch watch_coco_stats --help`` - Print statistics about a kwcoco file with a focus on sensor / channel frequency and region information.

* ``python -m watch coco_intensity_histograms --help`` - Show per-band / per-sensor histograms of pixel intensities. This is useful for acessing the harmonization between sensors. 

* ``python -m watch coco_visualize_videos --help`` - Visualize a video sequence with and without annotations. This can also create an animation of arbitrary feature channels.

* ``python -m watch coco_align_geotiffs --help`` - Crop a set of unstructured kwcoco file (that registers a set of geotiffs) into a TA-2 ready kwcoco file containing cropped video sequences corresponding to each region in a specified set of regions files.

* ``python -m watch project_annotations --help`` - Project annotations from raw site/region models onto the pixel space of a kwcoco file. This also propogates these annotations in time as needed.

* ``python -m watch kwcoco_to_geojson --help`` - Transform "saliency" or "class" heatmaps into tracked geojson site models, and optionally score these with IARPA metrics.


Other important commands that are not exposed via the main CLI are:

* ``python -m watch.tasks.fusion.fit --help`` - Train a TA2 fusion model.
  
* ``python -m watch.tasks.fusion.predict --help`` - Predict using a pretrained TA2 fusion model on a target dataset.

* ``python -m watch.tasks.fusion.evaluate --help`` - Measure pixel-level quality metrics between a prediction and truth kwcoco file.


Note to developers: if an important script exists and is not listed here,
please submit an MR.

New Python command line scripts can be added under the ``watch/cli``
directory. New tools can be registered with the ``watch-cli`` tool in the
``watch/cli/__main__.py`` file, or invoked explicitly via ``python -m
watch.cli.<script-name>``.

Scripts that don’t quite belong in the WATCH Python module itself
(e.g. due to a lack of general purpose use, or lack of polish) can be
added to the ``scripts`` or ``dev`` directory. Generally, the ``scripts``
directory is for data processing and ``dev`` is for scripts related to
repository maintenence. 
  

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

Here is a complete, minimal example of how to add code to this repository, assuming you have followed the instructions above. You should be inside this repo's directory tree on your local machine and have the WATCH environment active.

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

   # If you are brave, use `git rebase -i origin/master` instead. It produces a
   # nicer git history, but can be more difficult for people unfamiliar with git.

   # make sure you lint your code!
   python dev/lint.py watch

   # make sure all tests pass (including ones you wrote!)
   python run_tests.py

   # and add your branch to gitlab.kitware.com
   git push --set-upstream origin my_new_branch

   # This will print a URL to make a MR (merge request)
   # Follow the steps on gitlab to submit this. Then it will be reviewed.
   # Tests and the linter will run on the CI, so make sure they work
   # on your local machine to avoid surprise failures.


To get your code merged, create an MR from your branch `here <https://gitlab.kitware.com/smart/watch/-/merge_requests>`_ and @ someone from Kitware to take a look at it. It is a good idea to create a `draft MR <https://docs.gitlab.com/ee/user/project/merge_requests/drafts.html>`_ a bit before you are finished, in order to ask and answer questions about your new feature and make sure it is properly tested.

You can use `markdown <https://docs.gitlab.com/ee/user/markdown.html>`_ to write an informative merge message.


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



.. _development environment: https://algorithm-toolkit.readthedocs.io/en/latest/dev-environment.html#
.. _atk docs: https://algorithm-toolkit.readthedocs.io/en/latest/index.html

.. |master-pipeline| image:: https://gitlab.kitware.com/smart/watch/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/smart/watch/-/pipelines/master/latest
.. |master-coverage| image:: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
   :target: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
