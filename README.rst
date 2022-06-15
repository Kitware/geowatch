WATCH - Wide Area Terrestrial Change Hypercube
==============================================

.. The large version wont work because github strips rst image rescaling. 
.. image:: https://ipfs.io/ipfs/QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq
   :height: 100px
   :align: left

|master-pipeline| |master-coverage|


This repository addresses the algorithmic challenges of the 
`IARPA SMART <https://www.iarpa.gov/research-programs/smart>`_ (Space-based
Machine Automated Recognition Technique) program.  The goal of this software is
analyze space-based imagery to perform broad-area search for natural and
anthropogenic events and characterize their extent and progression in time and
space.


The following table provides links to relevant resources for the SMART WATCH project:

+------------------------------------+----------------------------------------------------------------+
| The SMART WATCH Python Module      | https://gitlab.kitware.com/watch/watch/                        |
+------------------------------------+----------------------------------------------------------------+
| The SMART WATCH DVC Repo           | https://gitlab.kitware.com/watch/smart_watch_dvc/              |
+------------------------------------+----------------------------------------------------------------+

For internal collaberators, please refer to the `internal docs <docs/internal_resources.rst>`_ 

To contribute, please read the `contribution instructions <contribution_instructions.rst>`_.

Getting Started
---------------

Install Python
~~~~~~~~~~~~~~

Python 3.8+ is required for watch. Python versions can be managed with either
conda or pyenv. Working with conda is more beginner friendly, but pyenv has
less commercial restrictions, but requires a compiler certain system libraries
(e.g. openssl, sqlite3, readline, ffi, curses, bz2, etc..) to compile Python.
If you are able to compile Python We recommend using pyenv.

To install pyenv, see the `pyenv installation instructions <docs/install_python_pyenv.rst>`_.

To install conda, see the `conda installation instructions <docs/install_python_conda.rst>`_.

NOTE: If using conda, do NOT use ``conda install`` to install Python packages,
we only use conda to install the Python binaries. We exclusively use pip to
manage packages.


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
