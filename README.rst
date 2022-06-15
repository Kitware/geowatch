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

For information on testing please see `running and writing watch tests <testing_practices.rst>`_.

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

The current ``watch`` module struture is summarized as follows:


.. code:: bash

    ╙── watch {'.py': 4}
        ├─╼ cli {'.py': 54}
        ├─╼ datacube {'.py': 1}
        │   ├─╼ cloud {'.py': 2}
        │   └─╼ registration {'.py': 6}
        ├─╼ datasets {'.py': 2}
        ├─╼ demo {'.py': 8}
        ├─╼ gis {'.py': 5}
        │   └─╼ sensors {'.py': 2}
        ├─╼ rc {'.gtx': 1, '.json': 3, '.py': 2, '.xml': 1}
        ├─╼ tasks {'.py': 1}
        │   ├─╼ depth {'.json': 1, '.md': 1, '.py': 9}
        │   ├─╼ fusion {'.md': 1, '.py': 15}
        │   │   ├─╼ architectures {'.py': 4}
        │   │   ├─╼ datamodules {'.py': 4, '.pyx': 1}
        │   │   └─╼ methods {'.py': 2}
        │   ├─╼ invariants {'': 1, '.md': 1, '.py': 9}
        │   │   └─╼ data {'.py': 3}
        │   ├─╼ landcover {'.md': 1, '.py': 9}
        │   ├─╼ rutgers_material_change_detection {'.md': 1, '.py': 4}
        │   │   ├─╼ datasets {'.py': 5}
        │   │   ├─╼ models {'.py': 23, '.tmp': 1}
        │   │   └─╼ utils {'.py': 6}
        │   ├─╼ rutgers_material_seg {'.py': 5}
        │   │   ├─╼ datasets {'.py': 13}
        │   │   ├─╼ experiments {'.py': 31}
        │   │   ├─╼ models {'.py': 21}
        │   │   ├─╼ scripts {'.py': 3}
        │   │   └─╼ utils {'.py': 6}
        │   ├─╼ template {'.py': 3}
        │   ├─╼ tracking {'.py': 7}
        │   └─╼ uky_temporal_prediction {'': 1, '.md': 1, '.py': 7, '.yml': 1}
        │       ├─╼ models {'.py': 4}
        │       └─╼ spacenet {'.py': 2}
        │           └─╼ data {'.py': 2}
        │               └─╼ splits_unmasked {'.py': 2}
        └─╼ utils {'.py': 32}
            └─╼ lightning_ext {'.py': 5}
                └─╼ callbacks {'.py': 7, '.txt': 1}




Important WATCH Scripts
~~~~~~~~~~~~~~~~~~~~~~~

The SMART WATCH module comes with a command line interface (CLI). This can be invoked
via ``python -m watch --help`` (note: if the module has been pip installed
``python -m watch`` can be replaced with ``smartwatch`` for primary CLI commands).

In these examples we use the ``smartwatch`` invocation to be concise, but you
can simply replace them with ``python -m smartwatch`` if your shell does not
support the entrypoint.


The following is a list of the primary CLI commands:

* ``smartwatch find_dvc --help`` - Helper to return the path the the WATCH DVC Repo (if it is a known location)

* ``smartwatch watch_coco_stats --help`` - Print statistics about a kwcoco file with a focus on sensor / channel frequency and region information.

* ``smartwatch coco_intensity_histograms --help`` - Show per-band / per-sensor histograms of pixel intensities. This is useful for acessing the harmonization between sensors. 

* ``smartwatch coco_visualize_videos --help`` - Visualize a video sequence with and without annotations. This can also create an animation of arbitrary feature channels.

* ``smartwatch coco_align_geotiffs --help`` - Crop a set of unstructured kwcoco file (that registers a set of geotiffs) into a TA-2 ready kwcoco file containing cropped video sequences corresponding to each region in a specified set of regions files.

* ``smartwatch project_annotations --help`` - Project annotations from raw site/region models onto the pixel space of a kwcoco file. This also propogates these annotations in time as needed.

* ``smartwatch kwcoco_to_geojson --help`` - Transform "saliency" or "class" heatmaps into tracked geojson site models, and optionally score these with IARPA metrics.


Using ``--help`` shows the top level modal CLI:


.. code:: 

        usage: smartwatch [-h] [--version] {command}
                          ...

        The SMART WATCH CLI

        positional arguments:
            coco_add_watch_fields (add_fields)
                                Updates image transforms in a kwcoco json file to align all videos to a
            coco_align_geotiffs (align)
                                Create a dataset of aligned temporal sequences around objects of interest
            coco_extract_geo_bounds
                                Extract bounds of geojson tiffs (in a kwcoco file) into a regions file
            geotiffs_to_kwcoco  Create a kwcoco manifest of a set of on-disk geotiffs
            hello_world         opaque sub command
            watch_coco_stats (stats)
                                Print watch-relevant information about a kwcoco dataset
            merge_region_models
                                Combine the specific features from multiple region files into a single one.
            project_annotations (project)
                                Projects annotations from geospace onto a kwcoco dataset and optionally
            coco_show_auxiliary
                                Visualize kwcoco auxiliary channels to spot-inspect if they are aligned
            coco_visualize_videos (visualize)
                                Visualizes annotations on kwcoco video frames on each band
            coco_intensity_histograms (intensity_histograms)
                                Updates image transforms in a kwcoco json file to align all videos to a
            find_dvc            Command line helper to find the path to the watch DVC repo
            kwcoco_to_geojson   opaque sub command
            run_metrics_framework
                                opaque sub command
            torch_model_stats (model_info)
                                Print stats about a torch model.

        optional arguments:
          -h, --help            show this help message and exit
          --version             show version number and exit (default: False)
   


Other important commands that are not exposed via the main CLI are:

* ``python -m watch.tasks.fusion.fit --help`` - Train a TA2 fusion model.
  
* ``python -m watch.tasks.fusion.predict --help`` - Predict using a pretrained TA2 fusion model on a target dataset.

* ``python -m watch.tasks.fusion.evaluate --help`` - Measure pixel-level quality metrics between a prediction and truth kwcoco file.


Note to developers: if an important script exists and is not listed here,
please submit an MR.

New Python command line scripts can be added under the ``watch/cli`` directory.
New tools can be registered with the ``watch-cli`` tool in the
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
