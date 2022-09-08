****************
Watch Onboarding
****************

The purpose of this document is to guide a new user in the installation and
usage of the WATCH system.

This document assumes you are proficient with Python and have an understanding
of virtual environments.


For details on installing the watch system in development mode see 
`installing watch for development guide <../docs/installing_watch.rst.rst>`_.


Tutorials
---------

The following tutorials detail how to train simple fusion models


* Tutorial 1: `Toy RGB Experiment <../watch/tasks/fusion/experiments/crall/toy_experiments_rgb.sh>`_ 

* Tutorial 2: `Toy MSI Experiment <../watch/tasks/fusion/experiments/crall/toy_experiments_msi.sh>`_ 

* Tutorial 3: TODO: tutorial about kwcoco (See docs for `kwcoco <https://gitlab.kitware.com/computer-vision/kwcoco>`_)

* Tutorial 4: TODO: tutorial about watch.mlops


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



Important WATCH Scripts
-----------------------

The SMART WATCH module comes with a command line interface (CLI). This can be invoked
via ``python -m watch --help`` (note: if the module has been pip installed
``python -m watch`` can be replaced with ``smartwatch`` for primary CLI commands).

More information can be found in the `watch cli docs <docs/watch_cli.rst>`_.

In these examples we use the ``smartwatch`` invocation to be concise, but you
can simply replace them with ``python -m smartwatch`` if your shell does not
support the entrypoint.


The following is a list of the primary CLI commands:

* ``smartwatch find_dvc --help`` - Helper to return the path the the WATCH DVC Repo (if it is a known location)

* ``smartwatch stats --help`` - Print statistics about a kwcoco file with a focus on sensor / channel frequency and region information.

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
  


The "smartwatch_dvc" command
----------------------------

We provide a utility to help manage data paths called "smartwatch_dvc".  It
comes preconfigured with common paths for core-developer machines You can see
what paths are available by using the "list" command

.. code:: bash

    smartwatch_dvc list

which outputs something like this:


.. code::

                   name hardware         tags                                                               path  exists
    0    drop4_expt_ssd      ssd  phase2_expt                            /root/data/dvc-repos/smart_expt_dvc-ssd   False
    1    drop4_data_ssd      ssd  phase2_data                            /root/data/dvc-repos/smart_data_dvc-ssd   False
    2    drop4_expt_hdd      hdd  phase2_expt                                /root/data/dvc-repos/smart_expt_dvc   False
    3    drop4_data_hdd      hdd  phase2_data                                /root/data/dvc-repos/smart_data_dvc   False


To see full help use `smartwatch_dvc --help`

.. code:: bash

    usage: FindDVCConfig 

    Command line helper to find the path to the watch DVC repo

    positional arguments:
      command               can be find, set, add, list, or remove
      name                  specify a name to query or store or remove

    options:
      -h, --help            show this help message and exit
      --command COMMAND     can be find, set, add, list, or remove (default: find)
      --name NAME           specify a name to query or store or remove (default: None)
      --hardware HARDWARE   Specify hdd, ssd, etc..., Setable and getable property (default: None)
      --priority PRIORITY   Higher is more likely. Setable and getable property (default: None)
      --tags TAGS           User note. Setable and queryable property (default: None)
      --path PATH           The path to the dvc repo. Setable and queryable property (default: None)
      --verbose VERBOSE     verbosity mode (default: 1)
      --must_exist MUST_EXIST
                            if True, filter to only directories that exist. Defaults to false except on "find", which is True. (default: auto)
      --config CONFIG       special scriptconfig option that accepts the path to a on-disk configuration file, and loads that into this 'FindDVCConfig' object. (default: None)
      --dump DUMP           If specified, dump this config to disk. (default: None)
      --dumps               If specified, dump this config stdout (default: False)
