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

.. .. Under construction
.. .. | The SMART WATCH DVC Repo           | https://gitlab.kitware.com/watch/smart_watch_dvc/              |
.. .. +------------------------------------+----------------------------------------------------------------+


Getting Started
---------------

To quickly get started locally, clone the watch repo and run:

.. code:: bash

   ./run_developer_setup.py
 

Test everything is working by running

.. code:: bash

   ./run_tests.py

For more details see the `installing watch for development guide <docs/installing_watch.rst.rst>`_.


Development
-----------

For new collaberators, please refer to the `onboarding docs <docs/onboarding.rst>`_ 

For internal collaberators, please refer to the `internal docs <docs/internal_resources.rst>`_ 


The Watch CLI
-------------

The watch module comes with a set of command line tools. 
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


For more details about the WATCH CLI and other CLI tools included in this package see:
`the WATCH CLI docs <docs/watch_cli.rst>`_ 


Documentation
-------------

For quick reference, a list of current documentation files is:

* `Onboarding Docs <docs/onboarding.rst>`_

* `Internal Resources <docs/internal_resources.rst>`_

* `The WATCH CLI <docs/watch_cli.rst>`_

* Contribution:

  + `Contribution Instructions <docs/contribution_instructions.rst>`_

  + `Rebasing Procedure <docs/rebasing_procedure.md>`_

  + `Testing Practices <docs/testing_practices.md>`_

* Installing: 

  + `Installing WATCH <docs/installing_watch.rst>`_

  + `Installing Python via Conda <docs/install_python_conda.rst>`_

  + `Installing Python via PyEnv <docs/install_python_pyenv.rst>`_

* Fusion Related Docs:

  + `TA2 Fusion Overview <docs/fusion_overview.rst>`_

  + `TA2 Deep Dive Info <docs/ta2_deep_dive_info.md>`_

  + `TA2 Feature Integration <docs/ta2_feature_integration.md>`_

* Older Design Docs:

  + `Structure Proposal <docs/structure_proposal.md>`_

* Tutorials:

  + Tutorial 1: `Toy RGB Fusion Model Example <../watch/tasks/fusion/experiments/crall/toy_experiments_rgb.sh>`_ 

  + Tutorial 2: `Toy MSI Fusion Model Example <../watch/tasks/fusion/experiments/crall/toy_experiments_msi.sh>`_ 


.. |master-pipeline| image:: https://gitlab.kitware.com/smart/watch/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/smart/watch/-/pipelines/master/latest
.. |master-coverage| image:: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
   :target: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
