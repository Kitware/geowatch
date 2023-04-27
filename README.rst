GEOWATCH - Geographic Wide Area Terrestrial Change Hypercube
============================================================

.. The large version wont work because github strips rst image rescaling.
.. image:: https://ipfs.io/ipfs/QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq
   :height: 100px
   :align: left

|main-pipeline| |main-coverage|


This repository addresses the algorithmic challenges of the
`IARPA SMART <https://www.iarpa.gov/research-programs/smart>`_ (Space-based
Machine Automated Recognition Technique) program.  The goal of this software is
analyze space-based imagery to perform broad-area search for natural and
anthropogenic events and characterize their extent and progression in time and
space.


The following table provides links to relevant resources for the SMART WATCH project:

+----------------------------------------------------------+----------------------------------------------------------------+
| The Public GEOWATCH Python Module                        | https://gitlab.kitware.com/computer-vision/geowatch/           |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Internal SMART GEOWATCH Python Module                | https://gitlab.kitware.com/smart/watch/                        |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Phase 2 Internal SMART GEOWATCH DVC Data Repo        | https://gitlab.kitware.com/smart/smart_data_dvc/               |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Phase 2 Internal SMART GEOWATCH DVC Experiment Repo  | https://gitlab.kitware.com/smart/smart_expt_dvc/               |
+----------------------------------------------------------+----------------------------------------------------------------+


Getting Started
---------------

To quickly get started locally, clone the geowatch repo:


.. code:: bash

   # Create a directory for CODE
   mkdir -p $HOME/code
   # Internal
   # git clone https://gitlab.kitware.com/smart/watch/  $HOME/code/watch

   # Public
   git clone https://gitlab.kitware.com/computer-vision/geowatch/  $HOME/code/watch


Then run:

.. code:: bash

   cd $HOME/code/watch
   bash ./run_developer_setup.sh


Test everything is working by running

.. code:: bash

   ./run_tests.py

For more details see the `installing GEOWATCH for development guide <docs/environment/installing_watch.rst>`_.


Development
-----------

For new collaberators, please refer to the `onboarding docs <docs/onboarding.rst>`_

For internal collaberators, please refer to the `internal docs <docs/data/internal_resources.rst>`_


The GEOWATCH CLI
----------------

The ``geowatch`` module comes with a set of command line tools.
Using ``python -m geowatch --help`` or ``geowatch --help`` shows the top level modal CLI:

.. code::

    usage: geowatch [-h] [--version]
                    {add_fields,align,stats,reproject,visualize,spectra,dvcdir,run_tracker,iarpa_eval,model_stats,clean_geotiffs,animate,average_features,time_combine,crop_sitemodels,remove_bad_images,schedule,manager,aggregate}
                    ...

    🌐🌐🌐 The GEO-WATCH CLI 🌐🌐🌐

    An open source research and production environment for image and video
    segmentation and detection with geospatial awareness.

    Developed by Kitware. Funded by the IARPA SMART challenge.

    options:
      -h, --help            show this help message and exit
      --version             show version number and exit (default: False)

    commands:
      {add_fields,align,stats,reproject,visualize,spectra,dvcdir,run_tracker,iarpa_eval,model_stats,clean_geotiffs,animate,average_features,time_combine,crop_sitemodels,remove_bad_images,schedule,manager,aggregate}
                            specify a command to run
        add_fields (coco_add_watch_fields)
                            Updates image transforms in a kwcoco json file to align all videos to a
        align (coco_align, coco_align_geotiff, coco_align_geotiffs)
                            Create a dataset of aligned temporal sequences around objects of interest
        stats (watch_coco_stats)
                            Print watch-relevant information about a kwcoco dataset.
        reproject (project, reproject_annotations)
                            Projects annotations from geospace onto a kwcoco dataset and optionally
        visualize (coco_visualize_videos)
                            Visualizes annotations on kwcoco video frames on each band
        spectra (intensity_histograms, coco_spectra)
                            Updates image transforms in a kwcoco json file to align all videos to a
        dvcdir (find_dvc)   Command line helper to find the path to the watch DVC repo
        run_tracker (kwcoco_to_geojson)
                            Convert KWCOCO to IARPA GeoJSON
        iarpa_eval (run_metrics_framework)
                            Score IARPA site model GeoJSON files using IARPA's metrics-and-test-framework
        model_stats (model_info, torch_model_stats)
                            Print stats about a torch model.
        clean_geotiffs (coco_clean_geotiffs)
                            A preprocessing step for geotiff datasets.
        animate (gifify)    Convert a sequence of images into a video or gif.
        average_features (ensemble, coco_average_features)
                            Create a new kwcoco file with averaged features from multiple kwcoco files.
        time_combine (coco_time_combine)
                            Averages kwcoco images over a sliding temporal window in a video.
        crop_sitemodels (crop_sites_to_regions)
                            Crops site models to the bounds of a region model.
        remove_bad_images (coco_remove_bad_images)
                            Remove image frames that have little or nothing useful in them from a
        schedule (mlops_schedule, schedule_evaluation)
                            Driver for GEOWATCH mlops evaluation scheduling
        manager (mlops_manager)
                            Certain parts of these names have special nomenclature to make them easier
        aggregate (mlops_aggregate)
                            Aggregates results from multiple DAG evaluations.



As a researcher / developer / user the most important commands for you to know are:

* ``geowatch stats <kwcoco_file>`` - Get geowatch-relevant statistics about data in a kwcoco file

* ``geowatch visualize <kwcoco_file>`` - Visualize the image / videos / annotations in a kwcoco file.

* ``geowatch spectra <kwcoco_file>`` - Look at the distribution of intensity values per band / per sensor in a kwcoco file.

* ``geowatch model_stats <fusion_model_file>`` - Get stats / info about a trained fusion model.

* ``geowatch reproject`` - Reproject CRS84 (geojson) annoations to image space and write to a kwcoco file.

* ``geowatch align`` - Crop a kwcoco dataset based on CRS84 (geojson) regions.

* ``geowatch clean_geotiff`` - Heuristic to detect large regions of black pixels and edit them to NODATA in the geotiff.

* ``geowatch geotiffs_to_kwcoco`` - Create a kwcoco file from a set of on-disk geotiffs.

* ``smartwatch_dvc`` - Helper to register / retreive your DVC paths so scripts can be written agnostic to filesystem layouts. See `docs <docs/data/using_smartwatch_dvc.rst>`_ for more details.


For more details about the GEOWATCH CLI and other CLI tools included in this package see:
`the GEOWATCH CLI docs <docs/watch_cli.rst>`_


Documentation
-------------

For quick reference, a list of current documentation files is:

* `Onboarding Docs <docs/onboarding.rst>`_

* `Internal Resources <docs/data/internal_resources.rst>`_

* `The GEOWATCH CLI <docs/watch_cli.rst>`_

* Contribution:

  + `Contribution Instructions <docs/development/contribution_instructions.rst>`_

  + `Rebasing Procedure <docs/development/rebasing_procedure.rst>`_

  + `Testing Practices <docs/testing/testing_practices.rst>`_

  + `Supporting Projects <docs/misc/supporting_projects.rst>`_

  + `Coding Conventions <docs/development/coding_conventions.rst>`_

* Installing:

  + `Installing GEOWATCH <docs/environment/installing_watch.rst>`_

  + `Installing Python via Conda <docs/environment/install_python_conda.rst>`_

  + `Installing Python via PyEnv <docs/environment/install_python_pyenv.rst>`_

* Fusion Related Docs:

  + `TA2 Fusion Overview <docs/algorithms/fusion_overview.rst>`_

  + `TA2 Deep Dive Info <docs/algorithms/ta2_deep_dive_info.md>`_

  + `TA2 Feature Integration <docs/development/ta2_feature_integration.md>`_

* Older Design Docs:

  + `Structure Proposal <docs/misc/structure_proposal.md>`_

* Tutorials:

  + Tutorial 1: `Toy RGB Fusion Model Example <tutorial/tutorial1_rgb_network.sh>`_

  + Tutorial 2: `Toy MSI Fusion Model Example <tutorial/tutorial2_msi_network.sh>`_

  + Tutorial 3: `Feature Fusion Tutorial <tutorial/tutorial3_feature_fusion.sh>`_

  + Tutorial 4: `Misc Training Tutorial <tutorial/tutorial4_advanced_training.sh>`_


Acknowledgement
---------------

This research is based upon work supported in part by the Office of the
Director of National Intelligence (ODNI), 6 Intelligence Advanced Research
Projects Activity (IARPA), via 2021-2011000005. The views and conclusions
contained herein are those of the authors and should not be interpreted as
necessarily representing the official policies, either expressed or implied, of
ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to
reproduce and distribute reprints for governmental purposes notwithstanding any
copyright annotation therein


.. |main-pipeline| image:: https://gitlab.kitware.com/smart/watch/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/smart/watch/-/pipelines/main/latest
.. |main-coverage| image:: https://gitlab.kitware.com/smart/watch/badges/main/coverage.svg
   :target: https://gitlab.kitware.com/smart/watch/badges/main/coverage.svg
