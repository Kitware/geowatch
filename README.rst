WATCH - Wide Area Terrestrial Change Hypercube
==============================================

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
| The Public SMART WATCH Python Module                     | https://gitlab.kitware.com/watch/watch/                        |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Internal SMART WATCH Python Module                   | https://gitlab.kitware.com/smart/watch/                        |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Phase 2 Internal SMART WATCH DVC Data Repo           | https://gitlab.kitware.com/smart/smart_data_dvc/               |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Phase 2 Internal SMART WATCH DVC Experiment Repo     | https://gitlab.kitware.com/smart/smart_expt_dvc/               |
+----------------------------------------------------------+----------------------------------------------------------------+


Getting Started
---------------

To quickly get started locally, clone the watch repo:


.. code:: bash

   # Create a directory for CODE
   mkdir -p $HOME/code
   git clone https://gitlab.kitware.com/smart/watch/  $HOME/code/watch


Then run:

.. code:: bash

   cd $HOME/code/watch
   bash ./run_developer_setup.sh
 

Test everything is working by running

.. code:: bash

   ./run_tests.py

For more details see the `installing watch for development guide <docs/installing_watch.rst>`_.


Development
-----------

For new collaberators, please refer to the `onboarding docs <docs/onboarding.rst>`_ 

For internal collaberators, please refer to the `internal docs <docs/internal_resources.rst>`_ 


The Watch CLI
-------------

The watch module comes with a set of command line tools. 
Using ``python -m watch --help`` or ``smartwatch --help`` shows the top level modal CLI:

.. code:: 
    usage: smartwatch [-h] [--version]
                      {add_fields,coco_add_watch_fields,align,coco_align_geotiffs,stats,watch_coco_stats,reproject,project,reproject_annotations,visualize,coco_visualize_videos,spectra,intensity_histograms,coco_spectra,dvcdir,find_dvc,kwcoco_to_geojson,iarpa_eval,model_stats,model_info,torch_model_stats,clean_geotiffs,coco_clean_geotiffs,animate,gifify}
                      ...

    The SMART WATCH CLI

    positional arguments:
      {add_fields,coco_add_watch_fields,align,coco_align_geotiffs,stats,watch_coco_stats,reproject,project,reproject_annotations,visualize,coco_visualize_videos,spectra,intensity_histograms,coco_spectra,dvcdir,find_dvc,kwcoco_to_geojson,iarpa_eval,model_stats,model_info,torch_model_stats,clean_geotiffs,coco_clean_geotiffs,animate,gifify}
                            specify a command to run
        add_fields (coco_add_watch_fields)
                            Updates image transforms in a kwcoco json file to align all videos to a
        align (coco_align_geotiffs)
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
        kwcoco_to_geojson   Convert KWCOCO to IARPA GeoJSON
        iarpa_eval          opaque sub command
        model_stats (model_info, torch_model_stats)
                            Print stats about a torch model.
        clean_geotiffs (coco_clean_geotiffs)
                            A preprocessing step for geotiff datasets.
        animate (gifify)    Convert a sequence of images into a video or gif.

    options:
      -h, --help            show this help message and exit
      --version             show version number and exit (default: False)



As a researcher / developer / user the most important commands for you to know are:

* ``smartwatch stats <kwcoco_file>`` - Get watch-relevant statistics about data in a kwcoco file

* ``smartwatch visualize <kwcoco_file>`` - Visualize the image / videos / annotations in a kwcoco file.

* ``smartwatch spectra <kwcoco_file>`` - Look at the distribution of intensity values per band / per sensor in a kwcoco file.

* ``smartwatch model_stats <fusion_model_file>`` - Get stats / info about a trained fusion model.

* ``smartwatch reproject`` - Reproject CRS84 (geojson) annoations to image space and write to a kwcoco file.

* ``smartwatch align`` - Crop a kwcoco dataset based on CRS84 (geojson) regions.

* ``smartwatch clean_geotiff`` - Heuristic to detect large regions of black pixels and edit them to NODATA in the geotiff.

* ``smartwatch geotiffs_to_kwcoco`` - Create a kwcoco file from a set of on-disk geotiffs.

* ``smartwatch_dvc`` - Helper to register / retreive your DVC paths so scripts can be written agnostic to filesystem layouts.


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

  + `Supporting Projects <docs/supporting_projects.rst>`_

  + `Coding Oddities <docs/coding_oddities.rst>`_

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


.. |main-pipeline| image:: https://gitlab.kitware.com/smart/watch/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/smart/watch/-/pipelines/main/latest
.. |main-coverage| image:: https://gitlab.kitware.com/smart/watch/badges/main/coverage.svg
   :target: https://gitlab.kitware.com/smart/watch/badges/main/coverage.svg
