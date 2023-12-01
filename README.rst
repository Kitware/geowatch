GEOWATCH - Geographic Wide Area Terrestrial Change Hypercube
============================================================


.. https://ipfs.io/ipfs/bafybeia3xfmrj2mzgg5jwlxvhpoi6vuyftyphbdezppbpjgn4uqvqtxlcu/smart_watch.svg
.. https://ipfs.io/ipfs/QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq

.. The large version wont work because github strips rst image rescaling.
.. .. image:: https://ipfs.io/ipfs/QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq
.. .. image:: https://ipfs.io/ipfs/bafybeia3xfmrj2mzgg5jwlxvhpoi6vuyftyphbdezppbpjgn4uqvqtxlcu/smart_watch.svg

.. image:: https://i.imgur.com/0HESHf7.png
   :height: 50px
   :align: left

|main-pipeline| |main-coverage| |Pypi| |Downloads|


This repository addresses the algorithmic challenges of the
`IARPA SMART <https://www.iarpa.gov/research-programs/smart>`_ (Space-based
Machine Automated Recognition Technique) program.  The goal of this software is
analyze space-based imagery to perform broad-area search for natural and
anthropogenic events and characterize their extent and progression in time and
space.


The following table provides links to relevant resources for the SMART WATCH project:

+----------------------------------------------------------+----------------------------------------------------------------+
| The GEOWATCH Gitlab Repo                                 | https://gitlab.kitware.com/computer-vision/geowatch/           |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Phase 2 Internal SMART GEOWATCH DVC Data Repo        | https://gitlab.kitware.com/smart/smart_data_dvc/               |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Phase 2 Internal SMART GEOWATCH DVC Experiment Repo  | https://gitlab.kitware.com/smart/smart_expt_dvc/               |
+----------------------------------------------------------+----------------------------------------------------------------+


Purpose & Features
------------------

Geowatch can be used to train, predict, and evaluate segmentation models on
multi-sensor image or video data.
Polygons can be extracted or "tracked" across frames in a video to produce
vectorized predictions.

Images can be in different resolutions, may be paired with geospatial metadata
(but this is not required), and have any number of sensed or derived raster
bands. Each raster band for an image need not be at the same resolution. The
only requirement is that there is an affine transform that relates each
underlying "asset space" into "image space", which in turn must be similarly
related to each "video space". These transforms and all other details of the
dataset are provided in a kwcoco file.

Dataloaders are setup to work with kwcoco files, and at train time details like
mean/std computation, classes, frequency weights are handled automatically as
opposed to common practice of hardcoding those values somewhere in a config
file or in the code. In this way geowatch seeks to run on the input data the
user provides, rather than make assumptions about it. The only restriction is
that the data must be registered in a
`kwcoco <https://gitlab.kitware.com/computer-vision/kwcoco>`_ file, which is
easy to do, can co-exist with arbitrary other on-disk structures, and has many
benefits. Images can be arbitrarily large or small, and can be used in-situ
(i.e. the raw images need not be pre-processed in any way), although some
formats (e.g. COGs) will be more efficient than others.


Slides:

* `KQH Demo Slides <https://docs.google.com/presentation/d/1HKH_sGJX4wH60j8t4iDrZN8nH71jGX1vbCXFRIDVI7c/edit#slide=id.p>`_.

* `Geowatch Software Overview <https://docs.google.com/presentation/d/125kMWZIwfS85lm7bvvCwGAlYZ2BevCfBLot7A72cDk8/edit#slide=id.g282ae2e4546_0_5>`_.

Use Case: Heavy Construction
----------------------------

The motivating use-case for this software is detection of heavy construction
events and the classification of their phases.


The first image illustrates the broad area search (BAS) component of the
pipeline that uses low spatio-temporal resolution data to detect candidate
"salient" regions for further processing.

.. image:: https://i.imgur.com/tilGphj.gif
   :height: 100px
   :align: left

The next main component of the system is activity characterization (AC) where
higher resolution data is used to refine predicted information. In this case we
classify each polygon as a different phase of construction. In the above
example there are 3 detected candidates. We now zoom in on the one in the
middle.

.. image:: https://i.imgur.com/2EBpDGZ.gif
   :height: 100px
   :align: left

This shows the system detecting the construction of the KHQ building and
classifying its phases. This demo was run on public data, and can be reproduced
with `Tutorial 6 <tutorial/tutorial6_predict_KHQ.sh>`_. The system was not
trained on this region.

Getting Started
---------------

To quickly get started locally, assuming you have `Python installed <docs/environment/install_python.rst>`_,
you can install geowatch with pip.


.. code:: bash

   pip install geowatch[headless]

   # OR for a more fully featured install use:
   pip install geowatch[headless,optional,development,tests]


This gives you access to the GEOWATCH CLI.

.. code:: bash

   geowatch --help

One library that we cannot get via the standard pip mechanism is GDAL. We have
to install this manually from the Kitware hosted GDAL large image wheels.

.. code:: bash

    pip install --prefer-binary GDAL>=3.4.1 --find-links https://girder.github.io/large_image_wheels

    # NEW in 0.8.0. Instead of using the above command you can run:
    geowatch finish_install


If you use the fully featured install command (which you can run after the
fact), you can test that your install is functioning correctly by running the
doctests:

.. code:: bash

    xdoctest watch


For more details see the `installing GEOWATCH for development guide <docs/environment/installing_geowatch.rst>`_.

We also have limited windows support, see `installing GEOWATCH on Windows  <docs/environment/windows.rst>`_.


Tutorials
---------

We have a set of `tutorials <./tutorials>`_ related to training models and predicting with the
system.

* Tutorial 1: `Toy RGB Fusion Model Example <tutorial/tutorial1_rgb_network.sh>`_

* Tutorial 2: `Toy MSI Fusion Model Example <tutorial/tutorial2_msi_network.sh>`_

* Tutorial 3: `Feature Fusion Tutorial <tutorial/tutorial3_feature_fusion.sh>`_

* Tutorial 4: `Misc Training Tutorial <tutorial/tutorial4_advanced_training.sh>`_

* Tutorial 5: `KR2 BAS SMART Demo <tutorial/tutorial5_bas_prediction.sh>`_

* Tutorial 6: `KHQ SMART Demo <tutorial/tutorial6_predict_KHQ.sh>`_


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

  + `Installing GEOWATCH <docs/environment/installing_geowatch.rst>`_

  + `Installing GEOWATCH on Windows <docs/environment/windows.rst>`_

  + `Installing Python via Conda <docs/environment/install_python_conda.rst>`_

  + `Installing Python via PyEnv <docs/environment/install_python_pyenv.rst>`_

* Fusion Related Docs:

  + `TA2 Fusion Overview <docs/algorithms/fusion_overview.rst>`_

  + `TA2 Deep Dive Info <docs/algorithms/ta2_deep_dive_info.md>`_

  + `TA2 Feature Integration <docs/development/ta2_feature_integration.md>`_

* Older Design Docs:

  + `Structure Proposal <docs/misc/structure_proposal.md>`_


Development
-----------

For new collaborators, please refer to the `onboarding docs <docs/onboarding.rst>`_

For internal collaborators, please refer to the `internal docs <docs/data/internal_resources.rst>`_

For more details about the GEOWATCH CLI and other CLI tools included in this package see:
`the GEOWATCH CLI docs <docs/watch_cli.rst>`_


Related Work
------------

There are other GIS and segmentation focused torch packages out there:

* https://github.com/microsoft/torchgeo - Torch geo provides many custom
  dataloaders for standard datasets. In contrast, we provide a single data
  loader for kwcoco files.

* https://github.com/azavea/raster-vision - based on chips, whereas ours
  focuses on the ability to process data in-situ (using the help of
  `delayed_image <https://gitlab.kitware.com/computer-vision/delayed_image>`_).

* https://github.com/open-mmlab/mmsegmentation - A very good package (and
  research group), we use some the mmlabs models, but their library doesn't
  have the data flexibility (e.g. large image support) that kwcoco provides.


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


.. |main-pipeline| image:: https://gitlab.kitware.com/computer-vision/geowatch/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/geowatch/-/pipelines/main/latest
.. |main-coverage| image:: https://gitlab.kitware.com/computer-vision/geowatch/badges/main/coverage.svg
   :target: https://gitlab.kitware.com/computer-vision/geowatch/badges/main/coverage.svg
.. |Pypi| image:: https://img.shields.io/pypi/v/geowatch.svg
   :target: https://pypi.python.org/pypi/geowatch
.. |Downloads| image:: https://img.shields.io/pypi/dm/geowatch.svg
   :target: https://pypistats.org/packages/geowatch
.. |ReadTheDocs| image:: https://readthedocs.org/projects/geowatch/badge/?version=latest
    :target: http://geowatch.readthedocs.io/en/latest/
