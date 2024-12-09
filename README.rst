GeoWATCH - Geographic Wide Area Terrestrial Change Hypercube
============================================================


.. https://ipfs.io/ipfs/bafybeia3xfmrj2mzgg5jwlxvhpoi6vuyftyphbdezppbpjgn4uqvqtxlcu/smart_watch.svg
.. https://ipfs.io/ipfs/QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq

.. The large version wont work because github strips rst image rescaling.
.. .. image:: https://ipfs.io/ipfs/QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq
.. .. image:: https://ipfs.io/ipfs/bafybeia3xfmrj2mzgg5jwlxvhpoi6vuyftyphbdezppbpjgn4uqvqtxlcu/smart_watch.svg

.. .. image:: https://i.imgur.com/0HESHf7.png

.. FULL SVG .. image:: https://data.kitware.com/api/v1/file/657ca7298c54f378b99229dc/download
.. FULL PNG..   image:: https://data.kitware.com/api/v1/file/657ca7698c54f378b99229e9/download
.. 64-THUMBNAIL .. image:: https://data.kitware.com/api/v1/file/657ca7df8c54f378b99229ee/download
.. 128-THUMBNAIL .. image:: https://data.kitware.com/api/v1/file/657ca8a78c54f378b99229f5/download

.. image:: https://data.kitware.com/api/v1/file/657ca8a78c54f378b99229f5/download
   :height: 64px
   :align: left

|GitlabCIPipeline| |GitlabCICoverage| |Pypi| |PypiDownloads| |ReadTheDocs|

GeoWATCH is an open source research and production environment for image and
video segmentation and detection with geospatial awareness.


This repository addresses the algorithmic challenges of the
`IARPA SMART <https://www.iarpa.gov/research-programs/smart>`_ (Space-based
Machine Automated Recognition Technique) program.  The goal of this software is
analyze space-based imagery to perform broad-area search for natural and
anthropogenic events and characterize their extent and progression in time and
space.


The following table provides links to relevant resources for the SMART WATCH project:

+----------------------------------------------------------+----------------------------------------------------------------+
| The GeoWATCH Gitlab Repo                                 | https://gitlab.kitware.com/computer-vision/geowatch/           |
+----------------------------------------------------------+----------------------------------------------------------------+
| Pypi                                                     | https://pypi.org/project/geowatch/                             |
+----------------------------------------------------------+----------------------------------------------------------------+
| Read the docs                                            | https://geowatch.readthedocs.io                                |
+----------------------------------------------------------+----------------------------------------------------------------+
| Slides                                                   | `Software Overview Slides`_  and `KHQ Demo Slides`_            |
+----------------------------------------------------------+----------------------------------------------------------------+
| Blog Post                                                | https://www.kitware.com/geowatch/                              |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Phase 2 Internal SMART GeoWATCH DVC Data Repo        | https://gitlab.kitware.com/smart/smart_data_dvc/               |
+----------------------------------------------------------+----------------------------------------------------------------+
| The Phase 2 Internal SMART GeoWATCH DVC Experiment Repo  | https://gitlab.kitware.com/smart/smart_expt_dvc/               |
+----------------------------------------------------------+----------------------------------------------------------------+

.. _Software Overview Slides: https://docs.google.com/presentation/d/125kMWZIwfS85lm7bvvCwGAlYZ2BevCfBLot7A72cDk8/

.. _KHQ Demo Slides: https://docs.google.com/presentation/d/1HKH_sGJX4wH60j8t4iDrZN8nH71jGX1vbCXFRIDVI7c/

Purpose & Features
------------------

GeoWATCH can be used to train, predict, and evaluate segmentation models on
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
file or in the code. In this way GeoWATCH seeks to run on the input data the
user provides, rather than make assumptions about it. The only restriction is
that the data must be registered in a
`kwcoco <https://gitlab.kitware.com/computer-vision/kwcoco>`_ file, which is
easy to do, can co-exist with arbitrary other on-disk structures, and has many
benefits. Images can be arbitrarily large or small, and can be used in-situ
(i.e. the raw images need not be pre-processed in any way), although some
formats (e.g. COGs) will be more efficient than others.


.. .. Slides:
.. .. * `KQH Demo Slides <https://docs.google.com/presentation/d/1HKH_sGJX4wH60j8t4iDrZN8nH71jGX1vbCXFRIDVI7c/edit#slide=id.p>`_.
.. .. * `GeoWATCH Software Overview <https://docs.google.com/presentation/d/125kMWZIwfS85lm7bvvCwGAlYZ2BevCfBLot7A72cDk8/edit#slide=id.g282ae2e4546_0_5>`_.

Use Case: Heavy Construction
----------------------------

The motivating use-case for this software is detection of heavy construction
events and the classification of their phases.


The first image illustrates the broad area search (BAS) component of the
pipeline that uses low spatio-temporal resolution data to detect candidate
"salient" regions for further processing.

.. .. image:: https://i.imgur.com/tilGphj.gif
.. image:: https://data.kitware.com/api/v1/file/657ca9778c54f378b99229fa/download
   :height: 100px
   :align: left

The next main component of the system is activity characterization (AC) where
higher resolution data is used to refine predicted information. In this case we
classify each polygon as a different phase of construction. In the above
example there are 3 detected candidates. We now zoom in on the one in the
middle.

.. .. image:: https://i.imgur.com/2EBpDGZ.gif
.. image:: https://data.kitware.com/api/v1/file/657ca9788c54f378b99229fd/download
   :height: 100px
   :align: left

This shows the system detecting the construction of the KHQ building and
classifying its phases. This demo was run on public data, and can be reproduced
with `Tutorial 6 <docs/source/manual/tutorial/tutorial6_predict_KHQ.sh>`_. The system was not
trained on this region.


Publicly released model weights indexed via the decentralized
`IPFS <https://en.wikipedia.org/wiki/InterPlanetary_File_System>`_
protocol as well as through Kitware's centralized
`Girder <https://girder.readthedocs.io/en/latest/>`_ system.
Note IPFS links use the ipfs.io gateway, but the CID can be used to accesss the data directly. See
`IPFS tutorials <https://docs.ipfs.tech/how-to/desktop-app/#install-ipfs-desktop>`_ for details.


+--------------------------------------------------------------------------------------------------------------+
| Publicly Released Model Weights                                                                              |
+---------------+----------------------------------------------------------------+-----------------------------+
| Release Date  | IPFS                                                           | Girder                      |
+===============+================================================================+=============================+
| 2024-01-11    | `bafybeiclo3c4bnhuumj77nxzodth442ybovw77cvbzp7ue23lsfnw4tyxa`_ | `65a94833d5d9e43895a66505`_ |
+---------------+----------------------------------------------------------------+-----------------------------+


.. _bafybeiclo3c4bnhuumj77nxzodth442ybovw77cvbzp7ue23lsfnw4tyxa: https://ipfs.io/ipfs/QmQonrckXZq37ZHDoRGN4xVBkqedvJRgYyzp2aBC5Ujpyp?redirectURL=bafybeiclo3c4bnhuumj77nxzodth442ybovw77cvbzp7ue23lsfnw4tyxa&autoadapt=0&requiresorigin=0&web3domain=0&immediatecontinue=1&magiclibraryconfirmation=0
.. _65a94833d5d9e43895a66505: https://data.kitware.com/#item/65a94833d5d9e43895a66505



System Requirements
-------------------

Before you start you must have
`installed Python <docs/source/manual/environment/install_python.rst>`_.
We currently support CPython versions 3.10 and 3.11.

Getting Started
---------------

The ``geowatch`` package is available on pypi and can be installed with pip.
To install a barebones version of geowatch with limited features and
dependencies, run:


.. code:: bash

   pip install geowatch[headless]

Note that it is import to specify "headless", to indicate that the
`opencv-python-headless <https://pypi.org/project/opencv-python-headless/>`_
variant of opencv that should be used. Alternatively, you could specify
"graphics" to use the
`opencv-python <https://pypi.org/project/opencv-python/>`_ variant, but we have
found that this can cause conflicts with Qt libraries.

Alternatively, for a fully featured install of GeoWATCH run:

.. code:: bash

   pip install geowatch[headless,optional,development,tests]


After installing ``geowatch`` from from pypi, you will have access to the
GeoWATCH command line interface (CLI).  At this point you should be able to use
the CLI to list available commands:

.. code:: bash

   geowatch --help

Unfortunately, the install is not complete. This is because binary wheels for
`GDAL <https://gdal.org/index.html>`_ are not available on pypi, and this means
we cannot access them at GeoWATCH install-time. Fortunately, Kitware
`hosts binary GDAL wheels <https://girder.github.io/large_image_wheels>`_, and
GeoWATCH provides a tool to install them and complete its installation.

.. code:: bash

    geowatch finish_install

If you use the fully featured install command (which can be run even if
GeoWATCH is already installed), or have at least installed
`xdoctest <https://github.com/Erotemic/xdoctest>`_, you can test that your
install is functioning correctly by running the doctests in the ``geowatch``
module:

.. code:: bash

    xdoctest -m geowatch


The GeoWATCH CLI has support for tab-complete, but this feature needs to `be enabled <docs/source/manual/development/coding_environment.rst>`_.

For more details see the `installing GeoWATCH for development guide <docs/source/manual/environment/installing_geowatch.rst>`_.

We also have limited windows support, see `installing GeoWATCH on Windows  <docs/source/manual/environment/windows.rst>`_.


Tutorials
---------

We have a set of `tutorials <./docs/source/manual/tutorial>`_ related to training models and predicting with the
system.

* Tutorial 1: `Toy RGB Fusion Model Example <docs/source/manual/tutorial/tutorial1_rgb_network.sh>`_

* Tutorial 2: `Toy MSI Fusion Model Example <docs/source/manual/tutorial/tutorial2_msi_network.sh>`_

* Tutorial 3: `Feature Fusion Tutorial <docs/source/manual/tutorial/tutorial3_feature_fusion.sh>`_

* Tutorial 4: `Misc Training Tutorial <docs/source/manual/tutorial/tutorial4_advanced_training.sh>`_

* Tutorial 5: `KR2 BAS SMART Demo <docs/source/manual/tutorial/tutorial5_bas_prediction.sh>`_

* Tutorial 6: `KHQ SMART Demo <docs/source/manual/tutorial/tutorial6_predict_KHQ.sh>`_


Documentation
-------------

For quick reference, a list of current documentation files is:

* `Onboarding Docs <docs/source/manual/onboarding.rst>`_

* `Internal Resources <docs/source/manual/data/internal_resources.rst>`_

* `The GeoWATCH CLI <docs/source/manual/watch_cli.rst>`_

* Contribution:

  + `Contribution Instructions <docs/source/manual/development/contribution_instructions.rst>`_

  + `Rebasing Procedure <docs/source/manual/development/rebasing_procedure.rst>`_

  + `Testing Practices <docs/source/manual/testing/testing_practices.rst>`_

  + `Supporting Projects <docs/source/manual/misc/supporting_projects.rst>`_

  + `Coding Conventions <docs/source/manual/development/coding_conventions.rst>`_

* Installing:

  + `Installing GeoWATCH <docs/source/manual/environment/installing_geowatch.rst>`_

  + `Installing GeoWATCH on Windows <docs/source/manual/environment/windows.rst>`_

  + `Installing Python via Conda <docs/source/manual/environment/install_python_conda.rst>`_

  + `Installing Python via PyEnv <docs/source/manual/environment/install_python_pyenv.rst>`_

* Fusion Related Docs:

  + `TA2 Fusion Overview <docs/source/manual/algorithms/fusion_overview.rst>`_

  + `TA2 Deep Dive Info <docs/source/manual/algorithms/ta2_deep_dive_info.md>`_

  + `TA2 Feature Integration <docs/source/manual/development/ta2_feature_integration.md>`_

* Older Design Docs:

  + `Structure Proposal <docs/source/manual/misc/structure_proposal.md>`_


Development
-----------

For new collaborators, please refer to the `onboarding docs <docs/source/manual/onboarding.rst>`_

For internal collaborators, please refer to the `internal docs <docs/source/manual/data/internal_resources.rst>`_

For more details about the GeoWATCH CLI and other CLI tools included in this package see:
`the GeoWATCH CLI docs <docs/source/manual/watch_cli.rst>`_

The ``geowatch`` module is built on top of several other
`supporting libraries <docs/source/manual/misc/supporting_projects.rst>`_
developed by Kitware. Familiarity with these packages will make it easier to
understand the GeoWATCH codebase.
Particularly, developers should be have some familiarity with
`kwcoco <https://gitlab.kitware.com/computer-vision/kwcoco>`_,
`kwimage <https://gitlab.kitware.com/computer-vision/kwimage>`_,
`scriptconfig <https://gitlab.kitware.com/utils/scriptconfig>`_, and
`ubelt <https://github.com/Erotemic/ubelt>`_.
Also helpful is familiarity with
`ndsampler <https://gitlab.kitware.com/computer-vision/ndsampler>`_,
`delayed_image <https://gitlab.kitware.com/computer-vision/delayed_image>`_,
`cmd_queue <https://gitlab.kitware.com/computer-vision/cmd_queue>`_, and
`xdoctest <https://github.com/Erotemic/xdoctest>`_.



Related Work
------------

There are other GIS and segmentation focused torch packages out there:

* `TorchGeo <https://github.com/microsoft/torchgeo>`_ - Torch geo provides many custom
  dataloaders for standard datasets. In contrast, we provide a single data
  loader for kwcoco files.

* `Raster Vision <https://github.com/azavea/raster-vision>`_ - A similar framework. One major difference is that ours provides the ability to construct multi-sensor batches with heterogeneous resolutions (using the help of `delayed_image <https://gitlab.kitware.com/computer-vision/delayed_image>`_).

* `MMSegmentation <https://github.com/open-mmlab/mmsegmentation>`_ -
  Contains standardized models with flexible model configuration.
  We use some the mmlabs models, but their library doesn't have the data
  flexibility (e.g. large image support) that kwcoco provides.


Published Research
------------------

* IGARSS 2024: `Slides <IGARSS 2024 Slides_>`__, `Conference Page <https://2024.ieeeigarss.org/view_paper.php?PaperNum=5431>`_, `Paper (ArXiV) <IGARSS 2024 ArXiV Paper_>`__, `Paper (DeSci) <IGARSS 2024 DeSci Paper_>`__

* WACV 2024: `Open Access Paper <WACV 2024 Paper_>`__.

.. _IGARSS 2024 Slides: https://docs.google.com/presentation/d/1DVcXlIUEt95rT9y6IB5UqUxoL99bubDwB-qd-F81pOU/edit#slide=id.g2e7e44f7987_0_7985

.. _WACV 2024 Paper: https://openaccess.thecvf.com/content/WACV2024/html/Greenwell_WATCH_Wide-Area_Terrestrial_Change_Hypercube_WACV_2024_paper.html

.. _IGARSS 2024 ArXiV Paper: https://arxiv.org/abs/2407.06337

.. _IGARSS 2024 DeSci Paper: https://nodes.desci.com/node/Bcb-oq85_EzGTZvwAHo8Xj9FmbHujHEqJLW3e8ljlq4


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


.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/geowatch/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/geowatch/-/pipelines/main/latest

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/geowatch/badges/main/coverage.svg
   :target: https://gitlab.kitware.com/computer-vision/geowatch/badges/main/coverage.svg

.. |Pypi| image:: https://img.shields.io/pypi/v/geowatch.svg
   :target: https://pypi.python.org/pypi/geowatch

.. |PypiDownloads| image:: https://img.shields.io/pypi/dm/geowatch.svg
   :target: https://pypistats.org/packages/geowatch

.. |ReadTheDocs| image:: https://readthedocs.org/projects/geowatch/badge/?version=latest
    :target: http://geowatch.readthedocs.io/en/latest/
