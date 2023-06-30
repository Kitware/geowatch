GEOWATCH - Geographic Wide Area Terrestrial Change Hypercube
============================================================

.. The large version wont work because github strips rst image rescaling.
.. image:: https://ipfs.io/ipfs/QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq
   :height: 100px
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

We have a set of tutorials related to training models and predicting with the
system.

* Tutorial 1: `Toy RGB Fusion Model Example <tutorial/tutorial1_rgb_network.sh>`_

* Tutorial 2: `Toy MSI Fusion Model Example <tutorial/tutorial2_msi_network.sh>`_

* Tutorial 3: `Feature Fusion Tutorial <tutorial/tutorial3_feature_fusion.sh>`_

* Tutorial 4: `Misc Training Tutorial <tutorial/tutorial4_advanced_training.sh>`_


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

For new collaberators, please refer to the `onboarding docs <docs/onboarding.rst>`_

For internal collaberators, please refer to the `internal docs <docs/data/internal_resources.rst>`_

For more details about the GEOWATCH CLI and other CLI tools included in this package see:
`the GEOWATCH CLI docs <docs/watch_cli.rst>`_


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
.. |Pypi| image:: https://img.shields.io/pypi/v/geowatch.svg
   :target: https://pypi.python.org/pypi/geowatch
.. |Downloads| image:: https://img.shields.io/pypi/dm/geowatch.svg
   :target: https://pypistats.org/packages/geowatch
.. |ReadTheDocs| image:: https://readthedocs.org/projects/geowatch/badge/?version=latest
    :target: http://geowatch.readthedocs.io/en/latest/
