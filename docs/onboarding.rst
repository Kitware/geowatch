****************
Watch Onboarding
****************

The purpose of this document is to guide a new user in the installation and
usage of the WATCH system.

This document assumes you are proficient with Python and have an understanding
of virtual environments.


For details on installing the watch system in development mode see 
`installing watch for development guide <../docs/installing_watch.rst.rst>`_.



This onboarding document is a work in progress. While things are are still
being organized, here is a list of current documentation files:


* `Onboarding Docs <../docs/onboarding.rst>`_
* `Internal Resources <../docs/internal_resources.rst>`_
* `The WATCH CLI <../docs/watch_cli.rst>`_

* Contribution:
  + `Contribution Instructions <../docs/contribution_instructions.rst>`_
  + `Rebasing Procedure <../docs/rebasing_procedure.md>`_
  + `Testing Practices <../docs/testing_practices.md>`_

* Installing: 
  + `Installing WATCH <../docs/installing_watch.rst>`_
  + `Installing Python via Conda <../docs/install_python_conda.rst>`_
  + `Installing Python via PyEnv <../docs/install_python_pyenv.rst>`_

* Fusion Related Docs:
  + `TA2 Fusion Overview <../docs/fusion_overview.rst>`_
  + `TA2 Deep Dive Info <../docs/ta2_deep_dive_info.md>`_
  + `TA2 Feature Integration <../docs/ta2_feature_integration.md>`_

* Older Design Docs:
  + `Structure Proposal <../docs/structure_proposal.md>`_


.. ..To contribute, please read the `contribution instructions <docs/contribution_instructions.rst>`_.
.. ..For information on testing please see `running and writing watch tests <docs/testing_practices.rst>`_.


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
