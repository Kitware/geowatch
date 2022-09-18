****************
Watch Onboarding
****************

The purpose of this document is to guide a new user in the installation and
usage of the WATCH system.

This document assumes you are proficient with Python and have an understanding
of virtual environments.


For internal collaberators you will want to clone this watch Python repo, and
the two DVC repos one for data and the other for experimental results. We
recommend the following directory structure:

.. code:: bash

   # Create a directory for CODE
   mkdir -p $HOME/code

   # Create a directory for DATA
   # Either:
   mkdir -p $HOME/data  
   
   # XOR: create a symlink to a drive that has space on it.
   ln -s /data $HOME/data  

   # Clone the code repos
   git clone https://gitlab.kitware.com/smart/watch/  $HOME/code/watch

   # Clone the data repos
   git clone https://gitlab.kitware.com/smart/smart_data_dvc/  $HOME/data/smart_data_dvc
   git clone https://gitlab.kitware.com/smart/smart_expt_dvc/  $HOME/data/smart_expt_dvc


For details on installing the watch system in development mode see 
`installing watch for development guide <../docs/installing_watch.rst.rst>`_.



This onboarding document is a work in progress. While things are are still
being organized, here is a list of current documentation files:


For quick reference, review the list of current documentation files in the `README <../README.rst>`_


Please ensure you review :

  + `Contribution Instructions <docs/contribution_instructions.rst>`_

  + `Rebasing Procedure <docs/rebasing_procedure.md>`_

  + `Testing Practices <docs/testing_practices.md>`_

  + `Coding Oddities <docs/coding_oddities.rst>`_

  + `Supporting Projects <docs/supporting_projects.rst>`_


TODO: Point to the FAQ examples in kwcoco and other projects with them


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

.. Generated via: python ~/code/watch/dev/repo_structure_for_readme.py

.. code:: bash

    ╙── watch {'.py': 4}
        ├─╼ cli {'.py': 33}
        ├─╼ demo {'.py': 8}
        ├─╼ mlops {'.py': 7}
        ├─╼ stac {'.py': 3}
        ├─╼ gis {'.py': 5}
        │   └─╼ sensors {'.py': 2}
        ├─╼ rc {'.gtx': 1, '.json': 3, '.py': 2, '.xml': 1}
        ├─╼ utils {'.py': 30}
        │   └─╼ lightning_ext {'.py': 5}
        │       └─╼ callbacks {'.py': 7, '.txt': 1}
        └─╼ tasks {'.py': 1}
            ├─╼ fusion {'.md': 1, '.py': 14}
            │   ├─╼ datamodules {'.py': 7, '.pyx': 1}
            │   ├─╼ methods {'.py': 3}
            │   └─╼ architectures {'.py': 4}
            ├─╼ depth {'.json': 1, '.md': 1, '.py': 9}
            ├─╼ rutgers_material_seg {'.py': 5}
            │   ├─╼ datasets {'.py': 13}
            │   ├─╼ experiments {'.py': 31}
            │   ├─╼ models {'.py': 21}
            │   ├─╼ utils {'.py': 6}
            │   └─╼ scripts {'.py': 3}
            ├─╼ invariants {'': 1, '.md': 1, '.py': 9}
            │   └─╼ data {'.py': 3}
            ├─╼ rutgers_material_change_detection {'.md': 1, '.py': 4}
            │   ├─╼ datasets {'.py': 5}
            │   ├─╼ models {'.py': 23, '.tmp': 1}
            │   └─╼ utils {'.py': 6}
            ├─╼ landcover {'.md': 1, '.py': 9}
            ├─╼ uky_temporal_prediction {'': 1, '.md': 1, '.py': 7, '.yml': 1}
            │   ├─╼ spacenet {'.py': 2}
            │   │   └─╼ data {'.py': 2}
            │   │       └─╼ splits_unmasked {'.py': 2}
            │   └─╼ models {'.py': 4}
            └─╼ tracking {'.py': 7}
