*******************
GEOWATCH Onboarding
*******************

The purpose of this document is to guide a new user in the installation and
usage of the GEOWATCH system.

This document assumes you are proficient with Python and have an understanding
of virtual environments.


For internal collaberators you will want to clone this geowatch Python repo, and
the two DVC repos one for data and the other for experimental results. We
recommend the following directory structure:

.. code:: bash

   # Create a directory for CODE
   mkdir -p $HOME/code

   # Create a directory for DATA
   # Either:
   mkdir -p $HOME/data/

   # XOR: create a symlink to a drive that has space on it.
   ln -s /data $HOME/data

   # Clone the code repos
   # Internal
   # git clone https://gitlab.kitware.com/smart/watch/  $HOME/code/watch

   # Public
   git clone https://gitlab.kitware.com/computer-vision/geowatch/  $HOME/code/watch

   # Clone the data repos
   mkdir -p $HOME/data/dvc-repos
   git clone https://gitlab.kitware.com/smart/smart_data_dvc/ $HOME/data/dvc-repos/smart_data_dvc
   git clone https://gitlab.kitware.com/smart/smart_expt_dvc/ $HOME/data/dvc-repos/smart_expt_dvc

   # Or if you have credentials
   git clone git@gitlab.kitware.com:smart/smart_data_dvc.git $HOME/data/dvc-repos/smart_data_dvc
   git clone git@gitlab.kitware.com:smart/smart_expt_dvc.git $HOME/data/dvc-repos/smart_expt_dvc


For details on installing the geowatch system in development mode see
`installing geowatch for development guide <../docs/environment/installing_geowatch.rst>`_.

For more detailed instructions on setting up the DVC repos see:
`accessing dvc repos <../docs/data/access_dvc_repos.rst>`_.


This onboarding document is a work in progress. While things are are still
being organized, here is a list of current documentation files:


For quick reference, review the list of current documentation files in the `README <../README.rst>`_


Please ensure you review :

  + `Contribution Instructions <../docs/development/contribution_instructions.rst>`_

  + `Rebasing Procedure <../docs/development/rebasing_procedure.rst>`_

  + `Testing Practices <../docs/testing/testing_practices.rst>`_

  + `Coding Conventions <../docs/development/coding_conventions.rst>`_

  + `Supporting Projects <../docs/misc/supporting_projects.rst>`_


TODO: Point to the FAQ examples in kwcoco and other projects with them


Tutorials
---------

The following tutorials detail how to train simple fusion models


* Tutorial 1: `Toy RGB Tutorial <../tutorial/tutorial1_rgb_network.sh>`_

* Tutorial 2: `Toy MSI Tutorial <../tutorial/tutorial2_msi_network.sh>`_

* Tutorial 3: `Feature Fusion Tutorial <../tutorial/tutorial3_feature_fusion.sh>`_

* Tutorial 4: TODO: tutorial about kwcoco (See docs for `kwcoco <https://gitlab.kitware.com/computer-vision/kwcoco>`_)


Module Structure
-----------------

The current ``geowatch`` module struture is summarized as follows:

.. Generated via: python ~/code/watch/dev/maintain/repo_structure_for_readme.py

.. code:: bash

    ╙── watch {'.py': 4, '.sh': 1}
        ├─╼ cli {'.py': 47}
        │   └─╼ dag_cli {'.py': 9, '.rst': 1}
        ├─╼ demo {'.py': 8}
        │   └─╼ metrics_demo {'.py': 6}
        ├─╼ mlops {'.py': 14}
        ├─╼ stac {'.py': 4}
        ├─╼ monkey {'.py': 10}
        ├─╼ geoannots {'.py': 3}
        ├─╼ gis {'.py': 5}
        │   └─╼ sensors {'.py': 2}
        ├─╼ rc {'.json': 3, '.gtx': 1, '.xml': 1, '.py': 2}
        ├─╼ utils {'.py': 49}
        │   └─╼ lightning_ext {'.py': 11}
        │       └─╼ callbacks {'.py': 7, '.txt': 1}
        └─╼ tasks {'.py': 1}
            ├─╼ fusion {'.py': 11, '.md': 1}
            │   ├─╼ datamodules {'.py': 12}
            │   │   └─╼ temporal_sampling {'.py': 8, '.pyx': 1}
            │   ├─╼ methods {'.py': 7}
            │   └─╼ architectures {'.py': 6}
            ├─╼ dino_detector {'.py': 3, '.sh': 1}
            ├─╼ depth {'.py': 9, '.json': 1, '.md': 1}
            ├─╼ rutgers_material_seg {'.py': 5}
            │   ├─╼ datasets {'.py': 13}
            │   ├─╼ experiments {'.py': 31}
            │   ├─╼ models {'.py': 21}
            │   ├─╼ utils {'.py': 6}
            │   └─╼ scripts {'.py': 3}
            ├─╼ metrics {'.py': 3}
            ├─╼ cold {'.py': 9, '.txt': 6, '.yaml': 1}
            ├─╼ invariants {'.py': 8, '.md': 1, '': 1}
            │   └─╼ data {'.py': 4}
            ├─╼ rutgers_material_change_detection {'.py': 4, '.md': 1}
            │   ├─╼ datasets {'.py': 5}
            │   ├─╼ models {'.py': 23, '.tmp': 1}
            │   └─╼ utils {'.py': 6}
            ├─╼ landcover {'.py': 8, '.md': 1}
            ├─╼ uky_temporal_prediction {'.py': 7, '.md': 1, '.yml': 1, '': 1}
            │   ├─╼ spacenet {'.py': 2}
            │   │   └─╼ data {'.py': 2}
            │   │       └─╼ splits_unmasked {'.py': 2}
            │   └─╼ models {'.py': 4}
            └─╼ tracking {'.py': 7}

