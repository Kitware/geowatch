*******************
GeoWATCH Onboarding
*******************

The purpose of this document is to guide a new user in the installation and
usage of the GeoWATCH system.

For a high level overview of the system, see the `overview <overview.rst>`_.

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
   git clone https://gitlab.kitware.com/smart/smart_phase3_data/ $HOME/data/dvc-repos/smart_phase3_data
   git clone https://gitlab.kitware.com/smart/smart_phase3_expt/ $HOME/data/dvc-repos/smart_phase3_expt

   # Or if you have credentials
   git clone git@gitlab.kitware.com:smart/smart_phase3_data.git $HOME/data/dvc-repos/smart_phase3_data
   git clone git@gitlab.kitware.com:smart/smart_phase3_expt.git $HOME/data/dvc-repos/smart_phase3_expt


For details on installing the geowatch system in development mode see
`installing geowatch for development guide <environment/installing_geowatch.rst>`_.

For more detailed instructions on setting up the DVC repos see:
`accessing dvc repos <data/access_dvc_repos.rst>`_.


This onboarding document is a work in progress. While things are are still
being organized, here is a list of current documentation files:


For quick reference, review the list of current documentation files in the `README <../../README.rst>`_


Please ensure you review :

  + `Contribution Instructions <development/contribution_instructions.rst>`_

  + `Rebasing Procedure <development/rebasing_procedure.rst>`_

  + `Testing Practices <testing/testing_practices.rst>`_

  + `Coding Conventions <development/coding_conventions.rst>`_

  + `Supporting Projects <misc/supporting_projects.rst>`_


TODO: Point to the FAQ examples in kwcoco and other projects with them


Tutorials
---------

The following tutorials detail how to train simple fusion models


* Tutorial 1: `Toy RGB Tutorial <./tutorial/tutorial1_rgb_network.sh>`_

* Tutorial 2: `Toy MSI Tutorial <./tutorial/tutorial2_msi_network.sh>`_

* Tutorial 3: `Feature Fusion Tutorial <./tutorial/tutorial3_feature_fusion.sh>`_

* Tutorial 4: TODO: tutorial about kwcoco (See docs for `kwcoco <https://gitlab.kitware.com/computer-vision/kwcoco>`_)


Module Structure
-----------------

The current ``geowatch`` module struture is summarized as follows:

.. Generated via: python ~/code/watch/dev/maintain/repo_structure_for_readme.py

.. code:: bash

    ╙── geowatch {'.py': 4, '': 1}
        ├─╼ cli {'.py': 45}
        │   ├─╼ smartflow {'.py': 17, '.rst': 1}
        │   └─╼ special {'.py': 2}
        ├─╼ demo {'.py': 8}
        │   └─╼ metrics_demo {'.py': 6}
        ├─╼ mlops {'.py': 13}
        ├─╼ stac {'.py': 3}
        ├─╼ monkey {'.py': 10}
        ├─╼ geoannots {'.py': 3}
        ├─╼ gis {'.py': 5}
        │   └─╼ sensors {'.py': 2}
        ├─╼ rc {'.json': 3, '.gtx': 1, '.xml': 1, '.py': 2}
        │   └─╼ requirements {'.txt': 18, '.py': 1}
        ├─╼ utils {'.py': 39}
        │   └─╼ lightning_ext {'.py': 13}
        │       └─╼ callbacks {'.py': 8, '.txt': 1}
        └─╼ tasks {'.py': 1}
            ├─╼ fusion {'.py': 9, '.md': 1}
            │   ├─╼ datamodules {'.py': 10}
            │   │   └─╼ temporal_sampling {'.py': 7, '.pyx': 1}
            │   ├─╼ methods {'.py': 7}
            │   └─╼ architectures {'.py': 6}
            ├─╼ dino_detector {'.py': 3, '.sh': 1}
            ├─╼ depth {'.py': 8, '.json': 1, '.md': 1}
            ├─╼ sam {'.py': 2}
            ├─╼ rutgers_material_seg {'.py': 5}
            │   ├─╼ datasets {'.py': 13}
            │   ├─╼ models {'.py': 21}
            │   ├─╼ utils {'.py': 6}
            │   └─╼ scripts {'.py': 3}
            ├─╼ metrics {'.py': 3}
            ├─╼ cold {'.py': 11, '.yaml': 1}
            ├─╼ mae {'.py': 4, '': 1}
            ├─╼ invariants {'.py': 8, '.md': 1, '': 1}
            │   ├─╼ late_fusion {'.py': 3}
            │   ├─╼ data {'.py': 4}
            │   └─╼ utils {'.py': 6}
            ├─╼ rutgers_material_change_detection {'.py': 4, '.md': 1}
            │   ├─╼ datasets {'.py': 5}
            │   ├─╼ models {'.py': 23, '.tmp': 1}
            │   └─╼ utils {'.py': 6}
            ├─╼ landcover {'.py': 8, '.md': 1}
            ├─╼ uky_temporal_prediction {'.py': 7, '.md': 1, '.yml': 1, '': 1}
            │   └─╼ models {'.py': 4}
            ├─╼ tracking {'.py': 11}
            └─╼ depth_pcd {'.py': 5}

