***********************************
Accessing Phase3 GeoWATCH DVC Repos
***********************************

This document was written on 2024-01-23 and is relevant for phase 3 data.
For phase2 data instructions see: `old phase 2 document <./access_dvc_repos.rst>`_.

For Phase 3, there are two primary internal DVC repos:

* The Phase 3 Internal SMART GeoWATCH DVC **Data** Repo:  https://gitlab.kitware.com/smart/smart_phase3_data/

* The Phase 3 Internal SMART GeoWATCH DVC **Experiment** Repo: https://gitlab.kitware.com/smart/smart_phase3_expt/


Prerequistes
------------

To clone the DVC repos you must have access to `gitlab.kitware.com/smart <https://gitlab.kitware.com/smart>`_.
If you do not have permission please contact someone at Kitware to gain access and have them add you to the smart group.

Once you have access to `gitlab.kitware.com/smart <https://gitlab.kitware.com/smart>`_, ensure that you
have ssh keys setup and registered with gitlab. More details on generating ssh
keys and registering them with gitlab can be found in the
`ssh setup instructions <../environment/getting_started_ssh_keys.rst>`_.

To access the internal DVC remotes you must have AWS credentials.
For details see `the aws getting started docs <../environment/getting_started_aws.rst>`_.


You should also have DVC installed.
See `getting started with dvc <../environment/getting_started_dvc.rst>`_
if you are unfamiliar with the concepts of DVC.


Clone the Repos
---------------

Assuming you have your ssh keys registered with gitlab.kitware.com, and you are
a member of the smart group, you should be able to clone the repos with ssh
credentials.


We recommend using ``$HOME/data/dvc-repos`` as the location for storing the DVC
repos, but we will abstract this with an environment variable
``DVC_REPOS_DIR``, that you can change to the location you want to store the
data. (Note: that some geowatch tools can auto-detect DVC repos if they are
in the recommended locations).


.. code:: bash

   # Ensure you have git on your system
   dpkg -l git > /dev/null || sudo apt install git -y

   # This is the recommended location to checkout DVC repos. Change as needed
   DVC_REPOS_DIR=$HOME/data/dvc-repos
   mkdir -p "$DVC_REPOS_DIR"

   # Clone the Data DVC Repo
   git clone git@gitlab.kitware.com:smart/smart_phase3_data.git $DVC_REPOS_DIR/smart_phase3_data

   # Clone the Experiment DVC Repo
   git clone git@gitlab.kitware.com:smart/smart_phase3_expt.git $DVC_REPOS_DIR/smart_phase3_expt


The clone should be very fast. A DVC repo is just a git repo that contains
*pointers* to data that lives elsewhere. The next section provides instructions
on how to access that data.


Register Repos
--------------

To keep instructions machine agnostic, we want to register the paths to the DVC
repos with the ``geowatch_dvc`` tool (which may soon be superseded by the
`simple_dvc <https://gitlab.kitware.com/computer-vision/simple_dvc>`_ ``sdvc`` tool).


In order to make reproducing results as easy as copy/pasting commands into a
terminal, we provide the ``geowatch_dvc`` tool to register the paths to their
DVC repos as follows:

When you register your data / experiment paths, the DVC examples in this repo
will generally work out of the box. The important part is that your path agrees
with the tags used in the examples.

.. code:: bash

   # This is the recommended location to checkout DVC repos. Change as needed
   DVC_REPOS_DIR=$HOME/data/dvc-repos

   # If you cloned to an ssd, change this to "ssd" instead. In the future we
   # should try to auto-detect this.
   HARDWARE=hdd

   # Register the path you cloned the smart_data_dvc and smart_expt_dvc repositories to.
   geowatch_dvc add smart_phase3_data --path="$DVC_REPOS_DIR/smart_phase3_data" --tags=phase3_data --hardware="$HARDWARE"
   geowatch_dvc add smart_phase3_expt --path="$DVC_REPOS_DIR/smart_phase3_expt" --tags=phase3_expt --hardware="$HARDWARE"


You can check what repos have been registered with the "list" command:

.. code:: bash

   geowatch_dvc list


The examples in this repo will generally use this pattern to query for the
machine-specific data location. Ensure that these commands work and output
the correct paths

.. code:: bash

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)

    # Test to make sure these work.
    echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
    echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"


Access Data
-----------

Now that you have the DVC repos and have cloned them, let's access some data.

This will require that you have your AWS credentials setup. By default the DVC
repos are configured to access a remote called "aws" via the iarpa aws profile.


First ensure DVC is installed with the S3 backend:

.. code:: bash

   # Ensure the latest dvc is installed
   pip install "dvc[s3]" -U


Now, navigate to the repo. We will pull the data for KR_R001 in the
Aligned-Drop8-ARA version of the data.

.. code:: bash

    # Navigate to the kwcoco bundle
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=auto)
    cd $DVC_DATA_DPATH

    # List the files that exist
    ls

    # List files inside the dataset of interest
    ls Aligned-Drop8-ARA

    # List files inside a region of interest
    ls Aligned-Drop8-ARA/KR_R001

You will notice that there are several folders and some ".dvc" files. We need
to use these to access the data they are pointing to. We can do this by pulling
the data onto the machine.

First lets pull both of the kwcoco files and the landsat (L8) images from the "aws" remote.

.. code:: bash

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=auto)
    cd $DVC_DATA_DPATH/Aligned-Drop8-ARA/KR_R001
    dvc pull -r aws -- *.kwcoco.zip.dvc  L8.dvc


The data is currently setup such that there is a DVC file per sensor, so if you
only care about certain sensors, you only need to download that relevant data.
However, you will likely need to filter non-existing images out of the kwcoco
file (e.g. to select only landsat images use ``kwcoco subset --src in.kwcoco.json --dst out.kwcoco.json --select_images '.sensor_coarse == "L8"'``)

Now lets pull the data for the other sensors, this will take slightly longer

.. code:: bash

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=auto)
    cd $DVC_DATA_DPATH/Aligned-Drop8-ARA/KR_R001
    dvc pull -r aws -- S2.dvc WV.dvc

Check that all the data for this region pulled correctly:


.. code:: bash

   kwcoco validate -- *.kwcoco.zip
