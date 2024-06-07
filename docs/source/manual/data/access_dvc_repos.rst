****************************
Accessing GeoWATCH DVC Repos
****************************

NOTE (2024-01-23): See the `new Phase 3 DVC repo docs <./access_internal_phase3_dvc_repos.rst>`_ for the latest data.

This document outlines how to access the GeoWATCH DVC repos for internal
collaberators.

DVC stands for Data Version Control, and is a layer on top of git that helps
manage larger data files. For more information on DVC see
`getting started with dvc <getting_started_dvc.rst>`_.

.. note: As the system expands these docs should also expand to detail how to use public DVC repos.


As of 2022-09-29 there are two primary internal DVC repos:

* The Phase 2 Internal SMART GeoWATCH DVC **Data** Repo:  https://gitlab.kitware.com/smart/smart_data_dvc/

* The Phase 2 Internal SMART GeoWATCH DVC **Experiment** Repo: https://gitlab.kitware.com/smart/smart_expt_dvc/


Note: There is an additional repo for Drop7 Cropped AC/SC data:

* The Phase 2 Internal SMART GeoWATCH DVC AC/SC **Data** Repo:  https://gitlab.kitware.com/smart/smart_drop7


As of 2024-03-01 there are two primary internal DVC repos:

* The Phase 3 Internal SMART GeoWATCH DVC **Data** Repo:  https://gitlab.kitware.com/smart/smart_phase3_data/

* The Phase 3 Internal SMART GeoWATCH DVC **Experiment** Repo: https://gitlab.kitware.com/smart/smart_phase3_expt/


This document will outline how to clone the DVC repos, and then how to pull
relevant data from them.

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
   git clone git@gitlab.kitware.com:smart/smart_data_dvc.git $DVC_REPOS_DIR/smart_data_dvc

   # Clone the Experiment DVC Repo
   git clone git@gitlab.kitware.com:smart/smart_expt_dvc.git $DVC_REPOS_DIR/smart_expt_dvc


The clone should be very fast. A DVC repo is just a git repo that contains
*pointers* to data that lives elsewhere. The next section provides instructions
on how to access that data.


Access data in the Data DVC repo
--------------------------------

Assuming you have cloned the data DVC repo the next step is to access data in it.

This will require that you have your AWS credentials setup. By default the DVC
repos are configured to access a remote called "aws" via the iarpa aws profile.


First ensure DVC is installed with the S3 backend:

.. code:: bash

   # Ensure dvc is installed
   pip install dvc[s3]


To start lets pull the data associated with one the BAS "Drop4" datasets. This
is part the "Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC" kwcoco bundle. We will
refer to this with an environment variable ``DATASET_CODE``.


.. code:: bash

    # Navigate to the kwcoco bundle
    DVC_REPOS_DIR=$HOME/data/dvc-repos
    DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC

    cd $DVC_REPOS_DIR/smart_data_dvc/$DATASET_CODE

    # List the files that exist
    ls

You will notice that there are several folders and some ".dvc" files. We need
to use these to access the data they are pointing to.

Currently (as of 2022-09-29) the annotations are pointed to by the
"splits.zip.dvc" file and the images for each region are pointed to by their
own DVC file.

Lets start by grabbing the kwcoco annotation files. The following command will
pull the data pointed to by the ``splits.zip.dvc`` file from the ``aws`` DVC
remote.

.. code:: bash

    dvc pull -r aws splits.zip.dvc


This should download in a few seconds.  Now if you ``ls`` you should see
``splits.zip``. Unzip the kwcoco files from this archive.

.. code:: bash

    unzip splits.zip

Now if you ``ls`` you should see
``data_train.kwcoco.json`` ``data.kwcoco.json`` and ``data_vali.kwcoco.json``.

Note that we only have the kwcoco files, we still have not pulled any of the
images that they point to.

To inspect these files we need to ensure we have kwcoco installed. So ``pip
install kwcoco`` if needed.

Now, if you were to run:

.. code:: bash

   kwcoco validate data_vali.kwcoco.json

You will see that there are 17714 missing images.

To get started more quickly, lets only work with a subset of the data. We can
make a new kwcoco file that only points to landsat8 data in "KR_R001" via the
``kwcoco subset`` command:


.. code:: bash

   kwcoco subset \
       --src data_vali.kwcoco.json \
       --dst data_KR_R001.kwcoco.json \
       --select_videos '.name == "KR_R001"' \
       --select_images '.sensor_coarse == "L8"'

Running ``kwcoco validate data_KR_R001.kwcoco.json`` on this file will now report only 1705 missing images,
which will correspond to the data pointed to by the ``KR_R001/L8.dvc`` file.
To obtain this data we can run:

.. code:: bash

    dvc pull -r aws KR_R001/L8.dvc


This will take a bit longer, but likely no more than a minute or two. Now running:

.. code:: bash

   kwcoco validate data_KR_R001.kwcoco.json


will report no issues.

Using ``kwcoco stats data_KR_R001.kwcoco.json`` will provide some information about the dataset.


We could use ``kwcoco show data_KR_R001.kwcoco.json`` to inspect the data, but
because this is MSI imagery it would be more appropriate to use
``geowatch visualize data_KR_R001.kwcoco.json``
(assuming the geowatch system has been installed).
Likewise, ``geowatch stats data_KR_R001.kwcoco.json`` can provide more geowatch-relevant information.


It is now possible to use this kwcoco file for testing purposes.

Obtaining the rest of the data is similar: simply use ``dvc pull``, and keep in
mind ``kwcoco subset`` is a useful tool for taking out only a smaller part of
the data.


To download all of the data in a directory run with the `-R` flag for recursive.

.. code:: bash

    dvc pull -r aws -R .


After this downloads, any of the kwcoco files in the directory can be used.


We recommend using ``geowatch_dvc`` tool to register the path you cloned
these repos to as illustrated in `using_geowatch_dvc <./using_geowatch_dvc.rst>`_.
