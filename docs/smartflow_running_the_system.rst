Prerequisites
-------------

* `AWS <getting_started_aws.rst>`

* `Kubectl <getting_started_kubectl.rst>`

* `Smartflow Setup <getting_started_smartflow.rst>`


NOTE: this file is currently a working notes document.


SeeAlso
-------

* `Connors Smartflow Training Nodes <smartflow_training_fusion_models.md>`

* Dags live in: https://gitlab.kitware.com/smart/watch-smartflow-dags


High level:

    * build container

    * bake model into it

    * push container into T&E gitlab

    * tweak dags to point to new container



How to Bake a Model into a Dockerfile (OLD)
-------------------------------------------

* Must be run in repo root
* Ensure whatever variant of the repo you want to be run is checked out.
* Need a base directory with a model in ``./models``.

.. code:: bash

    DOCKER_BUILDKIT=1 \
        docker build --build-arg BUILD_STRICT=1 -f dockerfiles/ta2_features.Dockerfile . \
        --tag registry.smartgitlab.com/kitware/watch/ta2:post-jan31-invariant-rescaled-debug4


In the DAG need to change path to point to the new baked in model.

Need to push container to smartgitlab



Building the Pyenv Docker Image (NEW)
-------------------------------------

If you have not built a docker image, we will need to do so.

There are two images that need to be build, first the
`pyenv dockerfile <../dockerfiles/pyenv.Dockerfile>`_.
And then the watch dockerfile that builds on top of it
`watch dockerfile <../dockerfiles/watch.Dockerfile>`_. The heredocs in these
files provide futher instructions.

Here we will go over the basic use case for a specific version of Python /
dependencies. We will use Python 3.11 and strict dependencies. We will assume
that your watch repo is in ``~/code/watch``, but if it is not then change the
environment variable:

.. code:: bash

   WATCH_REPO_DPATH=$HOME/code/watch

   # Create a directory on the host for context
   mkdir -p $HOME/temp/container-staging

   # For pyenv we will use an empty directory
   mkdir -p $HOME/temp/container-staging/empty

   DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t pyenv:311 \
        --build-arg PYTHON_VERSION=3.11.2 \
        -f $WATCH_REPO_DPATH/dockerfiles/pyenv.Dockerfile \
        $HOME/temp/container-staging/empty

First, to build the pyenv image

.. code:: bash



How to Bake a Model into a Pyenv Dockerfile (NEW)
-------------------------------------------------

Assuming that you have already build a pyenv docker image we will add a model to it.


.. code:: bash

   # This is the name of the pyenv watch image that you built
   IMAGE_NAME=watch:311-loose

   NEW_IMAGE_NAME=${IMAGE_NAME}-baked-v2

   # Run the base image as a container so we can put stuff into it
   # We will use DVC to facilitate the transfer to keep things consistent
   # We mount our local experiment directory, and pull relevant files
   DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
   docker run \
       --volume $DVC_EXPT_DPATH:/host-smart_expt_dvc:ro \
       -td --name temp_container $IMAGE_NAME

   docker exec -t temp_container pip install dvc
   docker exec -t temp_container mkdir -p /root/data
   docker exec -t temp_container git clone /host-smart_expt_dvc/.git /root/data/smart_expt_dvc

   docker exec -w /root/data/smart_expt_dvc -t temp_container \
       dvc remote add host /host-smart_expt_dvc/.dvc/cache

   # Workaround DVC Issue by removing aws remote
   # References: https://github.com/iterative/dvc/issues/9264
   docker exec -w /root/data/smart_expt_dvc -t temp_container \
       dvc remote remove aws

   # Pull in relevant models you want to bake into the container
   # These will be specified relative to the experiment DVC repo
   docker exec -w /root/data/smart_expt_dvc -t temp_container \
       dvc pull --remote host \
       models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt \
       models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt \
       models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt \
       models/uky/uky_invariants_2022_12_17/TA1_pretext_model/pretext_package.pt \
       models/landcover/sentinel2.pt

   # Save the modified container as a new image
   docker commit temp_container $NEW_IMAGE_NAME

   # Cleanup the temp container
   docker stop temp_container
   docker rm temp_container

   # Push the container to smartgitlab
   docker tag $NEW_IMAGE_NAME registry.smartgitlab.com/kitware/$NEW_IMAGE_NAME
   docker push registry.smartgitlab.com/kitware/$NEW_IMAGE_NAME


Jon Notes (will turn these into docs soon):

.. code:: bash

    # SeeAlso: ~/code/watch-smartflow-dags/KIT_TA2_PYENV_TEST.py

    # git@gitlab.kitware.com:smart/watch-smartflow-dags.git
    # This is the repo containing the smartflow dags

    LOCAL_DAG_DPATH=$HOME/code/watch-smartflow-dags
    DAG_FNAME=KIT_TA2_PYENV_TEST.py

    aws s3 --profile iarpa cp $LOCAL_DAG_DPATH/$DAG_FNAME \
        s3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v4/dags/$DAG_FNAME


Running Dags After Containers are Using
---------------------------------------

Now we edit a DAG file for airflow


.. git clone git@gitlab.kitware.com:smart/watch-smartflow-dags.git


Choose a DAG file in ~/code/watch-smartflow-dags/ then edit it to give it a unique name

.e.g. ~/code/watch-smartflow-dags/KIT_TA2_20221121_BATCH.py


* change name of file and then change ``EVALUATION`` to be a unique string to name it what you want.

* change the image names / tags e.g.
    image="registry.smartgitlab.com/kitware/watch/ta2:Ph2Nov21EvalBatch", these are all "pod tasks" create_pod_task

* ``purpose`` is something about the node that it runs on.
  For a subset of valid options see: https://smartgitlab.com/blacksky/smartflow/-/blob/118140a81362c5721b5e9bb65ab967fb8bd28163/CHANGELOG.md

* make cpu limit a bit less than what is availble on the pod.

* Copy the DAG to smartflow S3:
    aws s3 --profile iarpa cp Kit_DatasetGeneration.py s3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v2/dags/Kit_DatasetGeneration.py


Need to run service to access airflow gui:

.. code:: bash

    kubectl -n airflow port-forward service/airflow-webserver 2746:8080

navigate to localhost:2746/home


Now dags show up in the GUI.
