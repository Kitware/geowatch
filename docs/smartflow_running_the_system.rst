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



How to Bake a Model into a Dockerfile
-------------------------------------

* Must be run in repo root
* Ensure whatever variant of the repo you want to be run is checked out.
* Need a base directory with a model in ``./models``.

.. code:: bash

    DOCKER_BUILDKIT=1 \
        docker build --build-arg BUILD_STRICT=1 -f dockerfiles/ta2_features.Dockerfile . \
        --tag registry.smartgitlab.com/kitware/watch/ta2:post-jan31-invariant-rescaled-debug4


In the DAG need to change path to point to the new baked in model.

Need to push container to smartgitlab




How to Bake a Model into a Pyenv Dockerfile (NEW)
-------------------------------------------------

Assuming that you have already build a pyenv docker image (see last heredoc in
../dockerfiles/watch.Dockerfile ) we will add a model to it.

.. code:: bash

   # This is the model you are interested in baking in.
   DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

   MODEL_REL_FNAME=models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
   HOST_MODEL_FPATH=$DVC_EXPT_DPATH/$MODEL_REL_FNAME
   CONTAINER_MODEL_FPATH=/smart_expt_dvc/$MODEL_REL_FNAME
   CONTAINER_MODEL_DNAME=$(python -c "import pathlib; print(str(pathlib.Path('$CONTAINER_MODEL_FPATH').parent))")
   echo $HOST_MODEL_FPATH
   echo $CONTAINER_MODEL_DNAME
   echo $CONTAINER_MODEL_FPATH

   # This is the name of the pyenv watch image that you built
   IMAGE_NAME=watch:311-loose

   NEW_IMAGE_NAME=${IMAGE_NAME}-baked

   # Run the base image as a container so we can put stuff into it
   docker run -td --name temp_container $IMAGE_NAME
   docker exec -t temp_container mkdir -p "$CONTAINER_MODEL_DNAME"
   docker cp $HOST_MODEL_FPATH "temp_container:/$CONTAINER_MODEL_FPATH"

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
