
SeeAlso:

    * Getting Started AWS Notes: ../docs/getting_started_aws.rst

    * Connors Smartflow Training Nodes: ../docs/smartflow_training_fusion_models.md

    * Dags live in: https://gitlab.kitware.com/smart/watch-smartflow-dags


High level:

    * build container

    * bake model into it

    * push container into T&E gitlab

    * tweak dags to point to new container

    * 

Prerequisites:


# ~/code/watch/aws/install_aws_local_requirements.sh
* Do you have ``kubectl``

.. code:: bash

   export AWS_PROFILE=iarpa
   kubectl -n airflow port-forward service/airflow-webserver 8080:8080 



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



.. code:: text

   AWS S3 URL

   https://s3.console.aws.amazon.com/s3/buckets/smartflow-023300502152-us-west-2?prefix=smartflow%2Fenv%2Fkitware-prod-v2%2Fwork%2F&region=us-west-2#

   # Check known "good" bucket for permission
   aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/

   # Check this bucket for permission
   aws s3 --profile iarpa ls s3://smartflow-023300502152-us-west-2

   export AWS_PROFILE=iarpa
   aws sts --profile iarpa assume-role \
       --role-arn arn:aws:iam::023300502152:role/smartflow-kitware-prod-v2-us-west-2-eks-admin \
       --role-session-name JonCrallSession
