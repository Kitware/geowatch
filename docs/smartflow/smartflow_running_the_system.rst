Prerequisites
-------------

* `AWS <../../docs/environment/getting_started_aws.rst>`

* `Kubectl <../../docs/environment/getting_started_kubectl.rst>`

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



Building the Pyenv-GEOWATCH Docker Image
-------------------------------------

If you have not built a docker image, we will need to do so.

There are two images that need to be build, first the
`pyenv dockerfile <../../dockerfiles/pyenv.Dockerfile>`_.
And then the watch dockerfile that builds on top of it
`watch dockerfile <../../dockerfiles/watch.Dockerfile>`_. The heredocs in these
files provide futher instructions.

Here we will go over the basic use case for a specific version of Python /
dependencies. We will use Python 3.11 and strict dependencies. We will assume
that your watch repo is in ``~/code/watch``, but if it is not then change the
environment variable:

.. code:: bash

    export WATCH_REPO_DPATH=$HOME/code/watch
    export STAGING_DPATH=$HOME/temp/container-staging
    # The name/tag of the image we will create
    export PYENV_IMAGE=pyenv:3.11.2
    # The Python version we want
    export PYTHON_VERSION=3.11.2

    # Create a directory on the host for context
    mkdir -p $STAGING_DPATH

    # For pyenv we will use an empty directory
    mkdir -p $STAGING_DPATH/empty

    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t $PYENV_IMAGE \
        --build-arg PYTHON_VERSION=$PYTHON_VERSION \
        -f $WATCH_REPO_DPATH/dockerfiles/pyenv.Dockerfile \
        $STAGING_DPATH/empty

    # Optional: push this image to kitware and smartgitlab registries

    # optional: Push to smartgitlab
    docker tag $PYENV_IMAGE registry.smartgitlab.com/kitware/$PYENV_IMAGE
    docker push registry.smartgitlab.com/kitware/$PYENV_IMAGE

    # optional: Push to gitlab.kitware.com
    docker tag $PYENV_IMAGE gitlab.kitware.com:4567/smart/watch/$PYENV_IMAGE
    docker push gitlab.kitware.com:4567/smart/watch/$PYENV_IMAGE


Now that the pyenv image ``pyenv:3.11.2`` has been created we can quickly test it:

.. code:: bash

    export PYENV_IMAGE=pyenv:3.11.2

    # Hello world should always run
    docker run --runtime=nvidia -it $PYENV_IMAGE echo "hello world"

    # Ensure the right python is exposed by default
    docker run --runtime=nvidia -it $PYENV_IMAGE python --version

    # if you have a GPU you can run
    docker run --runtime=nvidia -it $PYENV_IMAGE nvidia-smi


Now we build the watch image on top of the pyenv image. To ensure we do this
cleanly we will make a fresh clone of your local repo which will ensure you
dont accidently bake in any secrets or other large files.

.. code:: bash

    export WATCH_REPO_DPATH=$HOME/code/watch
    export STAGING_DPATH=$HOME/temp/container-staging
    export PYENV_IMAGE=pyenv:3.11.2
    export WATCH_VERSION=$(python -c "import watch; print(watch.__version__)")
    export BUILD_STRICT=1

    # A descriptive name for our watch image
    PYENV_TAG_SUFFIX=$(python -c "print('$PYENV_IMAGE'.replace(':', ''))")
    if [[ "$BUILD_STRICT" == "1" ]]; then
        export WATCH_IMAGE=watch:$WATCH_VERSION-strict-$PYENV_TAG_SUFFIX
    else
        export WATCH_IMAGE=watch:$WATCH_VERSION-loose-$PYENV_TAG_SUFFIX
    fi
    echo "
    ===========
    WATCH_REPO_DPATH = $WATCH_REPO_DPATH
    STAGING_DPATH    = $STAGING_DPATH
    WATCH_VERSION    = $WATCH_VERSION
    PYENV_IMAGE      = $PYENV_IMAGE
    BUILD_STRICT     = $BUILD_STRICT
    -----------
    WATCH_IMAGE=$WATCH_IMAGE
    ===========
    "

    # Create a directory on the host for context
    mkdir -p $STAGING_DPATH
    # For watch we make a fresh clone of our local repo
    [ -d $STAGING_DPATH/watch ] && rm -rf $STAGING_DPATH/watch
    git clone --origin=host-$HOSTNAME $WATCH_REPO_DPATH/.git $STAGING_DPATH/watch

    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t "$WATCH_IMAGE" \
        --build-arg "BUILD_STRICT=$BUILD_STRICT" \
        --build-arg "BASE_IMAGE=$PYENV_IMAGE" \
        -f $STAGING_DPATH/watch/dockerfiles/watch.Dockerfile .

    # Optional: push this image to kitware and smartgitlab registries

    # optional: Push to smartgitlab
    docker tag $WATCH_IMAGE registry.smartgitlab.com/kitware/$WATCH_IMAGE
    docker push registry.smartgitlab.com/kitware/$WATCH_IMAGE

    # optional: Push to gitlab.kitware.com
    docker tag $WATCH_IMAGE gitlab.kitware.com:4567/smart/watch/$WATCH_IMAGE
    docker push gitlab.kitware.com:4567/smart/watch/$WATCH_IMAGE


It is a good idea to run some tests to ensure the image built properly

.. code:: bash

    # Hello world should always run
    docker run --runtime=nvidia -it $WATCH_IMAGE echo "hello world"

    # Ensure the right python is exposed by default
    docker run --runtime=nvidia -it $WATCH_IMAGE python --version

    # Ensure the watch module is exposed by default
    docker run --runtime=nvidia -it $WATCH_IMAGE geowatch --version

    # if you have a GPU you can run
    docker run --runtime=nvidia -it $WATCH_IMAGE nvidia-smi

    # run the full test suite
    docker run --runtime=nvidia -it $WATCH_IMAGE ./run_tests.py


You may wish to upload this base image to the smartgitlab registry, but we will
need to bake in models, so this step is optional, but useful if you want to
build the base image on one machine and then bake in models on a different
machine.

.. code:: bash

    # Push the container to smartgitlab
    docker tag $WATCH_IMAGE registry.smartgitlab.com/kitware/$WATCH_IMAGE

    docker push registry.smartgitlab.com/kitware/$WATCH_IMAGE


**How to make a quick image update**

See: Update the code / models in an existing image


How to Bake a Model into a Pyenv Dockerfile
-------------------------------------------

Assuming that you have already build a pyenv docker image we will add a model
to it.

.. code:: bash

   # Set this to the name of the pyenv watch image that you built
   IMAGE_NAME=watch:0.4.5-strict-pyenv3.11.2

   NEW_IMAGE_NAME=${IMAGE_NAME}-models-2023-03-28
   echo $NEW_IMAGE_NAME

   # These are more models than we really need, but it will let use resuse this image for more experiments
   MODELS_OF_INTEREST="
   models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
   models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step0.pt
   models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
   models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
   models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt \
   models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt \
   models/uky/uky_invariants_2022_12_17/TA1_pretext_model/pretext_package.pt \
   models/landcover/sentinel2.pt
   "

   DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

   # Ensure the models of interest are pulled locally on your machine
   (cd $DVC_EXPT_DPATH && dvc pull -r aws $MODELS_OF_INTEREST)

   # We are also going to bake the metrics and data DVC into the repo too for
   # completeness
   DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
   METRICS_REPO_DPATH=$(python -c "import iarpa_smart_metrics, pathlib; print(pathlib.Path(iarpa_smart_metrics.__file__).parent.parent)")

   # Run the base image as a container so we can put stuff into it
   # We will use DVC to facilitate the transfer to keep things consistent
   # We mount our local experiment directory, and pull relevant files
   docker run \
       --volume $DVC_EXPT_DPATH:/host-smart_expt_dvc:ro \
       --volume $DVC_DATA_DPATH:/host-smart_data_dvc:ro \
       --volume $METRICS_REPO_DPATH:/host-metrics_repo:ro \
       -td --name temp_container $IMAGE_NAME

   docker exec -t temp_container pip install dvc
   docker exec -t temp_container mkdir -p /root/data
   docker exec -t temp_container git clone /host-smart_expt_dvc/.git /root/data/smart_expt_dvc
   docker exec -t temp_container git clone /host-smart_data_dvc/.git /root/data/smart_data_dvc
   docker exec -t temp_container git clone /host-metrics_repo/.git /root/code/metrics-and-test-framework

   docker exec -w /root/data/smart_expt_dvc -t temp_container \
       dvc remote add host /host-smart_expt_dvc/.dvc/cache

   # Workaround DVC Issue by removing aws remote
   # References: https://github.com/iterative/dvc/issues/9264
   docker exec -w /root/data/smart_expt_dvc -t temp_container \
       dvc remote remove aws

   # Pull in relevant models you want to bake into the container
   # These will be specified relative to the experiment DVC repo
   docker exec -w /root/data/smart_expt_dvc -t temp_container \
       dvc pull --remote host $MODELS_OF_INTEREST


   # Save the modified container as a new image
   docker commit temp_container $NEW_IMAGE_NAME

   # Cleanup the temp container
   docker stop temp_container
   docker rm temp_container

   # Push the container to smartgitlab
   docker tag $NEW_IMAGE_NAME registry.smartgitlab.com/kitware/$NEW_IMAGE_NAME
   docker push registry.smartgitlab.com/kitware/$NEW_IMAGE_NAME
   echo $NEW_IMAGE_NAME

   # optional: Push to gitlab.kitware.com
   docker tag $WATCH_IMAGE gitlab.kitware.com:4567/smart/watch/$WATCH_IMAGE
   docker push gitlab.kitware.com:4567/smart/watch/$WATCH_IMAGE


Update the code / models in an existing image
---------------------------------------------

Say you need to make a small change to the code, but don't want to rebuild the
entire model. We can handle that case by mounting the latest repos onto the
container, setting the remotes of the repo to point to those, pulling the
latest code, and commiting the change as a new image.

.. code:: bash


   export WATCH_REPO_DPATH=$HOME/code/watch
   export DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

   IMAGE_NAME=watch:0.4.5-strict-pyenv3.11.2-models-2023-03-28
   NEW_IMAGE_NAME=watch:0.4.5-strict-pyenv3.11.2-models-2023-03-28-v04

   # Mount the image with
   docker run \
       --volume $DVC_EXPT_DPATH:/host-smart_expt_dvc:ro \
       --volume $WATCH_REPO_DPATH:/host-watch_repo:ro \
       -td --name temp_container $IMAGE_NAME

   docker exec -w /root/code/watch  -t temp_container \
       git remote add host /host-watch_repo/.git

   docker exec -w /root/code/watch  -t temp_container \
       git pull host dev/0.4.5

   # Save the modified container as a new image
   docker commit temp_container $NEW_IMAGE_NAME

   docker stop temp_container
   docker rm temp_container

   # Push the container to smartgitlab
   echo $NEW_IMAGE_NAME
   docker tag $NEW_IMAGE_NAME registry.smartgitlab.com/kitware/$NEW_IMAGE_NAME
   docker push registry.smartgitlab.com/kitware/$NEW_IMAGE_NAME


How to Submit a DAG
-------------------

.. .. SeeAlso: ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV_V13.py
   ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_V13.py

Ensure that you have the DAG repo

.. code:: bash

    # This is the repo containing the smartflow dags
   git clone git@gitlab.kitware.com:smart/watch-smartflow-dags.git $HOME/code/watch-smartflow-dags


Choose a DAG file and modify it as necessary (TODO, describe this in more
detail).


Once you have a DAG file ready upload it to AWS via:

.. code:: bash

    # The path to our DAG repo
    LOCAL_DAG_DPATH=$HOME/code/watch-smartflow-dags

    # The name of the DAG file we edited
    DAG_FNAME=KIT_TA2_PREEVAL10_PYENV_V13.py

    # Upload the DAG file to AWS
    aws s3 --profile iarpa cp $LOCAL_DAG_DPATH/$DAG_FNAME \
        s3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v4/dags/$DAG_FNAME


If you have not done so ensure that we are forwarding the smartflow web service
to your machine:

.. code:: bash

    kubectl -n airflow port-forward service/airflow-webserver 2746:8080

Now, navigate to your airflow GUI in the browser at ``localhost:2746/home``,
which can be done via the command:

.. code:: bash

   # Not working?
   python -c "import webbrowser; webbrowser.open('https://localhost:2746/home', new=1)"


To debug interactively you can log into an existing run:


.. code:: bash

    kubectl -n airflow get pods
    # Find your POD_ADDR
    # POD_ADDR=site-cropped-kwcoco-6254ac27fab04f0b8eb302ac19b09745
    # kubectl -n airflow exec -it pods/$POD_ADDR -- bash

    # Script to list and exec into a running pod
    python -c "if True:
    import json
    import pandas as pd
    import rich
    import ubelt as ub
    info = ub.cmd('kubectl -n airflow get pods -o json')
    data = json.loads(info['out'])

    rows = []
    for item in data['items']:
        row = {
            'name': item['metadata']['name'],
            'status': item['status']['phase'],
            'startTime': item['status']['startTime'],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    rich.print(df.to_string())
    import rich.prompt
    ans = rich.prompt.Prompt.ask('which one?', choices=list(map(str, df.index.to_list())))
    idx = int(ans)
    pod_addr = df.iloc[idx]['name']
    ub.cmd(f'kubectl -n airflow exec -it pods/{pod_addr} -- bash', system=True)
    "


More notes:

.. code:: bash

    kubectl -n airflow logs pods/{pod_addr}



To interact with airflow on the command line, you need to exec into the airflow
scheduler pod.


.. code:: bash

    JQ_QUERY='.items[] | select(.metadata.name | startswith("airflow-scheduler-")) | .metadata.name'
    AIRFLOW_SCHEDULER_POD_NAME=$(kubectl -n airflow get pods -o json | jq -r "$JQ_QUERY")
    echo "AIRFLOW_SCHEDULER_POD_NAME=$AIRFLOW_SCHEDULER_POD_NAME"

    # Get a shell into the scheduler to run airflow commands
    kubectl -n airflow exec -it pods/$AIRFLOW_SCHEDULER_POD_NAME -- /bin/bash

    # Inside the airflow shell
    echo '

    airflow dags list

    airflow dags list -o json > dags.json

    airflow dags list-jobs

    # To run a dag you need to trigger and unpause it.
    airflow dags trigger kit_ta2_preeval10_pyenv_t29_batch_AE_R001
    airflow dags unpause kit_ta2_preeval10_pyenv_t29_batch_AE_R001

    airflow dags trigger kit_ta2_preeval10_pyenv_t29_batch_KW_R001
    airflow dags unpause kit_ta2_preeval10_pyenv_t29_batch_KW_R001

    REGION_IDS=("KR_R002" "KR_R001" "NZ_R001")
    for REGION_ID in "${REGION_IDS[@]}"; do
        echo "trigger $REGION_ID"
        airflow dags trigger kit_ta2_preeval10_pyenv_t29_batch_$REGION_ID
        airflow dags unpause kit_ta2_preeval10_pyenv_t29_batch_$REGION_ID
    done

    REGION_IDS=("KR_R002" "KR_R001" "NZ_R001" "KW_R001" "AE_R001")
    for REGION_ID in "${REGION_IDS[@]}"; do
        echo "trigger $REGION_ID"
        airflow dags trigger kit_ta2_preeval10_pyenv_t31_batch_$REGION_ID
        airflow dags unpause kit_ta2_preeval10_pyenv_t31_batch_$REGION_ID
    done


    # Status queries
    airflow dags list-jobs -d kit_ta2_preeval10_pyenv_t33_post1_batch_KR_R001 -o yaml
    airflow dags list-runs -d kit_ta2_preeval10_pyenv_t33_post1_batch_KR_R001 -o yaml
    '


    ### Alternative - execute commands from local shell
    # Oddly this tends to send outputs with color that we need to strip out.
    kubectl -n airflow exec -it pods/$AIRFLOW_SCHEDULER_POD_NAME -- airflow dags list -o json > dags.json
    cat dags.json | sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2};?)?)?[mGK]//g" | cat > dags_nocolor.json

    python -c "if True:
        import json
        import pathlib
        import cmd_queue

        # Build pattern to identify the jobs you want to run
        import xdev
        pattern = xdev.MultiPattern.coerce([
            f'kit_ta2_preeval10_pyenv_t{t}*'
            for t in [31, 35]
        ])
        data = json.loads(pathlib.Path('dags_nocolor.json').read_text())

        # Build cmd-queue with the commands to execute
        queue = cmd_queue.Queue.create(backend='serial')
        prefix = 'kubectl -n airflow exec -it pods/$AIRFLOW_SCHEDULER_POD_NAME -- '
        for item in data:
            if pattern.match(item['dag_id']):
                print(item['dag_id'])
                queue.submit(prefix + 'airflow dags trigger ' + item['dag_id'])
                queue.submit(prefix + 'airflow dags unpause ' + item['dag_id'])

        # It is a good idea to comment out the run to check that you
        # are doing what you want to do before you actually execute.
        queue.print_commands()
        queue.run()
    "



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


Running Dags After Containers are Using (OLD)
---------------------------------------------

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
