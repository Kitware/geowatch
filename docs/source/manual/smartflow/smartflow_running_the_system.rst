===============================
Running The System In Smartflow
===============================

.. note ::

   This file is currently a working document and contains a lot of notes

This goes over how to run the GeoWATCH system in smartflow.

The outline of this document is:

* `Building the Docker Image <SectionBuildDocker_>`__,
    + `Building the Pyenv Base Image <BuildPyenv_>`__,
    + `Building the GeoWATCH Image 2 <BuildGeowatch_>`__.
    + `Baking Models into the Image <BakeModel_>`__.
    + `Updating Existing Images <UpdateImage_>`__.
* `Submit a DAG <SubmitDAG_>`__.
    + `Debugging a DAG <DebugDAGS_>`__.
* `Old Notes <OldNotes_>`__.

Prerequisites
=============

Be sure you have

* `Setup the AWS CLI <../environment/getting_started_aws.rst>`_

* `Setup the kubectl CLI <../environment/getting_started_kubectl.rst>`_

* `Setup smartflow <getting_started_smartflow.rst>`_


Section 0: NOTICE OF AUTOMATION
===============================

This entire process has been scripted and lives in the `watch-smartflow-dags repo <https://gitlab.kitware.com/smart/watch-smartflow-dags>`_ repo.

The
`prepare_system.sh <https://gitlab.kitware.com/smart/watch-smartflow-dags/-/blob/main/prepare_system.sh>`_
script is the main driver. TODO: We should document how to use that script here
instead of these manual instructions.


Other script of interest are:

* `run_smartflow_dags.py <https://gitlab.kitware.com/smart/watch-smartflow-dags/-/blob/main/run_smartflow_dags.py>`_ - This is used by prepare_system to trigger dags and can be used as a standalone script to trigger dags that have already been uploaded.

* `pull_results.py <https://gitlab.kitware.com/smart/watch-smartflow-dags/-/blob/main/pull_results.py>`_ - This pulls results down from smartflow DAG runs and can optionally sumarize metrics.


.. _SectionBuildDocker:

Section 1: The GeoWATCH Docker Image
====================================

In this section we will go over how to build the docker image used in a submission:


There are two images that need to be build, first the
`pyenv dockerfile <../../../../dockerfiles/pyenv.Dockerfile>`_.
And then the watch dockerfile that builds on top of it
`watch dockerfile <../../../../dockerfiles/watch.Dockerfile>`_.
The heredocs in these files provide futher instructions.
We will also need to add models to the image


.. _BuildPyenv:

Building the Pyenv Base Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


.. _BuildGeowatch:

Building the GeoWATCH Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


.. _BakeModel:

How to Bake a Model into a Pyenv Dockerfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

   DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

   # Ensure the models of interest are pulled locally on your machine
   (cd $DVC_EXPT_DPATH && dvc pull -r aws $MODELS_OF_INTEREST)

   # We are also going to bake the metrics and data DVC into the repo too for
   # completeness
   DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
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


.. _UpdateImage:

Update An Existing Image
^^^^^^^^^^^^^^^^^^^^^^^^

Say you need to make a small change to the code, but don't want to rebuild the
entire model. We can handle that case by mounting the latest repos onto the
container, setting the remotes of the repo to point to those, pulling the
latest code, and commiting the change as a new image.

.. code:: bash


   export WATCH_REPO_DPATH=$HOME/code/watch
   export DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

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


.. _SubmitDAG:

How to Submit a DAG
===================

.. .. SeeAlso: ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV_V13.py
   ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_V13.py


We maintain the airflow DAGS in the `watch-smartflow-dags repo <https://gitlab.kitware.com/smart/watch-smartflow-dags>`_.
Ensure that you have the DAG repo:

.. code:: bash

    # This is the repo containing the smartflow dags
   git clone git@gitlab.kitware.com:smart/watch-smartflow-dags.git $HOME/code/watch-smartflow-dags


Choose a DAG file and modify it as necessary


.. note::

    TODO: Describe in more detail


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


Building / Modifying a DAG
==========================

Our smartflow DAGs are built as sequences of smartflow CLI commands that wrap
our local CLI commands. These smartflow CLI commands live in
`geowatch/cli/smartflow <../../../../geowatch/cli/smartflow>`_.

Each of these uses ffsspec to grab manifests of available assets from an s3
bucket, which then points to the data the task could use. It is the scripts job
to pull the data, perform the computation, print debugging info, and push
results and debug data back to a new output bucket.


See [ComputeInstanceTypes]_ for details on available instance types.

Also see

References:
    .. [ComputeInstanceTypes] https://smartgitlab.com/blacksky/smartflow/-/blob/main/docs/Framework/Smartflow-Framework.md#selecting-compute-resources-for-tasks


A table as of 2024-05-26 of compute resources is::

    | Purpose             | RAM/CPU ratio | GPUs     | Accelerator       | SSD*         | SSD Size                 | Price Range ($/hr) |
    |--------------------:|--------------:|---------:|------------------:|-------------:|-------------------------:|-------------------:|
    | general             | 4g/core       |          |                   | No           |                          | 0.10-9.677         |
    | gpu-aws-inferentia1 | 2G/core       | 1, 4, 16 | inferentia1       | No           |                          | 0.228-4.721        |
    | gpu-aws-inferentia2 | 2G/core       | 1, 6, 12 | inferentia2       | No           |                          | 1.967-12.99        |
    | gpu-aws-trainium1   | 4G/core       | 1, 16    | trainium1         | Yes          | 1 x 475 GB, 4 x 1900 GB  | 1.34-21.50         |
    | gpu-nvidia-a10g     | 4G/core       | 1, 4, 8  | nvidia-a10g       | Yes          | 1 x 250 GB - 2 x 3800 GB | 1.006-16.288       |
    | gpu-nvidia-t4       | 4G/core       | 1, 4, 8  | nvidia-tesla-t4   | Yes          | 1 x 125 GB - 2 x 900 GB  | 0.526-7.824        |
    | gpu-nvidia-v100     | 7.625G/core   | 1, 4, 8  | nvidia-tesla-v100 | If Requested | 2 x 900 GB               | 3.06-31.212        |
    | high-cpu            | 2G/core       |          |                   | If Requested | 1 x 50 GB - 4 x 1900GB   | 0.085-6.4512       |
    | high-mem            | 4G/core       |          |                   | If Requested | 1 x 75 GB - 4 x 1900GB   | 0.1068-5.34        |



.. _RunningDAGS:

Running DAGS
^^^^^^^^^^^^

In the GUI you can simply search for your dag and hit the run buttom.

To programatically interact with airflow on the command line, you need to exec
into the airflow scheduler pod.


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
    airflow dags list-runs -d kit_eval_11_rerun_batch_AE_R001 -o yaml
    '


    ### Alternative - execute commands from local shell
    # Oddly this tends to send outputs with color that we need to strip out.
    JQ_QUERY='.items[] | select(.metadata.name | startswith("airflow-scheduler-")) | .metadata.name'
    AIRFLOW_SCHEDULER_POD_NAME=$(kubectl -n airflow get pods -o json | jq -r "$JQ_QUERY")
    export AIRFLOW_SCHEDULER_POD_NAME
    kubectl -n airflow exec -it pods/$AIRFLOW_SCHEDULER_POD_NAME -- airflow dags list -o json > dags.json
    cat dags.json | sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2};?)?)?[mGK]//g" | cat > dags_nocolor.json

    airflow dag_state kit_eval_11_rerun_batch_AE_C002

    # Note:
    # This idea will be further developed in
    ~/code/watch-smartflow-dags/monitor_dags.py

    python -c "if True:
        import json
        import pathlib
        import cmd_queue

        # Build pattern to identify the jobs you want to run
        import xdev
        pattern = xdev.MultiPattern.coerce([
            'kit_eval_11_rerun_batch*'
            #f'kit_ta2_preeval10_pyenv_t{t}*'
            #for t in [31, 35]
        ])
        # FIXME: the json can be output with an error, need to strip it.
        text = pathlib.Path('dags_nocolor.json').read_text()
        data = json.loads(text[86:])


        valid_rows = []
        for item in data:
            if pattern.match(item['dag_id']):
                valid_rows.append(item)


        if 0:
            # Query the status of the selected dags
            import os
            AIRFLOW_SCHEDULER_POD_NAME = os.environ['AIRFLOW_SCHEDULER_POD_NAME']
            prefix = f'kubectl -n airflow exec -it pods/{AIRFLOW_SCHEDULER_POD_NAME} -- '

            import base64
            # easy-to-represent char encoding of the strip ansi pattern
            pat = base64.b32decode(b'DNOFWKC3GAWTSXL3GEWDG7JIHNNTALJZLV5TCLBSPU5T6KJ7FE7VW3KHJNOQ====').decode('utf8')
            import re
            pat = re.compile(pat)
            from watch.utils.util_yaml import Yaml
            row_to_states = {}
            for row in valid_rows:
                dag_id = row['dag_id']
                info = ub.cmd(prefix + f'airflow dags list-runs -d {dag_id} -o yaml', shell=True)
                text = pat.sub('', info['out'])
                states = Yaml.loads(text)
                print(ub.urepr(states))
                row_to_states[dag_id] = states

            orig_row = {r['dag_id']: r for r in valid_rows}
            dag_info_rows = []
            for dag_id, states in row_to_states.items():
                row = orig_row[dag_id]
                if len(states) == 0:
                    row['status'] = None
                else:
                    mrs = states[-1]
                    row['status'] = mrs['state']
                    row['execution_date'] = mrs['execution_date']
                    row['run_id'] = mrs['run_id']
                    row['start_date'] = mrs['start_date']
                    row['end_date'] = mrs['end_date']
                dag_info_rows.append(row)

            import pandas as pd
            df = pd.DataFrame(dag_info_rows)
            import rich
            rich.print(df)

            num_need_run = pd.isna(df['status']).sum()
            num_running = (df['status'] == 'running').sum()
            print(f'num_need_run={num_need_run}')
            print(f'num_running={num_running}')

        import pandas as pd
        df = pd.DataFrame(valid_rows)
        import rich
        rich.print(df)

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



.. _DebugDAGS:

Debuggging DAGS
^^^^^^^^^^^^^^^

Here is a useful command to get a list of running pods that contain jobs.

.. code:: bash

    kubectl -n airflow get pods


Given a pod id there are useful commands

.. code:: bash

    # Pod logs
    kubectl -n airflow logs pods/{pod_addr}

    # Exec into a pod
    kubectl -n airflow exec -it pods/{pod_addr} -- bash


Here is a snippet to automatically list pods and allow you to select one to
exec info:

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

    from dateutil.parser import isoparse
    from datetime import datetime as datetime_cls
    utc_now = datetime_cls.utcnow()

    rows = []
    for item in data['items']:
        restart_count = sum([cs['restartCount'] for cs in item['status']['containerStatuses']])
        start_time = item['status']['startTime']
        start_dt = isoparse(start_time)
        utc_now = utc_now.replace(tzinfo=start_dt.tzinfo)
        age_delta = utc_now - start_dt
        row = {
            'name': item['metadata']['name'],
            'status': item['status']['phase'],
            'startTime': start_time,
            'restarts': restart_count,
            'age': str(age_delta),
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


.. _OldNotes:

Old Notes
=========

How to Bake a Model into a Dockerfile (OLD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we edit a DAG file for airflow


.. git clone git@gitlab.kitware.com:smart/watch-smartflow-dags.git


Choose a DAG file in ~/code/watch-smartflow-dags/ then edit it to give it a unique name
.e.g. ``~/code/watch-smartflow-dags/KIT_TA2_20221121_BATCH.py``


* change name of file and then change ``EVALUATION`` to be a unique string to name it what you want.

* change the image names / tags e.g. ``image="registry.smartgitlab.com/kitware/watch/ta2:Ph2Nov21EvalBatch"``, these are all "pod tasks" create_pod_task

* ``purpose`` is something about the node that it runs on.
  For a subset of valid options see: https://smartgitlab.com/blacksky/smartflow/-/blob/118140a81362c5721b5e9bb65ab967fb8bd28163/CHANGELOG.md

* make cpu limit a bit less than what is availble on the pod.

* Copy the DAG to smartflow S3:

  .. code:: bash

      aws s3 --profile iarpa cp Kit_DatasetGeneration.py s3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v2/dags/Kit_DatasetGeneration.py


Need to run service to access airflow gui:

.. code:: bash

    kubectl -n airflow port-forward service/airflow-webserver 2746:8080

navigate to ``localhost:2746/home``

Now dags show up in the GUI.


SeeAlso
^^^^^^^

* `Connor's Smartflow Training Nodes <smartflow_training_fusion_models.md>`_

* Dags live in: https://gitlab.kitware.com/smart/watch-smartflow-dags
