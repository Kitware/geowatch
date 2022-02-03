#!/bin/bash
__doc__='
This script assumes that the repo and permissions are setup.

The entrypoint will update the repo to the latest version of whatever branch
the Docker image was created with. 

Note: you must edit "ta2_train_workflow.yml" to specify the branch should be
used.

To submit these jobs run something like:

    cd "$HOME/code/watch/aws"
    argo submit "$HOME/code/watch/aws/ta2_train_workflow.yml" --watch

    # NOTE: It usually takes ~12-15 minutes for a job to startup after being
    # submitted

    # And then view the logs in real time (note: if the workflow ends, you need
    # to use the UI to access the old logs)
    # This is not 100% reliable has race conditions
    WORKFLOW_NAME=$(argo list --running | head -n 2 | tail -n 1 | cut -d" " -f1)
    echo "WORKFLOW_NAME = $WORKFLOW_NAME"
    argo logs "${WORKFLOW_NAME}" --follow

'

export SMART_DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
export WATCH_REPO_DPATH=$HOME/code/watch
source /opt/conda/etc/profile.d/conda.sh
conda activate watch

# Initialize the base DVC repo with AWS permissions
mkdir -p "$HOME/data/dvc-repos"
git clone "https://${DVC_GITLAB_USERNAME}:${DVC_GITLAB_PASSWORD}@gitlab.kitware.com/smart/smart_watch_dvc.git" "$SMART_DVC_DPATH"

cd "$SMART_DVC_DPATH"
dvc remote add aws-noprofile s3://kitware-smart-watch-data/dvc

# Grab the required datasets that we need
dvc pull Drop2-Aligned-TA1-2022-01/data.kwcoco.json.dvc -r aws-noprofile --quiet

#dvc checkout Drop1-Aligned-TA1-2022-01/data.kwcoco.json.dvc


export DVC_DPATH=$SMART_DVC_DPATH
export WORKDIR="$DVC_DPATH/training/$HOSTNAME/$USER"


# All outputs will be saved in a "workdir"
# Startup background process that will write data to S3 in realish time
mkdir -p "$WORKDIR"
source "$WATCH_REPO_DPATH/aws/smartwatch_s3_sync.sh"
smartwatch_s3_sync_forever_in_tmux "$WORKDIR"


# The following script should run the toy experiments end-to-end
export NDSAMPLER_DISABLE_OPTIONAL_WARNINGS=1
source "$WATCH_REPO_DPATH/watch/tasks/fusion/experiments/crall/toy_experiments_msi.sh"
