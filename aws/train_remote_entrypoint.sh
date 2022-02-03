#!/bin/bash
# Assume that all repos and permissions are setup.
# This is called by the train init script.

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
source "$WATCH_REPO_DPATH/watch/tasks/fusion/experiments/crall/toy_experiments_msi.sh"
