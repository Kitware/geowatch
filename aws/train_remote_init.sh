#!/bin/bash
# Initial steps of the training. This file is effectively locked in.  It is
# called on the branch as-is from when the docker container was created.  It
# calls the train_remote_entrypoint.sh script which can be "hot-reloaded" in
# effect because this script will update the repo before calling it.
__doc__='


Manual Local Debug:

    cd ~/code/watch
    transcrypt --cipher=aes-256-cbc --password "$WATCH_TRANSCRYPT_SECRET"
    source ~/code/watch/secrets/secrets
    export BRANCH_NAME=dev/flow49

    docker run --gpus all \
        -v $HOME/.aws:/root/.aws \
        --env AWS_PROFILE=iarpa \
        --env DVC_GITLAB_USERNAME \
        --env DVC_GITLAB_PASSWORD \
        --env BRANCH_NAME \
        --env WATCH_REPO_GITLAB_RO_DEPLOY_PASSWORD \
        --env WATCH_REPO_GITLAB_RO_DEPLOY_USERNAME \
        -it kitware/watch/ta2_training_v2 bash


Remote Submission:
    cd "$HOME/code/watch/aws"

    WORKFLOW_FPATH=$HOME/code/watch/aws/ta2_train_workflow.yml
    NAME_PREFIX=$(yq -r .metadata.generateName "$WORKFLOW_FPATH")
    argo submit "$WORKFLOW_FPATH" --watch
    WORKFLOW_NAME=$(argo list --running | argo list --running | grep "$NAME_PREFIX" | head -n 1 | cut -d" " -f1)
    argo logs "${WORKFLOW_NAME}" --follow


'


# First ensure we can grab the latest and greatest version of the current checkout
mkdir -p "$HOME/.ssh"
echo '|1|tln6/2oSoZ71GXymBD/DR6qjguM=|/DOcGHEnk4HujFZsiyAbN15hlp0= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBIiKR90e4+4i2gkAW81AiD0Sg/eycexpA+suyTl0e/9DxM4qVNgufZ5p98mRmk3Dz748O3JBNL60kvFKNXN7ZYg=' >> "$HOME/.ssh/known_hosts"
echo '|1|VnWfnYg/bKW2l/z9z8/3VZoTMMM=|iSpmXNap8X55Nc4WTWSD+/HjMus= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBIiKR90e4+4i2gkAW81AiD0Sg/eycexpA+suyTl0e/9DxM4qVNgufZ5p98mRmk3Dz748O3JBNL60kvFKNXN7ZYg=' >> "$HOME/.ssh/known_hosts"


export SMART_DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
export WATCH_REPO_DPATH=$HOME/code/watch
source /opt/conda/etc/profile.d/conda.sh

set -x

# Update the watch code repo
cd "$WATCH_REPO_DPATH"

# Add remote with our secret credentials and then pull from it
# Note: these secrets are populated in the container environment by the
# workflow file.  In other words, we registered these in the workflow and argo
# knows how to tell the container how to use them. If running locally you need
# to inject these into your environment.
git remote add custom "https://${WATCH_REPO_GITLAB_RO_DEPLOY_USERNAME}:${WATCH_REPO_GITLAB_RO_DEPLOY_PASSWORD}@gitlab.kitware.com/smart/watch.git"
git fetch custom

BRANCH_NAME=$1
if [[ "$BRANCH_NAME" == "" ]]; then 
    BRANCH_NAME=$(git branch --show)
fi

echo "BRANCH_NAME = $BRANCH_NAME"
git checkout "$BRANCH_NAME"
git pull custom "$BRANCH_NAME"

# Execute the latest and greatest entrypoint
source "$WATCH_REPO_DPATH/aws/train_remote_entrypoint.sh"
