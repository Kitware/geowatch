#!/bin/bash
# Initial steps of the training. This file is effectively locked in.  It is
# called on the branch as-is from when the docker container was created.  It
# calls the train_remote_entrypoint.sh script which can be "hot-reloaded" in
# effect because this script will update the repo before calling it.


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
git remote add custom "https://${DVC_GITLAB_USERNAME}:${DVC_GITLAB_PASSWORD}@gitlab.kitware.com/smart/watch.git"
#git fetch custom
git pull

# Execute the latest and greatest entrypoint
source "$WATCH_REPO_DPATH/aws/train_remote_entrypoint.sh"
