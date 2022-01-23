#!/bin/bash
# Assume that all repos and permissions are setup.
# This is called by the train init script.


# Initialize the base DVC repo with AWS permissions
mkdir -p "$HOME/data/dvc-repos"
git clone "https://${DVC_GITLAB_USERNAME}:${DVC_GITLAB_PASSWORD}@gitlab.kitware.com/smart/smart_watch_dvc.git" "$SMART_DVC_DPATH"
cd "$SMART_DVC_DPATH"
conda activate watch
dvc remote add aws-noprofile s3://kitware-smart-watch-data/dvc
dvc pull Drop1-Aligned-TA1-2022-01/data.kwcoco.json.dvc -r aws-noprofile --quiet
dvc checkout Drop1-Aligned-TA1-2022-01/data.kwcoco.json.dvc
