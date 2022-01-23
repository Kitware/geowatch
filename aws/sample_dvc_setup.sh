#!/bin/bash
# Intended as a demonstration for setting up DVC and checking out some
# data from within a container running in the Argo framework

git clone "https://${DVC_GITLAB_USERNAME}:${DVC_GITLAB_PASSWORD}@gitlab.kitware.com/smart/smart_watch_dvc.git"

source /opt/conda/etc/profile.d/conda.sh
set -x
cd smart_watch_dvc

conda activate watch
dvc remote add aws-noprofile s3://kitware-smart-watch-data/dvc

dvc pull Drop1-Aligned-TA1-2022-01/data.kwcoco.json.dvc -r aws-noprofile --quiet
dvc checkout Drop1-Aligned-TA1-2022-01/data.kwcoco.json.dvc
