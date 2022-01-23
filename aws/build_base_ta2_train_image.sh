#!/bin/bash
__doc__='
load_secrets
docker login gitlab.kitware.com:4567 --username $AD_USERNAME --password $AD_PASSWORD
docker pull gitlab.kitware.com:4567/computer-vision/ci-docker/miniconda3
'
#docker login gitlab.kitware.com:4567

cd "$HOME/code/watch/"
docker build -t "ci-docker/miniconda3" -f ./dockerfiles/ta2_training_v2.Dockerfile .

docker build -t "smartwatch/ta2-train-v2" --build-arg BUILD_STRICT=1 .
