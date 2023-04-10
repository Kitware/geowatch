#!/bin/bash
set -e

IMAGE=smart-watch-$(whoami)
echo Starting $IMAGE container from $(pwd) ...
docker build -f ./dev/dzyne/Dockerfile -t $IMAGE . 

docker run --rm -it \
 --user $(id -u):$(id -g) \
 --gpus device="0" \
 --shm-size=1024m \
 -w /watch \
 -v $(pwd):/watch \
 -v /media/bigdata/projects/smart:/bigdata_smart \
 -v /media/bigdata/dlau/smart/output:/smart_data_dvc \
 --name $(whoami)_watch \
 $IMAGE $@
