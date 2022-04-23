#!/bin/bash
__doc__='

load_secrets
source ~/internal/safe/secrets
docker login gitlab.kitware.com:4567 --username "$GITLAB_KITWARE_USERNAME" --password "$GITLAB_KITWARE_TOKEN"

source ~/internal/safe/secrets
docker login registry.smartgitlab.com --username "$CRALL_SMART_GITLAB_USERNAME" --password "$CRALL_SMART_GITLAB_TOKEN"
'

cd "$HOME/code/watch/"

# Ensure credentials are flushed
transcrypt -f

docker build -t "smart/watch/ta2_training_v2" -f ./dockerfiles/ta2_training_v2.Dockerfile --build-arg BUILD_STRICT=1 .

docker tag smart/watch/ta2_training_v2 gitlab.kitware.com:4567/smart/watch/ta2_training_v2
docker push gitlab.kitware.com:4567/smart/watch/ta2_training_v2

__test__="
docker run -it smart/watch/ta2_training_v2 bash
"
