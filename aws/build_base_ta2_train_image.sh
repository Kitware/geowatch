#!/bin/bash
__doc__='
load_secrets
docker login gitlab.kitware.com:4567 --username $GITLAB_KITWARE_USERNAME --password $GITLAB_KITWARE_TOKEN
'
#docker login gitlab.kitware.com:4567

cd "$HOME/code/watch/"
docker build -t "smart/watch/ta2_training_v2" -f ./dockerfiles/ta2_training_v2.Dockerfile --build-arg BUILD_STRICT=1 .

docker tag smart/watch/ta2_training_v2 gitlab.kitware.com:4567/smart/watch/ta2_training_v2
docker push gitlab.kitware.com:4567/smart/watch/ta2_training_v2

# f887e42a2206

__test__="
docker run -it f887e42a2206 bash
"
