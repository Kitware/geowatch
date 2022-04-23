#!/bin/bash
__doc__='

Depending on the server you want to upload your container to you will need to login to it.

load_secrets
docker login gitlab.kitware.com:4567 --username "$GITLAB_KITWARE_USERNAME" --password "$GITLAB_KITWARE_TOKEN"

load_secrets
docker login registry.smartgitlab.com --username "$CRALL_SMART_GITLAB_USERNAME" --password "$CRALL_SMART_GITLAB_TOKEN"


Furthermore, after you push your container to the registry, you need to provide
argo with the credentials to pull it down. 

load_secrets
KUBE_SECRET_NAME=crall-smartgitlab-secret 
DOCKER_SERVER=registry.smartgitlab.com
DOCKER_EMAIL=jon.crall@kitware.com
DOCKER_USER=$CRALL_SMART_GITLAB_USERNAME
DOCKER_PASSWORD=$CRALL_SMART_GITLAB_TOKEN
kubectl create secret docker-registry \
    "$KUBE_SECRET_NAME" \
    "--docker-server=$DOCKER_SERVER" \
    "--docker-username=$DOCKER_USER" \
    "--docker-password=$DOCKER_PASSWORD" \
    "--docker-email=$DOCKER_EMAIL"


This secret will then need to be registered in your workflow.yaml e.g.

spec:
  imagePullSecrets:
    - name: crall-smartgitlab-secret

'

cd "$HOME/code/watch/"

# Ensure credentials are flushed
transcrypt -f

docker build -t "smart/watch/ta2_training_v2" -f ./dockerfiles/ta2_training_v2.Dockerfile --build-arg BUILD_STRICT=1 .


# Push to kitware registery
#docker tag smart/watch/ta2_training_v2 gitlab.kitware.com:4567/smart/watch/ta2_training_v2
#docker push gitlab.kitware.com:4567/smart/watch/ta2_training_v2

# Push to IARPA registry
docker tag smart/watch/ta2_training_v2 registry.smartgitlab.com/kitware/watch/ta2_training_v2
docker push registry.smartgitlab.com/kitware/watch/ta2_training_v2

__test__="
docker run -it smart/watch/ta2_training_v2 bash
"
