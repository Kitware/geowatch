#!/bin/bash
WATCH_REPO_DPATH=$HOME/code/watch
source "$WATCH_REPO_DPATH/secrets/secrets"


# NOPE Create an access token
# https://gitlab.kitware.com/smart/watch/-/settings/access_tokens

# Create an DEPLOY token
# https://gitlab.kitware.com/smart/watch/-/settings/repository


echo "
apiVersion: v1
kind: Secret
metadata:
  name: watch-repo-gitlab-ro-deploy-token
type: kubernetes.io/basic-auth
stringData:
  username: $WATCH_REPO_GITLAB_RO_DEPLOY_USERNAME
  password: $WATCH_REPO_GITLAB_RO_DEPLOY_PASSWORD
" > "$WATCH_REPO_DPATH/secrets/kube-secret-watch-repo-gitlab-ro-deploy-token.yaml"

cat "$WATCH_REPO_DPATH/secrets/kube-secret-watch-repo-gitlab-ro-deploy-token.yaml"

kubectl apply -f "$WATCH_REPO_DPATH/secrets/kube-secret-watch-repo-gitlab-ro-deploy-token.yaml"

kubectl get secret dvc-gitlab-access-token --template='{{ range $key, $value := .data }}{{ printf "%s: %s\n" $key ($value | base64decode) }}{{ end }}'
kubectl get secret watch-repo-gitlab-ro-access-token --template='{{ range $key, $value := .data }}{{ printf "%s: %s\n" $key ($value | base64decode) }}{{ end }}'
kubectl get secret watch-repo-gitlab-ro-deploy-token --template='{{ range $key, $value := .data }}{{ printf "%s: %s\n" $key ($value | base64decode) }}{{ end }}'
