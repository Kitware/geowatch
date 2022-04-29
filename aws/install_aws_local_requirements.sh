#!/bin/bash

#Install kubectl CLI

mkdir -p "$HOME/tmp/kub"
cd "$HOME/tmp/kub"

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
echo "$(<kubectl.sha256)  kubectl" | sha256sum --check

PREFIX=$HOME/.local
chmod +x kubectl
mkdir -p "$PREFIX"/bin
cp ./kubectl "$PREFIX"/bin/kubectl
# ensure $PREFIX/bin is in the PATH

kubectl version --client

#Configure kubectl for EKS use automatically with AWS CLI:
aws eks update-kubeconfig --region us-west-2 --name WATCH_PRODUCTION --profile iarpa



# Manual
## Add to  ~/.kube/config
# This needs to go in users.user.exec.args
echo "
- --role-arn
- arn:aws:iam::023300502152:role/WATCH_PRODUCTION_EKS_ACCESS
" >> "$HOME/.kube/config"



# Test
kubectl get svc




#Install argo CLI
mkdir -p "$HOME/tmp/argo"
cd "$HOME/tmp/argo"

# Download the binary
curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.2.6/argo-linux-amd64.gz

# Unzip
gunzip argo-linux-amd64.gz

# Make binary executable
chmod +x argo-linux-amd64

# Move binary to path
PREFIX=$HOME/.local
cp ./argo-linux-amd64 "$PREFIX/bin/argo"

# Test installation
argo version
