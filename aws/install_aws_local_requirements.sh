#!/bin/bash

#Install kubectl CLI

mkdir -p "$HOME/tmp/kub"
cd "$HOME/tmp/kub"

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
echo "$(<kubectl.sha256)  kubectl" | sha256sum --check

chmod +x kubectl
mkdir -p ~/.local/bin
cp ./kubectl ~/.local/bin/kubectl
# and then add ~/.local/bin/kubectl to $PATH

kubectl version --client


#Configure kubectl for EKS use automatically with AWS CLI:
aws eks update-kubeconfig --region us-west-2 --name WATCH_PRODUCTION --profile iarpa



# Manual
## Add to  ~/.kube/config
echo "
- --role-arn
- arn:aws:iam::023300502152:role/WATCH_PRODUCTION_EKS_ACCESS
" >> "$HOME/.kube/config"



# Test
kubectl get svc




#Install kubectl CLI
mkdir -p "$HOME/tmp/argo"
cd "$HOME/tmp/argo"

# Download the binary
curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.2.6/argo-linux-amd64.gz

# Unzip
gunzip argo-linux-amd64.gz

# Make binary executable
chmod +x argo-linux-amd64

# Move binary to path
cp ./argo-linux-amd64 "$HOME/.local/bin/argo"

# Test installation
argo version
