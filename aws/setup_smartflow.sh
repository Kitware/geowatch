#!/bin/bash
# We are moving to smartflow, this document aims to describe how to use it in
# our use-case. More general docs are:
# https://smartgitlab.com/blacksky/smartflow/-/blob/main/docs/Administration/Deployment.md#installation-requirements

pip install awscli

# Apparently this is not needed? Because someone else did it?
#cd "$HOME"/code 
#git clone git@smartgitlab.com:blacksky/smartflow.git
#export AWS_PROFILE="iarpa"
#cd smartflow
#cd "$HOME"/code/smartflow 
#pip install -r requirements-scripts.txt
#source "$HOME"/code/watch/secrets/secrets
#ENVIRONMENT_NAME=kit-train-env
#DOCKER_USERNAME="SMARTGITLAB_WATCH_REGISTRY_RO_TOKEN"
#DOCKER_PASSWORD="$SMARTGITLAB_WATCH_REGISTRY_RO_TOKEN"
#AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
#python scripts/env_setup.py \
#     --aws_account_id "$AWS_ACCOUNT_ID" \
#     --environment_name "$ENVIRONMENT_NAME" \
#     --vpc_id "vpc-0123456789abcdef0" \
#     --private_subnet_ids "subnet-11111111111111111,subnet-22222222222222222,subnet-33333333333333333" \
#     --public_subnet_ids "subnet-44444444444444444,subnet-55555555555555555,subnet-66666666666666666" \
#     --docker_registry "registry.smartgitlab.com" \
#     --docker_username "$DOCKER_USERNAME" \
#     --docker_password "$DOCKER_PASSWORD" \
#     --ec2_key_pair "my-ec2-keypair" \



#You do not need to do any of that installation setup stuff
#(Apart from verification)
# https://smartgitlab.com/blacksky/smartflow/-/blob/main/docs/Administration/Deployment.md#verification

#You just need to get the kubernetes client: https://kubernetes.io/docs/tasks/tools/#kubectl
#And configure it to access the smartflow instance Yoni has already set up

export AWS_PROFILE="iarpa"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
echo "AWS_ACCOUNT_ID = $AWS_ACCOUNT_ID"
ENVIRONMENT_NAME="kitware-prod"
AWS_REGION="us-west-2"

# I needed to be granted permission by the admin
aws eks update-kubeconfig \
    --name "smartflow-${ENVIRONMENT_NAME}-eks" \
    --role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/smartflow-${ENVIRONMENT_NAME}-${AWS_REGION}-eks-admin"


# https://blacksky.smartgitlab.com/smartflow/markdown/Framework/Getting-Started.html#authoring-your-first-dag
# See: https://gitlab.kitware.com/smart/watch-smartflow-dags
