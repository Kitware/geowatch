Notes on running on AWS
=======================



For scripts that have setup instructions in them see:

* install_aws_local_requirements.sh
* build_base_ta2_train_image.sh


Install kubectl CLI
.. code:: bash

   ~/code/watch/aws/install_aws_local_requirements.sh


Configure kubectl for EKS use automatically with AWS CLI:


.. code:: bash

    aws eks update-kubeconfig --region us-west-2 --name WATCH_PRODUCTION --profile iarpa
       

Add to  ~/.kube/config

- --role-arn
- arn:aws:iam::023300502152:role/WATCH_PRODUCTION_EKS_ACCESS

.. code:: bash

    kubectl get svc


Install Argo CLI
https://github.com/argoproj/argo-workflows/releases/tag/v3.2.6



.. code:: bash

    argo list

    cd ~/code/watch/aws

    echo "
    annotations:
      email: '$(git config --get user.email)'
      purpose: |
        Check read access on DVC GitLab repo
      submitter: '$(git config --get user.name)'
    " > user_id.yaml


    argo submit hello-workflow.yaml --watch

    # Note that this will generate a unique job name
    cd $HOME/code/watch/aws
    argo submit dvc_check_workflow.yml --watch

    # List all workflows
    argo list

    # This is not 100% reliable has race conditions
    WORKFLOW_NAME=$(argo list --running | head -n 2 | tail -n 1 | cut -d' ' -f1)

    argo logs "${WORKFLOW_NAME}" --follow


.. code:: bash
   

   # To play with the training docker container

   docker run -it gitlab.kitware.com:4567/smart/watch/ta2_training_v2:981feb6592f2 bash

   apt update
   apt install ssh tmux tree curl iputils-ping -y

   See ~/code/watch/aws/train_remote_entrypoint.sh

   git remote --verbose

   argo list --running
