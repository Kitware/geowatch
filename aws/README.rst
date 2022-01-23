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

   docker run -it gitlab.kitware.com:4567/smart/watch/training:60d9b534 bash

   apt update
   apt install ssh tmux tree curl iputils-ping -y

   cd /watch
   git checkout master
   # ECDSA key fingerprint is SHA256:Lm8wR6C3yccDC25Og9xIpS+WtfKJdB1pVtAhdN82v0Q.
   cat $HOME/.ssh/known_hosts

   ping https://gitlab.kitware.com/smart/watch.git

   echo '|1|tln6/2oSoZ71GXymBD/DR6qjguM=|/DOcGHEnk4HujFZsiyAbN15hlp0= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBIiKR90e4+4i2gkAW81AiD0Sg/eycexpA+suyTl0e/9DxM4qVNgufZ5p98mRmk3Dz748O3JBNL60kvFKNXN7ZYg=' >> $HOME/.ssh/known_hosts
   echo '|1|VnWfnYg/bKW2l/z9z8/3VZoTMMM=|iSpmXNap8X55Nc4WTWSD+/HjMus= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBIiKR90e4+4i2gkAW81AiD0Sg/eycexpA+suyTl0e/9DxM4qVNgufZ5p98mRmk3Dz748O3JBNL60kvFKNXN7ZYg=' >> $HOME/.ssh/known_hosts
   git pull
   git remote --verbose

   argo list --running
