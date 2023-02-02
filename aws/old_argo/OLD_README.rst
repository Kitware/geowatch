Install Argo CLI
https://github.com/argoproj/argo-workflows/releases/tag/v3.2.6


Web UI

.. code:: bash

    # Start server
    kubectl -n argo port-forward svc/argo-server 2746:2746

    # Generate auth token, will need to copy it into gui
    argo auth token 
    # Login to server
    google-chrome https://127.0.0.1:2746/


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



To list Secrets

.. code:: bash

   kubectl get secrets


Web UI

.. code:: bash

    # Start server
    kubectl -n argo port-forward svc/argo-server 2746:2746

    # Generate auth token, will need to copy it into gui
    argo auth token 
    # Login to server
    google-chrome https://127.0.0.1:2746/


 
Notes on argo resources:

https://argoproj.github.io/argo-workflows/resource-duration/#:~:text=If%20requests%20are%20not%20set,CPU%20and%20100Mi%20for%20memory.
