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


.. code:: bash
   

   # To play with the training docker container

   docker run -it gitlab.kitware.com:4567/smart/watch/ta2_training_v2:981feb6592f2 bash

   apt update
   apt install ssh tmux tree curl iputils-ping -y

   See ~/code/watch/aws/train_remote_entrypoint.sh

   git remote --verbose



To list Secrets

.. code:: bash

   kubectl get secrets


Web UI

.. code:: bash

    # Start server
    kubectl -n airflow port-forward service/airflow-webserver 2746:8080

    # Login to server
    google-chrome https://127.0.0.1:2746/
