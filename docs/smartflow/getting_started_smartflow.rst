

Requirements
------------
We assume you have the aws and kubernetes command line tools installed.

* `get started with aws <../../docs/environment/getting_started_aws.rst>`

* `get started with kubectl <../../docs/environment/getting_started_kubectl.rst>`


Configure Kubernetes
--------------------

To configure kubernetes to talk to the Kitware smartflow server run:

NOTE: If you have an old configuration you should remove it. You can list
existing contexts with ``kubectl config get-contexts``, and
``kubectl config delete-context <chosen-context>`` to remove a context, or simply delete
``rm -rf $HOME/.kube/config`` if you want to start fresh.

.. code:: bash

    export ENVIRONMENT_NAME=kitware-prod-v4
    export AWS_PROFILE="iarpa"
    export AWS_REGION=us-west-2
    export AWS_ACCOUNT_ID=$(aws sts --profile "$AWS_PROFILE" get-caller-identity --query "Account" --output text)
    echo "
    Verify this is your correct kitware-smart AWS Account ID
    AWS_ACCOUNT_ID = $AWS_ACCOUNT_ID
    "

    aws eks --profile iarpa --region $AWS_REGION update-kubeconfig \
        --name "smartflow-${ENVIRONMENT_NAME}-eks" \
        --role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/smartflow-${ENVIRONMENT_NAME}-${AWS_REGION}-eks-admin"

Test that you can reach the service with:


.. code:: bash

   kubectl get svc


Troubleshooting
---------------

If you get the error message:


.. code:: bash

    error: exec plugin: invalid apiVersion "client.authentication.k8s.io/v1alpha1"


Update your AWS CLI and then reconfigure kubernetes by removing the previous
context and rerunning the configuration step.



Running the Webserver
---------------------

To access the airflow GUI we need tos tart a web service.

.. code:: bash

    kubectl -n airflow port-forward service/airflow-webserver 2746:8080

In your browser navigate to ``localhost:2746/home``, which can be done via the command:

.. code:: bash

   # Not working?
   python -c "import webbrowser; webbrowser.open('https://localhost:2746/home', new=1)"

   # python -c "import webbrowser; webbrowser.open('https://127.0.0.1:2746/home', new=1)"


References
----------

Blacksky also has detailed instructions for setting up smartflow and setting up DAGS.

* https://smartgitlab.com/blacksky/smartflow/-/blob/main/docs/Administration/Deployment.md

* https://blacksky.smartgitlab.com/smartflow/markdown/Framework/Getting-Started.html#authoring-your-first-dag


Next Steps
----------

* `running smartflow <smartflow_running_the_system.rst>`
* `copy large files to EFS <smartflow_copying_large_files_to_efs.md>`
* `training fusion models on AWS <smartflow_training_fusion_models.md>`
