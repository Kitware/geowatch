==============================
Getting Started With Smartflow
==============================


Requirements
------------
We assume you have the aws and kubernetes command line tools installed.

* `Get started with aws CLI <../environment/getting_started_aws.rst>`_.

* `Get started with kubectl CLI <../environment/getting_started_kubectl.rst>`_.


Configure Kubernetes
--------------------

To configure kubernetes to talk to the Kitware smartflow server run:


.. note::

    If you have an old configuration you should remove it. You can list
    existing contexts with

    ``kubectl config get-contexts``

    To remove a context:

    ``kubectl config delete-context <chosen-context>``

    If you want to start fresh, delete the .kube/config directory

    ``rm -rf $HOME/.kube/config``


.. code:: bash

    #export ENVIRONMENT_NAME=kitware-prod-v4

    export ENVIRONMENT_NAME="kw-v3-0-0"
    export AWS_PROFILE="iarpa"
    export AWS_REGION=us-west-2
    export AWS_ACCOUNT_ID=$(aws sts --profile "$AWS_PROFILE" get-caller-identity --query "Account" --output text)

    echo "
    Verify this is your correct kitware-smart AWS Account ID
    AWS_ACCOUNT_ID = $AWS_ACCOUNT_ID
    "

    aws eks --profile iarpa --region $AWS_REGION update-kubeconfig \
        --name "smartflow-${ENVIRONMENT_NAME}" \
        --role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/smartflow-${ENVIRONMENT_NAME}-${AWS_REGION}-eks-admin"

Test that you can reach the service with:


.. code:: bash

   kubectl get svc

   # or

   kubectl get nodes


Troubleshooting
---------------

If you get the error message:


.. code:: bash

    error: exec plugin: invalid apiVersion "client.authentication.k8s.io/v1alpha1"


Update your AWS CLI and then reconfigure kubernetes by removing the previous
context and rerunning the configuration step.


If you need permissions for something, it is helpful to send your account info which can be queried with:

.. code:: bash

   aws sts get-caller-identity


Running the Webserver
---------------------

To access the airflow GUI we need tos tart a web service.

.. code:: bash

    kubectl -n airflow port-forward service/airflow-webserver 2746:8080

In your browser navigate to ``localhost:2746/home``, which can be done via the command:

.. code:: bash

   # I'm not sure why, but does not seem to be working correctly.
   python -c "import webbrowser; webbrowser.open('https://localhost:2746/home', new=1)"


References
----------

Blacksky also has detailed instructions for setting up smartflow and setting up DAGS.

* https://smartgitlab.com/blacksky/smartflow/-/blob/main/docs/Administration/Deployment.md

* https://blacksky.smartgitlab.com/smartflow/markdown/Framework/Getting-Started.html#authoring-your-first-dag


Next Steps
----------

* `Running smartflow <smartflow_running_the_system.rst>`_

* `Copy large files to EFS <smartflow_copying_large_files_to_efs.md>`_

* `Training fusion models on AWS <smartflow_training_fusion_models.md>`_


SeeAlso
-------

k9s TUI - https://k9scli.io/topics/install/


https://smartgitlab.com/blacksky/smartflow


To upgrade smartflow


export ENVIRONMENT_NAME="kw-v3-0-0"
export AWS_PROFILE="iarpa"
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID=$(aws sts --profile "$AWS_PROFILE" get-caller-identity --query "Account" --output text)

.. code:: bash

    # Clone smartflow

    export AWS_PROFILE="iarpa"
    export SMARTFLOW_ENVIRONMENT_NAME="kw-v3-0-0"
    export SMARTFLOW_AWS_ACCOUNT_ID=$(aws sts --profile "$AWS_PROFILE" get-caller-identity --query "Account" --output text)

    echo $SMARTFLOW_AWS_ACCOUNT_ID
    python3 scripts/env_update.py \
        --aws_account_id $SMARTFLOW_AWS_ACCOUNT_ID \
        --environment_name $SMARTFLOW_ENVIRONMENT_NAME


# Can monitor progress in the cloud formation dashboard
https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2

To tune a parameter go to parameter store,

https://us-west-2.console.aws.amazon.com/systems-manager/parameters/?region=us-west-2&tab=Table


change it and then re-run the update script

To monitor EC2:

https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:instanceState=running;search=:smartflow;v=3;$case=tags:true%5C,client:false;$regex=tags:false%5C,client:false;sort=desc:tag:Name


To remove an old environment

.. code:: bash

    export AWS_PROFILE="iarpa"
    export TEARDOWN_SMARTFLOW_ENVIRONMENT_NAME="kitware-prod-v4-eks"
    export SMARTFLOW_AWS_ACCOUNT_ID=$(aws sts --profile "$AWS_PROFILE" get-caller-identity --query "Account" --output text)

    echo "SMARTFLOW_AWS_ACCOUNT_ID=$SMARTFLOW_AWS_ACCOUNT_ID"
    echo "TEARDOWN_SMARTFLOW_ENVIRONMENT_NAME=$TEARDOWN_SMARTFLOW_ENVIRONMENT_NAME"
    echo "AWS_PROFILE=$AWS_PROFILE"
    python3 scripts/env_teardown.py \
        --aws_account_id $SMARTFLOW_AWS_ACCOUNT_ID \
        --environment_name "$TEARDOWN_SMARTFLOW_ENVIRONMENT_NAME"
