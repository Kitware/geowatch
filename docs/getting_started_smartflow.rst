

Requirements
------------
We assume you have the aws and kubernetes command line tools installed. 

* `get started with aws <getting_started_aws.rst>`

* `get started with kubectl <getting_started_kubectl.rst>`


Configure Kubernetes
--------------------

To configure kubernetes to talk to the Kitware smartflow server run:


.. code:: bash

    ENVIRONMENT_NAME=kitware-prod-v2  
    export AWS_PROFILE="iarpa"
    AWS_ACCOUNT_ID=$(aws sts --profile "$AWS_PROFILE" get-caller-identity --query "Account" --output text)  
    echo "Verify this is your correct kitware-smart AWS Account ID"
    echo "AWS_ACCOUNT_ID = $AWS_ACCOUNT_ID"
    AWS_REGION=us-west-2  
      
    aws eks --profile iarpa --region $AWS_REGION update-kubeconfig \  
        --name "smartflow-${ENVIRONMENT_NAME}-eks" \  
        --role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/smartflow-${ENVIRONMENT_NAME}-${AWS_REGION}-eks-admin"  

Test that you can reach the service with:


.. code:: bash

   kubectl get svc


References
----------

Blacksky also has detailed instructions for setting up smartflow and setting up DAGS.

* https://smartgitlab.com/blacksky/smartflow/-/blob/main/docs/Administration/Deployment.md

* https://blacksky.smartgitlab.com/smartflow/markdown/Framework/Getting-Started.html#authoring-your-first-dag


Next Steps
----------

* ./smartflow_copying_large_files_to_efs.md
* ./smartflow_training_fusion_models.md
* ./smartflow_running_the_system.rst
