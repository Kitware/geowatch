# Prerequisites

Running smartflow requires setting up aws and kubectl. 
We have [streamlined aws and kubctl install instructions](../environment/getting_started_aws.rst) but you may also fine the
official resources useful:

Install:
- AWS CLI tool `aws` ([https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
- Kubernetes command-line tool `kubectl` ([https://kubernetes.io/docs/tasks/tools/#kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl))
- 

## AWS Configuration

Run `aws configure`. This common will ask you for some parameters:
-   `aws account id`: 023300502152
-   `region`: us-west-2
-   `user name`, `access key id`, `secret key`: Coordinate with [Yoni](mailto:yonatan.gefen@kitware.com) to get these if you don't have them already

## Configuring `kubctl` to reach smartflow
(From David Joy)

Once `kubectl` and `aws` are installed you'll want to configure it to be able to reach the cluster where Smartflow is running, here's a little bash script that should do that for you:

```
ENVIRONMENT_NAME=kitware-prod-v2  
export AWS_PROFILE="iarpa"
AWS_ACCOUNT_ID=$(aws sts --profile "$AWS_PROFILE" get-caller-identity --query "Account" --output text)  
echo "Verify this is your correct kitware-smart AWS Account ID"
echo "AWS_ACCOUNT_ID = $AWS_ACCOUNT_ID"
AWS_REGION=us-west-2  
  
aws eks --profile iarpa --region $AWS_REGION update-kubeconfig \  
	--name "smartflow-${ENVIRONMENT_NAME}-eks" \  
	--role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/smartflow-${ENVIRONMENT_NAME}-${AWS_REGION}-eks-admin"  
```


# Connecting to Smartflow

You can forward the Smartflow GUI port to your local machine with the following command:

```
kubectl -n airflow port-forward service/airflow-webserver 8080:8080  
```

And then reach the GUI at: [http://localhost:8080](http://localhost:8080/)

# Launch a `KIT_TRAIN_WITH_YAML` job

Find the `KIT_TRAIN_WITH_YAML` job in the main interface. To launch, click the green "Play" button, and choose to "Trigger DAG w/ config". Currently available options:
- `smart_version_tag`: Defaults to "main". This is the git tag for the smart repo that you want to run your experiment. You will want to change this if you have changes in a feature branch you want to try.
- `dataset_name`: Defaults to "Drop4-BAS". This is the name of the dataset stored on EFS that your model will be trained against. The assumed full path to the training kwcoco file (at this time) will be `/efs/work/greenwell/data/smart_data_dvc/$DATASET_NAME/data_train.kwcoco.json`.
- `experiment_name`: Defaults to "Baseline". Determines, among other things, where the default logs will write to.
- `experiment_yaml_relpath`: The relative path within the smart codebase to a yaml file describing your fusion experiment.

# Copying your own DAG to Smartflow's S3 Bucket

```
aws s3 --profile iarpa cp $DAG_FNAME s3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v2/dags/$DAG_FNAME
```

# Logging into a running DAG step / K8s pod

```
kubectl -n airflow get pods
kubectl -n airflow exec -it pods/$POD_ADDR -- bash
```
