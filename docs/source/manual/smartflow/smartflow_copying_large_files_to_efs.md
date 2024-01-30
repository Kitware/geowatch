
# Copying Large Files to EFS

## Prerequisites
Install:
- AWS CLI tool `aws` ([https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
- Kubernetes command-line tool `kubectl` ([https://kubernetes.io/docs/tasks/tools/#kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl))
- `rsync` installed on destination machine

## AWS Configuration
Run `aws configure`. This common will ask you for some parameters:
- `aws account id`: 023300502152
- `region`: us-west-2  
- `user name`, `access key id`, `secret key`: Coordinate with [Yoni](mailto:yonatan.gefen@kitware.com ) to get these if you don't have them already

## Configuring `kubctl` to reach smartflow
(From David Joy)

Once `kubectl` and `aws` are installed you'll want to configure it to be able to reach the cluster where Smartflow is running, here's a little bash script that should do that for you:
```
################################  
ENVIRONMENT_NAME=kitware-prod-v2  
################################  
AWS_ACCOUNT_ID=$(aws sts --profile iarpa get-caller-identity --query "Account" --output text)  
AWS_REGION=us-west-2  
  
aws eks --profile iarpa --region $AWS_REGION update-kubeconfig \  
	--name "smartflow-${ENVIRONMENT_NAME}-eks" \  
	--role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/smartflow-${ENVIRONMENT_NAME}-${AWS_REGION}-eks-admin"  
```

### `rsync` and kubernetes

Copying files to a kubernetes pod is tricky, below is a script which makes this less painful.
 [Source](https://serverfault.com/a/887402)

`krsync.sh`

```bash
#!/bin/bash

if [ -z "$KRSYNC_STARTED" ]; then
    export KRSYNC_STARTED=true
    exec rsync --blocking-io --rsh "$0" $@
fi

# Running as --rsh
namespace=''
pod=$1
shift

# If use uses pod@namespace rsync passes as: {us} -l pod namespace ...
if [ "X$pod" = "X-l" ]; then
    pod=$1
    shift
    namespace="-n $1"
    shift
fi

exec kubectl $namespace exec -i $pod -- "$@"
```

Then you can use krsync where you would normally rsync:
```
krsync -av --progress --stats src-dir/ pod:/dest-dir
```

Or you can set the namespace:
```
krsync -av --progress --stats src-dir/ pod@namespace:/dest-dir
```

# Connecting to Smartflow
You can forward the Smartflow GUI port to your local machine with the following command:

```
>>> kubectl -n airflow port-forward service/airflow-webserver 8080:8080  
```

And then reach the GUI at: [http://localhost:8080](http://localhost:8080/)

# Launch a `KIT_DEMO_WAIT`  job
Find the `KIT_DEMO_WAIT` job in the main interface. To launch, click the green "Play" button, and choose to run with params. Set the wait time to the amount of time you estimate rsyncing and then unpacking your dataset will take (plus a healthy buffer to account for error and slowdowns).

# Finally we can copy!
To find the $POD_ADDR of your waiting pod: 
```
kubectl -n airflow get pods
```

To log into it (if necessary to install `rsync` or other packages):
```
kubectl -n airflow exec -it pods/$POD_ADDR -- bash
```

To copy files to the EFS share mounted to the pod:
```
krsync -av --progress --stats $DATA_FPATH $POD_ADDR:/efs/work/$DEST_FPATH
```

In my experience, copying from KHQ runs at around 8-10 MB/s.

# Common pitfalls
1. When rsyncing a tarball, make sure to follow [[Dealing with tar error, cannot change owner]] when untarring the file.
2. You must have rsync executable in the pod image for this to work. On a minimal Docker image: `apt update; apt install rsync`
