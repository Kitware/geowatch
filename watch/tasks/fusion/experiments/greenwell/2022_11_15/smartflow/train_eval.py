from airflow import DAG
from airflow.models import Param
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s
from smartflow.operators.pod import create_pod_task

from datetime import datetime
from textwrap import dedent

DVC_DATA_DPATH = "/efs/work/greenwell/data/toy2_data_dvc"
DVC_EXPT_DPATH = "/efs/work/greenwell/data/toy2_expt_dvc"

with DAG(
    dag_id="KIT_DEMO_TRAIN",
    description="Kitware demo, starts a GPU node and then waits, allows user to login and examine the created node",
    params={
        # "dataset_name": Param(default="Drop4-BAS", type="string"),
        "smart_version_tag": Param(default="main", type="string"),
        # "experiment_bash_path": Param(default="experiments/greenwell/2022_11_01/project_data/expt_drop4_bas_native.sh", type="string"),
    },
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    default_view="grid",
    tags=["demo", "watch", "training"],
    start_date=datetime(2022, 3, 1),
) as dag:

    # """
    # Downloads (if necessary) the prescribed dataset from the DVC cache to the EFS work mount.
    # """
    # download_data = create_pod_task(
    #     task_id="download_data",
    #     image="amazon/aws-cli:latest",
    #     cmds=["bash", "-exc"],
    #     arguments=[
    #         dedent(
    #             "{{ params.dataset_name }}",
    #             """
    #             git clone
    #             """
    #         )
    #     ],
    #     purpose="general",
    #     cpu_limit="15",
    #     memory_limit="28G",
    #     mount_dshm=True,
    #     mount_efs_work=True,
    # )

    """
    Pulls down the SMART codebase and checks out the specified tag.
    Then trains model using specified bash script which defines the training approach. Typically, this will internally refer to config files.
    Finally, copies resulting model package to EFS.
    TODO: enable max_retries
    """
    train_model = create_pod_task(
        task_id="train_model",
        image="registry.smartgitlab.com/kitware/watch/ta2:Oct31-debug11",
        secrets=[
            Secret('env', 'WATCH_GITLAB_USERNAME', 'watch-gitlab-repo', 'username'),
            Secret('env', 'WATCH_GITLAB_PASSWORD', 'watch-gitlab-repo', 'password'),
        ],
        env_vars=[
            k8s.V1EnvVar(name="DVC_DATA_DPATH", value=DVC_DATA_DPATH),
            k8s.V1EnvVar(name="DVC_EXPT_DPATH", value=DVC_EXPT_DPATH),
        ],
        cmds=["bash", "-exc"],
        arguments=[
            dedent(
                """
                mkdir -p /root/code
                git clone https://$WATCH_GITLAB_USERNAME:$WATCH_GITLAB_PASSWORD@gitlab.kitware.com/smart/watch.git /root/code/watch
                cd /root/code/watch
                git remote update; git checkout {{ params.smart_version_tag }}

                # source run_developer_setup.sh
                pip install -r requirements.txt -v
                pip install -r requirements/gdal.txt
                # pip install -r requirements/headless.txt
                pip install -e .

                pip install -U delayed-image
                python -c "import delayed_image; print('delayed_image.version = ', delayed_image.__version__)"

                smartwatch_dvc add --name=toy_data_hdd --path=$DVC_DATA_DPATH --hardware=hdd --priority=100 --tags=toy_data_hdd
                mkdir -p $DVC_DATA_DPATH
                smartwatch_dvc add --name=toy_expt_hdd --path=$DVC_EXPT_DPATH --hardware=hdd --priority=100 --tags=toy_expt_hdd
                mkdir -p $DVC_EXPT_DPATH

                USER=airflow_root
                source /root/code/watch/watch/tasks/fusion/experiments/greenwell/examples/heterogeneous_native_msi.sh
                """
            )
        ],
        purpose="gpu-nvidia-t4-c32-m128-g1-od",  # TODO: choose multi-gpu node
        cpu_limit="15",
        memory_limit="28G",
        mount_dshm=True,
        mount_efs_work=True,
    )

    # """
    # Pulls down the SMART codebase and checks out the specified tag.
    # Then trains model using specified bash script which defines the training approach. Typically, this will internally refer to config files.
    # Finally, copies resulting model package to EFS.
    # TODO: enable max_retries
    # """
    # eval_model = create_pod_task(
    #     task_id="eval_model",
    #     image="registry.smartgitlab.com/kitware/watch/ta2:Oct31-debug11",
    #     cmds=["sleep"],
    #     arguments=[
    #         "{{ params.smart_version_tag }}",
    #         "{{ params.experiment_bash_path }}",
    #     ],
    #     purpose="gpu-nvidia-t4-c32-m128-g1-od", # TODO: choose single-gpu node
    #     cpu_limit="15",
    #     memory_limit="28G",
    #     mount_dshm=True,
    #     mount_efs_work=True,
    # )

    (
        # download_data >>
        train_model
        # >> eval_model
    )
