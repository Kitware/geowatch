from airflow import DAG
from airflow.models import Param
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s
from smartflow.operators.pod import create_pod_task

from datetime import datetime
from textwrap import dedent

with DAG(
    dag_id="KIT_TRAIN_WITH_YAML",
    description="Kitware demo: creates toy data, fits a model to it, makes predictions, and evaluates them.",
    params={
        "smart_version_tag": Param(default="main", type="string"),
        "dataset_name": Param(default="Drop4-BAS", type="string"),
        "experiment_name": Param(default="Baseline", type="string"),
        "experiment_yaml_relpath":
            Param(
                default="watch/tasks/fusion/experiments/greenwell/2023_01_05/bas_wvonly/config_common.yaml",
                type="string",
            ),
    },
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    default_view="grid",
    tags=["watch", "training"],
    start_date=datetime(2022, 3, 1),
) as dag:

    DATA_ROOT = "/efs/work/greenwell/data"
    DVC_EXPT_DPATH = f"{DATA_ROOT}/smart_expt_dvc"
    WORKDIR = f"{DVC_EXPT_DPATH}/training/smartflow/airflow_root"

    train_model = create_pod_task(
        task_id="train_model",
        image="registry.smartgitlab.com/kitware/watch/ta2:Oct31-debug11",
        secrets=[
            Secret('env', 'WATCH_GITLAB_USERNAME', 'watch-gitlab-repo', 'username'),
            Secret('env', 'WATCH_GITLAB_PASSWORD', 'watch-gitlab-repo', 'password'),
        ],
        env_vars=[
            k8s.V1EnvVar(name="WORKDIR", value=WORKDIR),
            k8s.V1EnvVar(name="DATA_ROOT", value=DATA_ROOT),
        ],
        cmds=["bash", "-exc"],
        arguments=[
            dedent(
                """
                ##############################################
                # Setup environment and codebase
                ##############################################
                mkdir -p /root/code
                git clone https://$WATCH_GITLAB_USERNAME:$WATCH_GITLAB_PASSWORD@gitlab.kitware.com/smart/watch.git /root/code/watch
                cd /root/code/watch
                git remote update; git checkout {{ params.smart_version_tag }}

                # source run_developer_setup.sh
                pip install -r requirements.txt -v
                pip install -r requirements/gdal.txt
                # pip install -r requirements/headless.txt
                pip install -e .

                # TODO: The following may not be necessary forever
                pip install -U delayed-image
                python -c "import delayed_image; print('delayed_image.version = ', delayed_image.__version__)"


                ##############################################
                # Train the model
                ##############################################
                CONFIG_FPATH=/root/code/watch/{{ params.experiment_yaml_relpath }}
                TRAIN_FPATH=$DATA_ROOT/{{ params.dataset_name }}/data_train.kwcoco.json
                VALI_FPATH=$DATA_ROOT/{{ params.dataset_name }}/data_vali.kwcoco.json
                DEFAULT_ROOT_DIR=$WORKDIR/{{ params.dataset_name }}/runs/{{ params.experiment_name }}
                python -m watch.tasks.fusion fit \
                    --config="$CONFIG_FPATH" \
                    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
                    --data.train_dataset="$TRAIN_FPATH" \
                    --data.vali_dataset="$VALI_FPATH" \
                    --data.num_workers=16 \
                    --trainer.max_steps=1000000 \
                    --trainer.accelerator="gpu" \
                    --trainer.precision=16 \
                    --trainer.devices="0,"

                smartwatch torch_model_stats "$DEFAULT_ROOT_DIR"/final_package.pt --stem_stats=True
                """
            )
        ],
        purpose="gpu-nvidia-t4-c32-m128-g1-od", # TODO: choose multi-gpu node
        cpu_limit="31",
        memory_limit="120G",
        mount_dshm=True,
        mount_efs_work=True,
    )

    predict = create_pod_task(
        task_id="predict",
        image="registry.smartgitlab.com/kitware/watch/ta2:Oct31-debug11",
        cmds=["bash", "-exc"],
        env_vars=[
            k8s.V1EnvVar(name="DVC_EXPT_DPATH", value=DVC_EXPT_DPATH),
            k8s.V1EnvVar(name="WORKDIR", value=WORKDIR),
            k8s.V1EnvVar(name="DATA_ROOT", value=DATA_ROOT),
        ],
        arguments=[
            dedent(
                """
                # TODO: build docker image with these steps already done
                cd /watch
                pip install -r requirements/development.txt
                pip install -r requirements/runtime.txt
                pip install -r requirements/optional.txt
                pip install -r requirements/gdal.txt
                pip install -e .

                TEST_FPATH=$DATA_ROOT/{{ params.dataset_name }}/data_vali.kwcoco.json
                DEFAULT_ROOT_DIR=$WORKDIR/{{ params.dataset_name }}/runs/{{ params.experiment_name }}"
                python -m watch.tasks.fusion.predict \
                    --test_dataset="$TEST_FPATH" \
                    --package_fpath="$DEFAULT_ROOT_DIR"/final_package.pt  \
                    --pred_dataset="$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json

                smartwatch stats "$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json
                """
            )
        ],
        purpose="gpu-nvidia-t4-c32-m128-g1-od", # TODO: choose single-gpu node
        cpu_limit="15",
        memory_limit="28G",
        mount_dshm=True,
        mount_efs_work=True,
    )

    eval_model = create_pod_task(
        task_id="eval_model",
        image="registry.smartgitlab.com/kitware/watch/ta2:Oct31-debug11",
        cmds=["bash", "-exc"],
        env_vars=[
            k8s.V1EnvVar(name="DVC_EXPT_DPATH", value=DVC_EXPT_DPATH),
            k8s.V1EnvVar(name="WORKDIR", value=WORKDIR),
            k8s.V1EnvVar(name="DATA_ROOT", value=DATA_ROOT),
        ],
        arguments=[
            dedent(
                """
                # TODO: build docker image with these steps already done
                cd /watch
                pip install -r requirements/development.txt
                pip install -r requirements/runtime.txt
                pip install -r requirements/optional.txt
                pip install -r requirements/gdal.txt
                pip install -e .

                TEST_FPATH=$DATA_ROOT/{{ params.dataset_name }}/data_vali.kwcoco.json
                DEFAULT_ROOT_DIR=$WORKDIR/{{ params.dataset_name }}/runs/{{ params.experiment_name }}"
                python -m watch.tasks.fusion.evaluate \
                    --true_dataset="$TEST_FPATH" \
                    --pred_dataset="$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json \
                    --eval_dpath="$DVC_EXPT_DPATH"/predictions/eval
                """
            )
        ],
        purpose="gpu-nvidia-t4-c32-m128-g1-od",
        cpu_limit="15",
        memory_limit="28G",
        mount_dshm=True,
        mount_efs_work=True,
    )

    train_model >> predict >> eval_model
