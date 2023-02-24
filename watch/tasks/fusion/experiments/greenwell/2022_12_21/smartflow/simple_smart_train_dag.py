from airflow import DAG
from airflow.models import Param
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s
from smartflow.operators.pod import create_pod_task

from datetime import datetime
from textwrap import dedent

with DAG(
    dag_id="KIT_TRAIN_NATIVE_BAS",
    description="Kitware demo: creates toy data, fits a model to it, makes predictions, and evaluates them.",
    params={
        "smart_version_tag": Param(default="main", type="string"),
    },
    catchup=False,
    schedule_interval=None,
    max_active_runs=1,
    default_view="grid",
    tags=["watch", "training"],
    start_date=datetime(2022, 3, 1),
) as dag:

    DVC_DATA_DPATH = "/efs/work/greenwell/data/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC"
    DVC_EXPT_DPATH = "/efs/work/greenwell/data/smart_expt_dvc"
    WORKDIR = f"{DVC_EXPT_DPATH}/training/smartflow/airflow_root"

    # Generate toy datasets
    TRAIN_FPATH = f"{DVC_DATA_DPATH}/data_train.kwcoco.json"
    VALI_FPATH = f"{DVC_DATA_DPATH}/data_vali.kwcoco.json"
    TEST_FPATH = f"{DVC_DATA_DPATH}/data_vali.kwcoco.json"

    EXPERIMENT_NAME = "Drop4-BAS_Heterogeneous"
    DATASET_CODE = "Drop4-BAS"
    DEFAULT_ROOT_DIR = f"{WORKDIR}/{DATASET_CODE}/runs/{EXPERIMENT_NAME}"

    """
    Training, Prediction, and Evaluation
    ------------------------------------

    Now that we are more comfortable with kwcoco files, lets get into the simplest
    and most direct way of training a fusion model. This is done by simply calling
    'watch.tasks.fusion.fit' as the main module. We will specify:

    * paths to the training and validation kwcoco files
    * what channels we want to early / late fuse (given by a kwcoco sensorchan spec)
    * information about the input chip size and temporal window
    * the underlying architecture
    * other deep learning hyperparameters

    In this tutorial we will use 'cpu' as our lightning accelerator. If you have an
    available gpu and want to use it, change this to 'gpu' and add the argument
    '--devices=0,'

    We will also specify a work directory that will be similar to directories used
    when real watch models are trained.
    """
    train_model = create_pod_task(
        task_id="train_model",
        image="registry.smartgitlab.com/kitware/watch/ta2:Oct31-debug11",
        secrets=[
            Secret('env', 'WATCH_GITLAB_USERNAME', 'watch-gitlab-repo', 'username'),
            Secret('env', 'WATCH_GITLAB_PASSWORD', 'watch-gitlab-repo', 'password'),
        ],
        env_vars=[
            k8s.V1EnvVar(name="EXPERIMENT_NAME", value=EXPERIMENT_NAME),
            k8s.V1EnvVar(name="DEFAULT_ROOT_DIR", value=DEFAULT_ROOT_DIR),
            k8s.V1EnvVar(name="TRAIN_FPATH", value=TRAIN_FPATH),
            k8s.V1EnvVar(name="VALI_FPATH", value=VALI_FPATH),
        ],
        cmds=["bash", "-exc"],
        arguments=[
            dedent(
                """
                ls /efs/work/greenwell
                echo "======================="
                ls /efs/work/greenwell/data
                echo "======================="
                ls /efs/work/greenwell/data/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC

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

                pip install -U delayed-image
                python -c "import delayed_image; print('delayed_image.version = ', delayed_image.__version__)"


                ##############################################
                # Train the model
                ##############################################
                python -m watch.tasks.fusion fit \
                    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
                    --data.train_dataset="$TRAIN_FPATH" \
                    --data.vali_dataset="$VALI_FPATH" \
                    --data.channels="red|green|blue|nir" \
                    --data.window_space_scale="10GSD" \
                    --data.input_space_scale="native" \
                    --data.output_space_scale="30GSD" \
                    --data.time_steps=3 \
                    --data.chip_size=128 \
                    --data.batch_size=64 \
                    --data.num_workers=30 \
                    --model=watch.tasks.fusion.methods.HeterogeneousModel \
                        --model.name="$EXPERIMENT_NAME" \
                        --model.token_width=16 \
                        --model.token_dim=32 \
                        --model.position_encoder=watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder \
                            --model.position_encoder.in_dims=3 \
                            --model.position_encoder.max_freq=3 \
                            --model.position_encoder.num_freqs=16 \
                        --model.backbone=watch.tasks.fusion.architectures.transformer.TransformerEncoderDecoder \
                            --model.backbone.encoder_depth=6 \
                            --model.backbone.decoder_depth=1 \
                            --model.backbone.dim=128 \
                            --model.backbone.queries_dim=96 \
                            --model.backbone.logits_dim=32 \
                    --optimizer=torch.optim.AdamW \
                        --optimizer.lr=1e-3 \
                        --optimizer.weight_decay=1e-3 \
                    --trainer.max_steps=1000000 \
                    --trainer.accelerator="gpu" \
                    --trainer.precision=16 \
                    --trainer.devices="0,"

                smartwatch torch_model_stats "$DEFAULT_ROOT_DIR"/final_package.pt --stem_stats=True
                """
            )
        ],
        purpose="gpu-nvidia-t4-c32-m128-g1-od",  # TODO: choose multi-gpu node
        cpu_limit="31",
        memory_limit="120G",
        mount_dshm=True,
        mount_efs_work=True,
    )

    """
    Now that we have an understanding of what metadata the model contains, we can
    start to appreciate the dead simplicity of predicting with it.

    To use a model to predict on an unseed kwcoco dataset (in this case the toy
    test set) we simply call the "watch.tasks.fusion.predict" script and pass it:

       * the kwcoco file of the dataset to predict on
       * the path to the model we want to predict with
       * the name of the output kwcoco file that will contain the predictions

    All necessary metadata you would normally have to (redundantly) specify in
    other frameworks is inferred by programmatically reading the model. You also
    have the option to overwrite prediction parameters. See --help for details, but
    for now lets just run with the defaults that match how the model was trained.

    Note that the test dataset contains groundtruth annotations. All annotations
    are stripped and ignored during prediction.

    The output of the predictions is just another kwcoco file, but it augments the
    input images with new channels corresponding to predicted heatmaps. We can use
    the "smartwatch stats" command to inspect what these new channels are.
    """
    predict = create_pod_task(
        task_id="predict",
        image="registry.smartgitlab.com/kitware/watch/ta2:Oct31-debug11",
        cmds=["bash", "-exc"],
        env_vars=[
            k8s.V1EnvVar(name="DVC_EXPT_DPATH", value=DVC_EXPT_DPATH),
            k8s.V1EnvVar(name="DEFAULT_ROOT_DIR", value=DEFAULT_ROOT_DIR),
            k8s.V1EnvVar(name="TEST_FPATH", value=TEST_FPATH),
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

                python -m watch.tasks.fusion.predict \
                    --channels="red|green|blue|nir" \
                    --test_dataset="$TEST_FPATH" \
                    --package_fpath="$DEFAULT_ROOT_DIR"/final_package.pt  \
                    --pred_dataset="$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json

                smartwatch stats "$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json
                """
            )
        ],
        purpose="gpu-nvidia-t4-c32-m128-g1-od",  # TODO: choose single-gpu node
        cpu_limit="15",
        memory_limit="28G",
        mount_dshm=True,
        mount_efs_work=True,
    )

    """
    The last step in this basic tutorial is to measure how good our model is.
    We can do this with pixelwise metrics.

    This is done by using "watch.tasks.fusion.evaluate" as the main module, and
    its arguments are:

        * The true kwcoco data with groundtruth annotations (i.e. the test dataset)
        * The pred kwcoco data that we predicted earlier
        * An output path for results
    """
    eval_model = create_pod_task(
        task_id="eval_model",
        image="registry.smartgitlab.com/kitware/watch/ta2:Oct31-debug11",
        cmds=["bash", "-exc"],
        env_vars=[
            k8s.V1EnvVar(name="DVC_EXPT_DPATH", value=DVC_EXPT_DPATH),
            k8s.V1EnvVar(name="TEST_FPATH", value=TEST_FPATH),
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
