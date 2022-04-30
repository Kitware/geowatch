#!/bin/bash
__doc__='
This script assumes that the repo and permissions are setup.

The entrypoint will update the repo to the latest version of whatever branch
the Docker image was created with. 

Note: you must edit "ta2_train_workflow.yml" to specify the branch should be
used.

To submit these jobs run something like:

    cd "$HOME/code/watch/aws"

    WORKFLOW_FPATH=$HOME/code/watch/aws/ta2_train_workflow.yml
    argo submit "$WORKFLOW_FPATH" --watch

    WORKFLOW_FPATH=$HOME/code/watch/aws/ta2_train_workflow.yml
    NAME_PREFIX=$(yq -r .metadata.generateName "$WORKFLOW_FPATH")
    WORKFLOW_NAME=$(argo list --running | argo list --running | grep "$NAME_PREFIX" | head -n 1 | cut -d" " -f1)
    argo logs "${WORKFLOW_NAME}" --follow

    # NOTE: It usually takes ~12-15 minutes for a job to startup after being
    # submitte

    # And then view the logs in real time (note: if the workflow ends, you need
    # to use the UI to access the old logs)
    # This is not 100% reliable has race conditions
    # Get a shell in the pod

    WORKFLOW_FPATH=$HOME/code/watch/aws/ta2_train_workflow.yml
    NAME_PREFIX=$(yq -r .metadata.generateName "$WORKFLOW_FPATH")
    WORKFLOW_NAME=$(argo list --running | argo list --running | grep "$NAME_PREFIX" | head -n 1 | cut -d" " -f1)
    kubectl exec $WORKFLOW_NAME -- ls -al /root
    
    echo "WORKFLOW_NAME = $WORKFLOW_NAME"
    kubectl exec --stdin --tty $WORKFLOW_NAME -- /bin/bash

kubectl exec -it ta2-train-sd8qs -c main -- /bin/bash

    kubectl exec --stdin --tty ta2-train-cdkc4 -- /bin/bash

    kubectl -n argo exec ta2-train-vxqb7 -- bash
    
    # Use this to check outputs
    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/
'
set -ex

export PYTHONUNBUFFERED=1
export SMART_DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
export WATCH_REPO_DPATH=$HOME/code/watch
source /opt/conda/etc/profile.d/conda.sh
conda activate watch

# Initialize the base DVC repo with AWS permissions
mkdir -p "$HOME/data/dvc-repos"
git clone "https://${DVC_GITLAB_USERNAME}:${DVC_GITLAB_PASSWORD}@gitlab.kitware.com/smart/smart_watch_dvc.git" "$SMART_DVC_DPATH"

cd "$SMART_DVC_DPATH"
dvc remote add aws-noprofile s3://kitware-smart-watch-data/dvc

ls -al "$HOME"

# Grab the required datasets that we need
#dvc pull Aligned-Drop3-L1/splits.zip.dvc -r aws-noprofile --quiet
#dvc checkout Drop1-Aligned-TA1-2022-01/data.kwcoco.json.dvc

if [[ "$HOSTNAME" == "" ]]; then 
    export HOSTNAME="unknown-host"
fi
if [[ "$USER" == "" ]]; then 
    export USER="unknown-user"
fi


export DVC_DPATH=$SMART_DVC_DPATH
export WORKDIR="$DVC_DPATH/training/$HOSTNAME/$USER"

mkdir -p "$WORKDIR"

#cat ~/.aws/config

# All outputs will be saved in a "workdir"
# Startup background process that will write data to S3 in realish time
source "$WATCH_REPO_DPATH/aws/smartwatch_s3_sync.sh"

CHECKIN_FPATH=$WORKDIR/ACK-$(date +"%Y%m%dT%H%M%S").txt
echo "check-in" > "$CHECKIN_FPATH"
cat /proc/cpuinfo >> "$CHECKIN_FPATH" || echo "no cpuinfo" >> "$CHECKIN_FPATH"
nvidia-smi >> "$CHECKIN_FPATH" || echo "no nvidia-smi" >> "$CHECKIN_FPATH"
echo "CHECKIN_FPATH='$CHECKIN_FPATH'"
#cat "$CHECKIN_FPATH"
smartwatch_s3_sync_single "$WORKDIR"


# HACK: how to specify dont use a profile to a --profile command?
aws s3 ls s3://kitware-smart-watch-data/sync_root/
#aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/

smartwatch_s3_sync_forever_in_tmux "$WORKDIR"


# The following script should run the toy experiments end-to-end
export NDSAMPLER_DISABLE_OPTIONAL_WARNINGS=1

#pip install pytorch_lightning -U
nvidia-smi
python -c "import torch; print('torch.__version__ = {}'.format(torch.__version__))"
python -c "import torch; print('torch.__file__ = {}'.format(torch.__file__))"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"


##### MSI TEST
#export NUM_TOY_TRAIN_VIDS=100
#export NUM_TOY_VALI_VIDS=5
#export NUM_TOY_TEST_VIDS=2
#source "$WATCH_REPO_DPATH/watch/tasks/fusion/experiments/crall/toy_experiments_msi.sh"


##### Real work
#export AWS_PROFILE=iarpa
#export AWS_DEFAULT_PROFILE=iarpa
#export AWS_REQUEST_PAYER='requester'

#apt-get update
#apt-get install unzip -y

cd "$SMART_DVC_DPATH"
ls -al
DATASET_CODE=Aligned-Drop3-L1
cd "$SMART_DVC_DPATH/$DATASET_CODE"
ls -al


__devnote__='
# The data was initialy copied into the efsdata cache manually via

SMART_DVC_DPATH=$(smartwatch_dvc)
du -sh $SMART_DVC_DPATH/.dvc/cache

mkdir -p /efsdata/smart_watch_dvc/.dvc/
rsync -vrPR $SMART_DVC_DPATH/.dvc/./cache 
'

if [ -d /efsdata/smart_dvc_cache ]; then
    # If we have a mounted efs cache, then lets use it
    apt update || true
    apt install rsync
    mkdir -p "$SMART_DVC_DPATH"/.dvc/cache
    time rsync -nvrPR -p /efsdata/smart_watch_dvc/.dvc/./cache "$SMART_DVC_DPATH/.dvc/"
    dvc checkout splits.zip.dvc
    # This seems to fail the first time. No idea why that is. Try it a few times.
    dvc checkout -R . || \
        dvc checkout -R . || \
        dvc checkout -R . || \
        dvc checkout -R . 
else
    dvc pull splits.zip.dvc -r aws-noprofile 
    # This seems to fail the first time. No idea why that is. Try it a few times.
    dvc pull -R . -r aws-noprofile || \
        dvc pull -R . -r aws-noprofile || \
        dvc pull -R . -r aws-noprofile || \
        dvc pull -R . -r aws-noprofile 
fi

#sleep 28800
unzip -o splits.zip
#7z x splits.zip

DVC_DPATH=$(smartwatch_dvc)
echo "DVC_DPATH = $DVC_DPATH"
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-L1
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=L1_BASELINE_EXPERIMENT_${HOSTNAME}_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME


export CUDA_VISIBLE_DEVICES=0
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=L1_Template\
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus 1 \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=none \
    --draw_interval=5000m \
    --num_draw=1 \
    --amp_backend=apex \
    --init="$INITIAL_STATE" \
    --num_sanity_val_steps=0 \
    --dump "$WORKDIR/configs/drop3_l1_baseline_20220425.yaml"

#export CUDA_VISIBLE_DEVICES=0
#DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-L1
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=L1_BASELINE_AWS_${HOSTNAME}_V006
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_l1_baseline_20220425.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --max_epoch_length=496 \
    --time_sampling=hardish3 \
    --chip_size=256 \
    --time_steps=11 \
    --time_span=6m \
    --accumulate_grad_batches=3 \
    --learning_rate=3e-4 \
    --max_epochs=160 \
    --patience=160 \
    --gpus 1 \
    --num_workers=6 \
    --init="$INITIAL_STATE"

__doc__='

Execute instructions:

    argo submit "$HOME/code/watch/aws/ta2_train_workflow.yml" --watch

    argo_follow_recent(){
        WORKFLOW_FPATH=$HOME/code/watch/aws/ta2_train_workflow.yml
        NAME_PREFIX=$(yq -r .metadata.generateName "$WORKFLOW_FPATH")
        WORKFLOW_NAME=$(argo list --running | argo list --running | grep "$NAME_PREFIX" | head -n 1 | cut -d" " -f1)
        #argo logs "${WORKFLOW_NAME}" --follow
        # kubctl gives better full logs
        kubectl logs "${WORKFLOW_NAME}" -c main --follow  
    }
    argo_follow_recent

    # Use this to check outputs
    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/ta2-train-xzzwv
    mkdir -p $HOME/data/aws-sync
    aws s3 --profile iarpa sync s3://kitware-smart-watch-data/sync_root/ta2-train-xzzwv/ $HOME/data/aws-sync


    TODO:
    Allow hard coded specs for the following:
        datamodule.dataset_stats = {
            "sensor_mode_hist": {
                ("L8", "blue|green|red|nir|swir16|swir22"): 1546,
                ("S2", "blue|green|red|nir|swir16|swir22"): 3574,
            },
            "input_stats": {
                ("L8", "blue|green|red|nir|swir16|swir22"): {
                    "mean": np.array([[[11602.561]],[[11679.435]],[[12304.289]],[[15570.234]],[[15133.022]],[[12781.117]]], dtype=np.float64),
                    "std": np.array([[[3267.417]],[[3930.465]],[[4846.832]],[[5261.937]],[[6077.674]],[[5040.008]]], dtype=np.float64),
                },
                ("S2", "blue|green|red|nir|swir16|swir22"): {
                    "mean": np.array([[[1250.079]],[[1264.795]],[[1408.939]],[[1839.118]],[[1887.035]],[[1453.906]]], dtype=np.float64),
                    "std": np.array([[[1016.882]],[[1088.721]],[[1313.853]],[[1473.62 ]],[[1633.727]],[[1320.249]]], dtype=np.float64),
                },
            },
            "class_freq": {
                "background": 719552801,
                "ignore": 4800073,
                "Unknown": 102704,
                "positive": 12401990,
                "negative": 0,
                "Site Preparation": 144313,
                "Active Construction": 818806,
                "Post Construction": 1335078,
                "No Activity": 172235,
    },
}
     
'
