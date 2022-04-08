#!/bin/bash
__doc__="

Requirements:

    * The SMART WATCH Python repo must be installed in the current python virtualenv

    * The SMART WATCH DVC Repo should be checked out and in a known location

    * The Drop2-Aligned-TA1-2022-01 dataset should be DVC-pulled so it is on-disk

"
#export CUDA_VISIBLE_DEVICES="1"

DVC_DPATH=/localdisk0/SCRATCH/watch/ben/smart_watch_dvc
DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

# Point to latest dataset version
DATASET_CODE=uky_invariants/features_22_03_14
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE

### TODO: CHANGE TO KWCOCO FILES THAT CONTAIN TEAM FEATURES
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json

### TODO: CHANGE INPUT CHANNELS TO NETWORK.
CHANNELS="blue|green|red|nir|swir16|swir22,invariants.0:7"
### e.g. "blue|green|red|nir|swir16|swir22,myfeat.0:8"

# Set initial state to a noop to train from scratch, or set it to an existing
# model to transfer the initial weights (as best as possible) using
# torch-liberator partial weight loading.
INITIAL_STATE="noop"
#INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"

### TODO: CHANGE TO A UNIQUE NAME FOR EACH EXPERIMENT
EXPERIMENT_NAME=features_late_fusion_V001

debug_notes(){
    # Print stats about train and validation datasets
    python -m watch stats "$VALI_FPATH" "$TRAIN_FPATH"
    python -m kwcoco stats "$VALI_FPATH" "$TRAIN_FPATH"
}

DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

__hyperparam_notes__='

The following hyperparams are reasonable defaults

Key hyperparams to pay attention to are:

* global_class_weight - when non-zero enables the SC classification head
* global_saliency_weight - when non-zero enables the BAS saliency head
* chip_size - pixel size of the spatial input window
* time_steps - number of frames to use
* time_sampling - strategy for temporal sampling. See --help for other options.
* tokenizer - method for breaking up the input data-cube into tokens
* normalize_inputs - number of dataset iterations to use to estimate mean/std for network normalization

See `python -m watch.tasks.fusion.fit --help` for details on each
hyperparameter.  Note, some parameters exposed in this help no longer work or
are not hooked up. Email Jon C if you have any questions.
'

python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
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
    --init="$INITIAL_STATE"


gather_checkpoint_notes(){
    __doc__="
    Every so often, I run the repackage command and gather the packaged
    checkpoints for evaluation.
    "

    # This method only works for the current fusion model
    # It would be better if the fit command was able to take care of this
    python -m watch.tasks.fusion.repackage repackage "$DEFAULT_ROOT_DIR/lightning_logs/version_*/checkpoints/*.ckpt"

    # To ensure the results of our experiments are maintained, we copy them to
    # the DVC directory.
    BASE_SAVE_DPATH=$DVC_DPATH/models/fusion/baseline
    EXPT_SAVE_DPATH=$BASE_SAVE_DPATH/$EXPERIMENT_NAME
    mkdir -p "$BASE_SAVE_DPATH"
    mkdir -p "$EXPT_SAVE_DPATH"

    cp "$DEFAULT_ROOT_DIR"/lightning_logs/version_*/checkpoints/*.pt "$EXPT_SAVE_DPATH"
}


predict_and_evaluate_checkpoints(){
    __doc__='
    Given the checkpoint candidates, we can "schedule" them for evaluation.
    This schedule evaluations script is a work in progress.
    There are two ways of using it:

    1. If run=0, all it does is build the appropriate bash commands to run 
       prediction and evaluation

    2. If run=1, it will launch the jobs via the hacky tmux-queue, which really
       should be a slurm queue. The number of jobs will depend on the setting
       of "--gpus". E.g. specify the index of the gpus to use --gpus="0,1,2,3"

    Note:
        This whole queueing system is a work in progress and if anyone knows
        any good libraries for Python that let you submit a bash job, specify
        how many concurrent jobs can be running at the same time, and allow
        jobs to depend on other jobs, let me know.  If this doesnt exist I want
        to make it with multiprocessing, tmux, and slurm backends.
    '
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0," \
            --model_globstr="$EXPT_SAVE_DPATH/*.pt" \
            --test_dataset="$VALI_FPATH" \
            --run=0 --skip_existing=True
}


aggregate_multiple_evaluations(){
    __doc__="
    This script will aggregate results over all packaged checkpoints with
    computed metrics. You can run this while the schedule_evaluation script is
    running. It will dump aggregate stats into the 'out_dpath' folder.
    "

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)

    EXPT_NAME_PAT="EARLY_FUSION_V001"
    MODEL_EPOCH_PAT="*"
    PRED_DSET_PAT="*"
    MEASURE_GLOBSTR=$DVC_DPATH/models/fusion/baseline/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/eval/curves/measures2.json
    python -m watch.tasks.fusion.aggregate_results \
        --measure_globstr="$MEASURE_GLOBSTR" \
        --out_dpath="$DVC_DPATH/agg_results/baseline" \
        --dset_group_key="Drop2-Aligned-TA1-2022-02-15_data_vali.kwcoco"

}
