#!/bin/bash
__doc__="

Requirements:

    * The SMART WATCH Python repo must be installed in the current python virtualenv

    * The SMART WATCH DVC Repo should be checked out and in a known location

    * The Drop2-Aligned-TA1-2022-01 dataset should be DVC-pulled so it is on-disk

"
#export CUDA_VISIBLE_DEVICES="1"

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

# Point to latest dataset version
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE

### TODO: CHANGE TO KWCOCO FILES THAT CONTAIN TEAM FEATURES
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json

### TODO: CHANGE INPUT CHANNELS TO NETWORK.
CHANNELS="blue|green|red|nir|swir16|swir22"
### e.g. "blue|green|red|nir|swir16|swir22,myfeat.0:8"

# Set initial state to a noop to train from scratch, or set it to an existing
# model to transfer the initial weights (as best as possible) using
# torch-liberator partial weight loading.
INITIAL_STATE="noop"
#INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"

### TODO: CHANGE TO A UNIQUE NAME FOR EACH EXPERIMENT
EXPERIMENT_NAME=BASELINE_EXPERIMENT_V001

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
