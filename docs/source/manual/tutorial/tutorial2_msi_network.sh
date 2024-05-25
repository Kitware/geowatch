#!/bin/bash
__doc__="
This demonstrates an end-to-end pipeline on multispectral toydata

This walks through the entire process of fit -> predict -> evaluate and the
output if you run this should end with something like

source ~/code/geowatch/docs/source/manual/tutorial/tutorial2_msi_network.sh
"

# Define wherever you want to store results and create the directories
# if they don't already exist.
DVC_DATA_DPATH=$HOME/data/dvc-repos/toy_data_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/toy_expt_dvc

# Create the above directories of they don't exist
mkdir -p "$DVC_DATA_DPATH"
mkdir -p "$DVC_EXPT_DPATH"

# The user can overwrite these variables, but will default if they are not given.
NUM_TOY_TRAIN_VIDS="${NUM_TOY_TRAIN_VIDS:=100}"  # If variable not set or null, use default.
NUM_TOY_VALI_VIDS="${NUM_TOY_VALI_VIDS:=5}"  # If variable not set or null, use default.
NUM_TOY_TEST_VIDS="${NUM_TOY_TEST_VIDS:=2}"  # If variable not set or null, use default.

# Generate toy datasets
TRAIN_FPATH=$DVC_DATA_DPATH/vidshapes_msi_train${NUM_TOY_TRAIN_VIDS}/data.kwcoco.json
VALI_FPATH=$DVC_DATA_DPATH/vidshapes_msi_vali${NUM_TOY_VALI_VIDS}/data.kwcoco.json
TEST_FPATH=$DVC_DATA_DPATH/vidshapes_msi_test${NUM_TOY_TEST_VIDS}/data.kwcoco.json

generate_data(){
    kwcoco toydata --key="vidshapes${NUM_TOY_TRAIN_VIDS}-frames5-randgsize-speed0.2-msi-multisensor" \
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_train${NUM_TOY_TRAIN_VIDS}" --verbose=1

    kwcoco toydata --key="vidshapes${NUM_TOY_VALI_VIDS}-frames5-randgsize-speed0.2-msi-multisensor" \
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_vali${NUM_TOY_VALI_VIDS}"  --verbose=1

    kwcoco toydata --key="vidshapes${NUM_TOY_TEST_VIDS}-frames6-randgsize-speed0.2-msi-multisensor" \
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_test${NUM_TOY_TEST_VIDS}" --verbose=1
}

print_stats(){
    # Print stats
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"
    geowatch stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"
}

# If the train file path does not exist then
if [[ ! -e "$TRAIN_FPATH" ]]; then
    generate_data
    print_stats
fi

__doc__="
Should look like
                                   dset  n_anns  n_imgs  n_videos  n_cats  r|g|b|disparity|gauss|B8|B11  B1|B8|B8a|B10|B11  r|g|b|flowx|flowy|distri|B10|B11
0  vidshapes_msi_train/data.kwcoco.json      80      40         8       3                            12                 12                                16
1   vidshapes_msi_vali/data.kwcoco.json      50      25         5       3                             9                 10                                 6
2   vidshapes_msi_test/data.kwcoco.json      24      12         2       3                             5                  3                                 4
"

demo_visualize_toydata(){

    python -m geowatch.cli.coco_visualize_videos \
        --src "$TRAIN_FPATH" \
        --channels="gauss|B11,r|g|b,B1|B8|B11" \
        --viz_dpath="$DVC_DATA_DPATH/vidshapes_msi_train_viz" --animate=False
}


# Run the function if you want to visualize the data
# demo_visualize_toydata

# Define the channels we want to use
# The sensors and channels are specified by the kwcoco SensorChanSpec
# in this example the data does not contain sensor metadata, so we
# use a "*" to indicate a generic sensor.
# A colon ":" separates channels from the sensors.
# A pipe "|" indicates channels are early fused
# A "," indicates groups of early fused channels are late fused Multiple
# sensors can be specified to the left of the channels and will distribute over
# commas.
# Note: the channel names dont really mean anything. The demo data generated
# for them is random / arbitrary. They simply simulate a case where you do have
# a meaningful multiple-channel dataset you want to learn with.
CHANNELS="(*):(disparity|gauss,X.2|Y:2:6,B1|B8a,flowx|flowy|distri)"
echo "CHANNELS = $CHANNELS"

echo "
In Tutorial #2 we expand the complexity of the 'fit' command compared to the
previous tutorial. We define some of the important parameters as environment
variables to illustrate relationships between them.
"

# Fit
DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyDataMSI_Demo_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")


# Try to set the accelerator to "GPU", but fallback to cpu if nvidia-smi is not found
if which nvidia-smi ; then
    echo "Detected an NVIDIA GPU"
    export ACCELERATOR=gpu
    # Set CUDA_VISIBLE_DEVICES to a comma separated list of the GPU #'s you want to
    # use (index may start at 0).
    export CUDA_VISIBLE_DEVICES=0
else
    echo "No GPU detected, falling back to CPU"
    export ACCELERATOR=cpu
    export CUDA_VISIBLE_DEVICES=
fi

# However, if you have only have a cpu, the tutorial will still run.
# Comment out the above two commands and enable the command below.
# export ACCELERATOR=cpu

DEVICES=$(python -c "if 1:
    import os
    if os.environ.get('ACCELERATOR', '') == 'cpu':
        print('1')
    else:
        n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
        print(','.join(list(map(str, range(n)))) + ',')
")
STRATEGY=$(python -c "if 1:
    import os
    if os.environ.get('ACCELERATOR', '') == 'cpu':
        print('auto')
    else:
        n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
        print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "
CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
ACCELERATOR = $ACCELERATOR
DEVICES = $DEVICES
STRATEGY = $STRATEGY
DDP_WORKAROUND = $DDP_WORKAROUND
"

# The following parameters are related and impose constraints on each other.
MAX_STEPS=400
MAX_EPOCHS=20
TRAIN_BATCHES_PER_EPOCH=20
ACCUMULATE_GRAD_BATCHES=1
BATCH_SIZE=2
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")
# Thus we expose this recommendation script to suggest changes to that satisfy
# the constraints. (Its ok if it isnt perfect)
python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"


# Now that we have important variables defined in the environment we can run
# the actual fit command.
DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
    data:
        num_workers          : 4
        train_dataset        : $TRAIN_FPATH
        vali_dataset         : $VALI_FPATH
        channels             : '$CHANNELS'
        time_steps           : 5
        chip_dims            : 128
        batch_size           : $BATCH_SIZE
    model:
        class_path: MultimodalTransformer
        init_args:
            name        : $EXPERIMENT_NAME
            arch_name   : smt_it_stm_p8
            window_size : 8
            dropout     : 0.1
            global_saliency_weight: 1.0
            global_class_weight:    1.0
            global_change_weight:   0.0
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: $TARGET_LR
        weight_decay: $WEIGHT_DECAY
        betas:
          - 0.9
          - 0.99
    trainer:
      accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
      default_root_dir     : $DEFAULT_ROOT_DIR
      accelerator          : $ACCELERATOR
      devices              : $DEVICES
      strategy             : $STRATEGY
      limit_train_batches  : $TRAIN_BATCHES_PER_EPOCH
      check_val_every_n_epoch: 1
      enable_checkpointing: true
      enable_model_summary: true
      log_every_n_steps: 5
      logger: true
      max_steps: $MAX_STEPS
      num_sanity_val_steps: 0
    initializer:
        init: noop
"

# This is the default final package that is done at the end of training.
# See comments inside of demo_force_repackage
PACKAGE_FPATH="$DEFAULT_ROOT_DIR"/final_package.pt


demo_force_repackage(){
    # TODO: the new watch.mlops tool will need to handle this soon.

    # NOTE: The above fit script might not produce the "best" checkpoint as the
    # final output package. To evaluate a different checkpoint it must first be
    # packaged.  (note this can be done while training is running so
    # intermediate checkpoints can be evaluated while the model is still
    # learning). The following is logic for how to "package" a single checkpoint.
    #
    # Look at all checkpoints
    ls "$DEFAULT_ROOT_DIR"/*/*/checkpoints/*.ckpt
    # Grab the latest checkpoint (this is an arbitrary choice)
    CHECKPOINT_FPATH=$(find "$DEFAULT_ROOT_DIR" -iname "*.ckpt" | tail -n 1)
    echo "CHECKPOINT_FPATH = $CHECKPOINT_FPATH"
    # Repackage a particular checkpoint as a torch.package .pt file.
    python -m geowatch.tasks.fusion.repackage repackage "$CHECKPOINT_FPATH"
    # Redefine package fpath variable to be that checkpoint
    PACKAGE_FPATH=$(python -m geowatch.tasks.fusion.repackage repackage "$CHECKPOINT_FPATH" | tail -n 1)
    echo "PACKAGE_FPATH = $PACKAGE_FPATH"
}


# Define where we will output predictions / evaluations
# Later we will see how mlops.schedule_evaluation will do this for you, but in
# this tutorial we are going to run prediction and evaluation manually.
PRED_FPATH=$DEFAULT_ROOT_DIR/predictions/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/evaluation

# Predict using one of the packaged models
python -m geowatch.tasks.fusion.predict \
       --package_fpath="$PACKAGE_FPATH" \
        --test_dataset="$TEST_FPATH" \
        --pred_dataset="$PRED_FPATH" \

# Dump stats of truth vs prediction.
# We should see soft segmentation masks in pred, but not in truth
kwcoco stats "$TEST_FPATH" "$PRED_FPATH"
geowatch stats "$TEST_FPATH" "$PRED_FPATH"

# Visualize pixel predictions with a raw band, predicted saliency, and predicted class.
geowatch visualize "$PRED_FPATH" --channels='B11,salient,star|superstar|eff' --smart=True

# Evaluate the predictions
python -m geowatch.tasks.fusion.evaluate \
        --true_dataset="$TEST_FPATH" \
        --pred_dataset="$PRED_FPATH" \
          --eval_dpath="$EVAL_DPATH"
