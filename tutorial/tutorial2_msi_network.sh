#!/bin/bash
__doc__="
This demonstrates an end-to-end pipeline on multispectral toydata

This walks through the entire process of fit -> predict -> evaluate and the
output if you run this should end with something like

source ~/code/watch/tutorial/toy_experiments_msi.sh
"

# Define wherever you want to store results
DVC_DATA_DPATH=$HOME/data/dvc-repos/toy_data_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/toy_expt_dvc

NUM_TOY_TRAIN_VIDS="${NUM_TOY_TRAIN_VIDS:-100}"  # If variable not set or null, use default.
NUM_TOY_VALI_VIDS="${NUM_TOY_VALI_VIDS:-5}"  # If variable not set or null, use default.
NUM_TOY_TEST_VIDS="${NUM_TOY_TEST_VIDS:-2}"  # If variable not set or null, use default.

# Generate toy datasets
TRAIN_FPATH=$DVC_DATA_DPATH/vidshapes_msi_train${NUM_TOY_TRAIN_VIDS}/data.kwcoco.json
VALI_FPATH=$DVC_DATA_DPATH/vidshapes_msi_vali${NUM_TOY_VALI_VIDS}/data.kwcoco.json
TEST_FPATH=$DVC_DATA_DPATH/vidshapes_msi_test${NUM_TOY_TEST_VIDS}/data.kwcoco.json

generate_data(){
    mkdir -p "$DVC_DATA_DPATH"

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
    smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"
}

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
    kwcoco toydata --key=vidshapes1-frames5-speed0.001-msi --bundle_dpath "$(realpath ./tmp)" --verbose=5 --use_cache=False
    python -m geowatch.cli.coco_visualize_videos \
        --src "$(realpath ./tmp/data.kwcoco.json)" \
        --channels="B1|B8|b" \
        --viz_dpath="$(realpath ./tmp)/_viz" \
        --animate=True

    python -m geowatch.cli.coco_visualize_videos \
        --src "$DVC_DATA_DPATH/vidshapes_msi_train100/data.kwcoco.json" \
        --channels="gauss|B11,r|g|b,B1|B8|B11" \
        --viz_dpath="$DVC_DATA_DPATH/vidshapes_msi_train100/_viz" --animate=True
}


# Define the channels we want to use
# The sensors and channels are specified by the kwcoco SensorChanSpec
# in this example the data does not contain sensor metadata, so we
# use a "*" to indicate a generic sensor.
# A colon ":" separates channels from the sensors.
# A pipe "|" indicates channels are early fused
# A "," indicates groups of early fused channels are late fused Multiple
# sensors can be specified to the left of the channels and will distribute over
# commas.
CHANNELS="(*):(disparity|gauss,X.2|Y:2:6,B1|B8a,flowx|flowy|distri)"
echo "CHANNELS = $CHANNELS"


# Fit
DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyDataMSI_Demo_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
MAX_STEPS=80000
TARGET_LR=3e-4
DDP_WORKAROUND=1 python -m geowatch.tasks.fusion fit --config "
    data:
        num_workers          : 4
        train_dataset        : $TRAIN_FPATH
        vali_dataset         : $VALI_FPATH
        channels             : '$CHANNELS'
        time_steps           : 5
        chip_dims            : 128
        batch_size           : 2
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
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.OneCycleLR
      init_args:
        max_lr: $TARGET_LR
        total_steps: $MAX_STEPS
        anneal_strategy: cos
        pct_start: 0.05
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: $TARGET_LR
        weight_decay: 1e-5
        betas:
          - 0.9
          - 0.99
    trainer:
      accumulate_grad_batches: 1
      default_root_dir     : $DEFAULT_ROOT_DIR
      accelerator          : gpu
      devices              : 0,1
      strategy             : ddp
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
smartwatch stats "$TEST_FPATH" "$PRED_FPATH"

# Visualize pixel predictions with a raw band, predicted saliency, and predicted class.
smartwatch visualize "$PRED_FPATH" --channels='B11,salient,star|superstar|eff' --smart=True

# Evaluate the predictions
python -m geowatch.tasks.fusion.evaluate \
        --true_dataset="$TEST_FPATH" \
        --pred_dataset="$PRED_FPATH" \
          --eval_dpath="$EVAL_DPATH"
