#!/bin/bash
__doc__="""
This demonstrates an end-to-end pipeline on multispectral toydata

This walks through the entire process of fit -> predict -> evaluate and the
output if you run this should end with something like

source ~/code/watch/watch/tasks/fusion/experiments/crall/toy_experiments_msi.sh
"""

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
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_train${NUM_TOY_TRAIN_VIDS}" --verbose=4

    kwcoco toydata --key="vidshapes${NUM_TOY_VALI_VIDS}-frames5-randgsize-speed0.2-msi-multisensor" \
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_vali${NUM_TOY_VALI_VIDS}"  --verbose=4

    kwcoco toydata --key="vidshapes${NUM_TOY_TEST_VIDS}-frames6-randgsize-speed0.2-msi-multisensor" \
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_test${NUM_TOY_TEST_VIDS}" --verbose=4
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

__doc__="""
Should look like 
                                   dset  n_anns  n_imgs  n_videos  n_cats  r|g|b|disparity|gauss|B8|B11  B1|B8|B8a|B10|B11  r|g|b|flowx|flowy|distri|B10|B11
0  vidshapes_msi_train/data.kwcoco.json      80      40         8       3                            12                 12                                16
1   vidshapes_msi_vali/data.kwcoco.json      50      25         5       3                             9                 10                                 6
2   vidshapes_msi_test/data.kwcoco.json      24      12         2       3                             5                  3                                 4
"""



demo_visualize_toydata(){
    kwcoco toydata --key=vidshapes1-frames5-speed0.001-msi --bundle_dpath "$(realpath ./tmp)" --verbose=5 --use_cache=False
    python -m watch.cli.coco_visualize_videos \
        --src "$(realpath ./tmp/data.kwcoco.json)" \
        --channels="B1|B8|b" \
        --viz_dpath="$(realpath ./tmp)/_viz" \
        --animate=True

    python -m watch.cli.coco_visualize_videos \
        --src "$DVC_DATA_DPATH/vidshapes_msi_train100/data.kwcoco.json" \
        --channels="gauss|B11,r|g|b,B1|B8|B11" \
        --viz_dpath="$DVC_DATA_DPATH/vidshapes_msi_train100/_viz" --animate=True
}


#function join_by {
#    # https://stackoverflow.com/questions/1527049/how-can-i-join-elements-of-an-array-in-bash
#    local d=${1-} f=${2-}
#    if shift 2; then
#      printf %s "$f" "${@/#/$d}"
#    fi
#}
#STREAMS=(
#    "disparity|gauss"
#    "X.2|Y:2:6"
#    "B1|B8a"
#    "flowx|flowy|distri"
#)
#CHANNELS=$(join_by , "${STREAMS[@]}")


# Define the arch and channels we want to use
ARCH=smt_it_stm_p8
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


DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

# Configure training hyperparameters to a baseline config file
TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
python -m watch.tasks.fusion.fit \
    --channels="$CHANNELS" \
    --method=MultimodalTransformer \
    --arch_name=$ARCH \
    --window_size=8 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_steps=5 \
    --chip_size=256 \
    --batch_size=1 \
    --tokenizer=linconv \
    --global_saliency_weight=1.0 \
    --global_change_weight=1.0 \
    --global_class_weight=1.0 \
    --time_sampling=soft2 \
    --time_span=1y \
    --devices='auto' \
    --accelerator='auto' \
    --accumulate_grad_batches=1 \
    --dump="$TRAIN_CONFIG_FPATH"

# Fit 
# Specify the expected input / output files
EXPERIMENT_NAME=ToyFusion_${ARCH}_v001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
python -u -m watch.tasks.fusion.fit \
           --config="$TRAIN_CONFIG_FPATH" \
    --name="$EXPERIMENT_NAME" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
       --package_fpath="$PACKAGE_FPATH" \
        --train_dataset="$TRAIN_FPATH" \
         --vali_dataset="$VALI_FPATH" \
          --num_workers="2" || echo "Fit command failed with bad return code"


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
    python -m watch.tasks.fusion.repackage repackage "$CHECKPOINT_FPATH"
    # Redefine package fpath variable to be that checkpoint
    PACKAGE_FPATH=$(python -m watch.tasks.fusion.repackage repackage "$CHECKPOINT_FPATH" | tail -n 1)
    echo "PACKAGE_FPATH = $PACKAGE_FPATH"
}


# TODO: update to the watch.mlops version of this 
# The "suggest" tool will determine paths that will help keep experimental
# results organized and separated from one another. 
SUGGESTIONS=$(
    python -m watch.tasks.fusion.organize suggest_paths  \
        --package_fpath="$PACKAGE_FPATH"  \
        --test_dataset="$TEST_FPATH")
PRED_FPATH="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"
EVAL_DPATH="$(echo "$SUGGESTIONS" | jq -r .eval_dpath)"

# Predict using one of the packaged models
python -m watch.tasks.fusion.predict \
        --write_preds=True \
        --write_probs=True \
        --test_dataset="$TEST_FPATH" \
       --package_fpath="$PACKAGE_FPATH" \
        --pred_dataset="$PRED_FPATH" \
        --write_probs=True

# Dump stats of truth vs prediction.
# We should see soft segmentation masks in pred, but not in truth
python -m kwcoco stats "$TEST_FPATH" "$PRED_FPATH"
python -m watch stats "$TEST_FPATH" "$PRED_FPATH"

# Evaluate the predictions
python -m watch.tasks.fusion.evaluate \
        --true_dataset="$TEST_FPATH" \
        --pred_dataset="$PRED_FPATH" \
          --eval_dpath="$EVAL_DPATH"
