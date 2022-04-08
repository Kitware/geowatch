#!/bin/bash
__doc__="""
This demonstrates an end-to-end pipeline on multispectral toydata

This walks through the entire process of fit -> predict -> evaluate and the
output if you run this should end with something like
"""

# Generate toy datasets
TOY_DATA_DPATH=$HOME/data/work/toy_change
TRAIN_FPATH=$TOY_DATA_DPATH/vidshapes_msi_train100/data.kwcoco.json
VALI_FPATH=$TOY_DATA_DPATH/vidshapes_msi_vali/data.kwcoco.json
TEST_FPATH=$TOY_DATA_DPATH/vidshapes_msi_test/data.kwcoco.json 

generate_data(){
    mkdir -p "$TOY_DATA_DPATH"
    kwcoco toydata --key=vidshapes-videos100-frames5-randgsize-speed0.2-msi-multisensor --bundle_dpath "$TOY_DATA_DPATH/vidshapes_msi_train100" --verbose=1
    kwcoco toydata --key=vidshapes-videos5-frames5-randgsize-speed0.2-msi-multisensor --bundle_dpath "$TOY_DATA_DPATH/vidshapes_msi_vali"  --verbose=1
    kwcoco toydata --key=vidshapes-videos2-frames6-randgsize-speed0.2-msi-multisensor --bundle_dpath "$TOY_DATA_DPATH/vidshapes_msi_test" --verbose=1 
}


print_stats(){
    # Print stats
    python -m kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"
    python -m watch stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"
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
    kwcoco toydata --key=vidshapes-videos1-frames5-speed0.001-msi --bundle_dpath "$(realpath ./tmp)" --verbose=5 --use_cache=False
    python -m watch.cli.coco_visualize_videos \
        --src "$(realpath ./tmp/data.kwcoco.json)" \
        --channels="B1|B8|b" \
        --viz_dpath="$(realpath ./tmp)/_viz" \
        --animate=True

    python -m watch.cli.coco_visualize_videos \
        --src "$TOY_DATA_DPATH/vidshapes_msi_train100/data.kwcoco.json" \
        --channels="gauss|B11,r|g|b,B1|B8|B11" \
        --viz_dpath="$TOY_DATA_DPATH/vidshapes_msi_train100/_viz" --animate=True
}


function join_by {
  # https://stackoverflow.com/questions/1527049/how-can-i-join-elements-of-an-array-in-bash
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}


ARCH=smt_it_stm_p8

STREAMS=(
    "disparity|gauss"
    "X.2|Y:2:6"
    "B1|B8a"
    "flowx|flowy|distri"
)

#CHANNELS="disparity|gauss,,B1|B8a"
CHANNELS=$(join_by , "${STREAMS[@]}")
echo "CHANNELS = $CHANNELS"

EXPERIMENT_NAME=ToyFusion_${ARCH}_v001
DATASET_NAME=ToyDataMSI

# Place training inside of our DVC directory
DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_NAME/runs/$EXPERIMENT_NAME

# Specify the expected input / output files
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 

SUGGESTIONS=$(
    python -m watch.tasks.fusion.organize suggest_paths  \
        --package_fpath="$PACKAGE_FPATH"  \
        --test_dataset="$TEST_FPATH")

PRED_FPATH="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"
EVAL_DPATH="$(echo "$SUGGESTIONS" | jq -r .eval_dpath)"

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/predict_$EXPERIMENT_NAME.yml 

#export CUDA_VISIBLE_DEVICES="0"

# Configure training hyperparameters
python -m watch.tasks.fusion.fit \
    --name="$EXPERIMENT_NAME" \
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
    --tokenizer=dwcnn \
    --global_saliency_weight=1.0 \
    --global_change_weight=1.0 \
    --global_class_weight=1.0 \
    --time_sampling=soft2 \
    --time_span=1y \
    --gpus=1 \
    --accumulate_grad_batches=1 \
    --dump="$TRAIN_CONFIG_FPATH"

# Configure prediction hyperparams
python -m watch.tasks.fusion.predict \
    --gpus=1 \
    --write_preds=True \
    --write_probs=True \
    --dump="$PRED_CONFIG_FPATH"


# Fit 
python -m watch.tasks.fusion.fit \
           --config="$TRAIN_CONFIG_FPATH" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
       --package_fpath="$PACKAGE_FPATH" \
        --train_dataset="$TRAIN_FPATH" \
         --vali_dataset="$VALI_FPATH" \
         --test_dataset="$TEST_FPATH" \
          --num_workers="2"


demo_force_repackage(){
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

# Predict using one of the packaged models
python -m watch.tasks.fusion.predict \
        --config="$PRED_CONFIG_FPATH" \
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
