__doc__="""
This demonstrates an end-to-end pipeline on multispectral toydata

This walks through the entire process of fit -> predict -> evaluate and the
output if you run this should end with something like
"""

# Generate toy datasets
DATA_DPATH=$HOME/data/work/toy_change
TRAIN_FPATH=$DATA_DPATH/vidshapes_msi_train/data.kwcoco.json
VALI_FPATH=$DATA_DPATH/vidshapes_msi_vali/data.kwcoco.json
TEST_FPATH=$DATA_DPATH/vidshapes_msi_test/data.kwcoco.json 

mkdir -p $DATA_DPATH
kwcoco toydata vidshapes8-frames5-speed0.2-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_train
kwcoco toydata vidshapes4-frames5-speed0.2-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_vali
kwcoco toydata vidshapes2-frames6-speed0.2-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_test


# Print stats
python -m kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH
python -m watch watch_coco_stats $TRAIN_FPATH 


python -m watch.cli.coco_visualize_videos \
    --src $DATA_DPATH/vidshapes_msi_train/data.kwcoco.json \
    --channels="B1|B8|B11,B1" \
    --viz_dpath=$DATA_DPATH/vidshapes_msi_train/_viz

items=$(jq -r '.videos[] | .name' $DATA_DPATH/vidshapes_msi_train/data.kwcoco.json)
for item in ${items[@]}; do
    echo "item = $item"
    python -m watch.cli.gifify --frames_per_second 1.0 \
        --inputs "$DATA_DPATH/vidshapes_msi_train/_viz/${item}/_anns/B1" \
        --output "$DATA_DPATH/vidshapes_msi_train/_viz/${item}_ann.gif"

    python -m watch.cli.gifify --frames_per_second 1.0 \
        --inputs "$DATA_DPATH/vidshapes_msi_train/_viz/${item}/_imgs/B1|B8|B11" \
        --output "$DATA_DPATH/vidshapes_msi_train/_viz/${item}_img.gif"
done




ARCH=smt_it_stm_p8

CHANNELS="B8|B1|B11|B8a"

EXPERIMENT_NAME=ToyFusion_${ARCH}_v001
DATASET_NAME=ToyDataMSI

WORKDIR=$DATA_DPATH/training/$HOSTNAME/$USER
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_NAME/runs/$EXPERIMENT_NAME

# Specify the expected input / output files
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 

SUGGESTIONS="$(python -m watch.tasks.fusion.organize suggest_paths \
    --package_fpath=$PACKAGE_FPATH \
    --test_dataset=$TEST_FPATH)"
PRED_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"
EVAL_DATASET="$(echo "$SUGGESTIONS" | jq -r .eval_dpath)"

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/predict_$EXPERIMENT_NAME.yml 

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
    --time_steps=4 \
    --chip_size=64 \
    --batch_size=2 \
    --tokenizer=dwcnn \
    --global_saliency_weight=1.0 \
    --global_change_weight=1.0 \
    --global_class_weight=1.0 \
    --time_sampling=hard \
    --time_span=1y \
    --gpus=1 \
    --accumulate_grad_batches=1 \
    --dump=$TRAIN_CONFIG_FPATH

# Configure prediction hyperparams
python -m watch.tasks.fusion.predict \
    --gpus=1 \
    --write_preds=True \
    --write_probs=False \
    --dump=$PRED_CONFIG_FPATH


# Fit 
python -m watch.tasks.fusion.fit \
           --config=$TRAIN_CONFIG_FPATH \
      --num_workers=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH

# Predict 
python -m watch.tasks.fusion.predict \
        --config=$PRED_CONFIG_FPATH \
        --test_dataset=$TEST_FPATH \
       --package_fpath=$PACKAGE_FPATH \
        --pred_dataset=$PRED_FPATH

# Evaluate 
python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_FPATH \
        --pred_dataset=$PRED_FPATH \
          --eval_dpath=$EVAL_DPATH
