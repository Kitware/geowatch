
#### Training

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

ARCH=smt_it_joint_p8

CHANNELS="blue|green|red|nir|swir16|coastal"

EXPERIMENT_NAME=ActivityClf_${ARCH}_raw_v016
DATASET_CODE=Drop1_Raw_Holdout

DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 


# Write train and prediction configs
CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --chip_size=64 \
    --batch_size=6 \
    --accumulate_grad_batches=10 \
    --num_workers=14 \
    --time_dilation=sample \
    --attention_impl=performer \
    --global_class_weight=1.0 \
    --neg_to_pos_ratio=0.5 \
    --global_change_weight=0.0 \
    --negative_change_weight=0.05 \
    --change_loss='cce' \
    --diff_inputs=False \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 

