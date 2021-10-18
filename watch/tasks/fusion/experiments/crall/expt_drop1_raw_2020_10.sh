# RAW BASELINE EXPERIMENT
# ~/code/watch/scripts/generate_ta2_features.sh

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json


WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_October2021
CHANNELS="blue|green|red|nir|swir16|swir22"
ARCH=smt_it_joint_p8
EXPERIMENT_NAME=Saliency_${ARCH}_raw_v002
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
#TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
#PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 


#PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
#SUGGESTIONS="$(python -m watch.tasks.fusion.organize suggest_paths \
#    --package_fpath=$PACKAGE_FPATH \
#    --test_dataset=$TEST_DATASET)"
#PRED_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"
#EVAL_DATASET="$(echo "$SUGGESTIONS" | jq -r .eval_dpath)"

# Write train and prediction configs
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=64 \
    --chip_overlap=0.0 \
    --time_steps=3 \
    --time_span=1y \
    --time_sampling=hard+distribute \
    --batch_size=4 \
    --accumulate_grad_batches=4 \
    --num_workers=0 \
    --attention_impl=performer \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=0.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --saliency_loss='focal' \
    --class_loss='cce' \
    --diff_inputs=False \
    --max_epochs=200 \
    --patience=200 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --num_draw=6 \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 

