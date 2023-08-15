# INV SHARED EXPERIMENT
# ~/code/watch/scripts/generate_ta2_features.sh
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_October2021
ARCH=smt_it_joint_p8
EXPERIMENT_NAME=Saliency_${ARCH}_raw_performer_s64_t13_hist_v005

KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json

CHANNELS="blue|green|red|nir|swir16|swir22"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 

# Write train and prediction configs
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=64 \
    --chip_overlap=0.0 \
    --time_steps=13 \
    --time_span=3y \
    --time_sampling=soft+distribute \
    --batch_size=8 \
    --accumulate_grad_batches=4 \
    --num_workers=16 \
    --attention_impl=performer \
    --neg_to_pos_ratio=1.0 \
    --global_class_weight=1.0 \
    --global_change_weight=1.0 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --saliency_loss='focal' \
    --class_loss='cce' \
    --normalize_inputs=2048 \
    --diff_inputs=False \
    --max_epochs=100 \
    --match_histograms=True \
    --patience=100 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --num_draw=8 \
    --dropout=0.1 \
    --window_size=8 \
    --eval_after_fit=True \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 




DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_October2021
ARCH=smt_it_joint_p8
EXPERIMENT_NAME=Saliency_${ARCH}_raw_performer_s64_t13_nohist_v006

KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json

CHANNELS="blue|green|red|nir|swir16|swir22"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 

# Write train and prediction configs
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=64 \
    --chip_overlap=0.0 \
    --time_steps=13 \
    --time_span=3y \
    --time_sampling=soft+distribute \
    --batch_size=8 \
    --accumulate_grad_batches=4 \
    --num_workers=16 \
    --attention_impl=performer \
    --neg_to_pos_ratio=1.0 \
    --global_class_weight=1.0 \
    --global_change_weight=1.0 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --saliency_loss='focal' \
    --class_loss='cce' \
    --normalize_inputs=2048 \
    --diff_inputs=False \
    --max_epochs=100 \
    --match_histograms=False \
    --patience=100 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --num_draw=8 \
    --dropout=0.1 \
    --window_size=8 \
    --eval_after_fit=True \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 
