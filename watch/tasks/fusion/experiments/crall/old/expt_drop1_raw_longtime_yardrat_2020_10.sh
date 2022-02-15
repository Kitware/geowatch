# INV SHARED EXPERIMENT
# ~/code/watch/scripts/generate_ta2_features.sh
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_October2021
ARCH=smt_it_joint_p8

KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}

#kwcoco subset --src $KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json \
#        --dst $KWCOCO_BUNDLE_DPATH/combo_train_s2_data.kwcoco.json \
#        --select_images '.sensor_coarse == "S2"'

#kwcoco subset --src $KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json \
#        --dst $KWCOCO_BUNDLE_DPATH/combo_vali_s2_data.kwcoco.json \
#        --select_images '.sensor_coarse == "S2"'

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json

python -m watch stats $TRAIN_FPATH
#CHANNELS="blue|green|red|nir|swir16|swir22"

CHANNELS="matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24|matseg_25|matseg_26|matseg_27|matseg_28|matseg_29|matseg_30|matseg_31|matseg_32|matseg_33|matseg_34|matseg_35|matseg_36|matseg_37|matseg_38|matseg_39|blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Saliency_${ARCH}_base_rutgers_s64_t5_v005
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
    --time_steps=5 \
    --time_span=2y \
    --time_sampling=soft+distribute \
    --batch_size=4 \
    --accumulate_grad_batches=8 \
    --num_workers=avail \
    --squash_modes=True \
    --attention_impl=exact \
    --neg_to_pos_ratio=0.3 \
    --global_class_weight=0.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --saliency_loss='dicefocal' \
    --class_loss='cce' \
    --normalize_inputs=2048 \
    --diff_inputs=False \
    --max_epochs=100 \
    --eval_after_fit=True \
    --match_histograms=False \
    --patience=100 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --num_draw=8 \
    --dropout=0.1 \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 



EXPERIMENT_NAME=Saliency_${ARCH}_base_rutgers_s128_t5_v006
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=128 \
    --chip_overlap=0.0 \
    --time_steps=5 \
    --time_span=2y \
    --time_sampling=soft+distribute \
    --batch_size=1 \
    --init="$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_base_rutgers_s64_t5_v005/lightning_logs/version_7/packages/package_epoch99_step22900.pt" \
    --accumulate_grad_batches=8 \
    --num_workers=avail \
    --squash_modes=True \
    --attention_impl=exact \
    --neg_to_pos_ratio=0.3 \
    --global_class_weight=0.05 \
    --global_change_weight=0.01 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --saliency_loss='dicefocal' \
    --change_loss='cce' \
    --class_loss='cce' \
    --normalize_inputs=2048 \
    --diff_inputs=True \
    --max_epochs=200 \
    --eval_after_fit=True \
    --match_histograms=False \
    --patience=200 \
    --gpus=1  \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --num_draw=8 \
    --dropout=0.1 \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 
