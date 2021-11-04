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
CHANNELS="blue|green|red|nir|swir16|swir22"
#CHANNELS="inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24|matseg_25|matseg_26|matseg_27|matseg_28|matseg_29|matseg_30|matseg_31|matseg_32|matseg_33|matseg_34|matseg_35|matseg_36|matseg_37|matseg_38|matseg_39|coastal|blue|green|red|nir|swir16|swir22|cirrus"


EXPERIMENT_NAME=Saliency_${ARCH}_toothbrush_s64_t3_v12
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
    --time_steps=3 \
    --time_span=1y \
    --diff_inputs=False \
    --match_histograms=False \
    --time_sampling=soft+distribute \
    --accumulate_grad_batches=1 \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=0.2 \
    --global_change_weight=1.0 \
    --global_saliency_weight=0.01 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --normalize_inputs=2048 \
    --max_epochs=100 \
    --patience=100 \
    --num_workers=16 \
    --gpus=1  \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 


packageup(){
    # This really should be automatic
    python ~/code/watch/watch/tasks/fusion/repackage.py /home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_toothbrush_s64_t3_v12/lightning_logs/version_1/checkpoints/epoch=12-step=3431.ckpt
    python ~/code/watch/watch/tasks/fusion/repackage.py /home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_toothbrush_s64_t3_v12/lightning_logs/version_1/checkpoints/epoch=17-step=4751.ckpt
    python ~/code/watch/watch/tasks/fusion/repackage.py /home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_toothbrush_s64_t3_v12/lightning_logs/version_1/checkpoints/epoch=49-step=13199.ckpt
}


EXPERIMENT_NAME=Saliency_${ARCH}_toothbrush_s96_t3_v13
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 

# Write train and prediction configs
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.0 \
    --time_steps=3 \
    --time_span=1y \
    --diff_inputs=False \
    --match_histograms=True \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=1.0 \
    --global_class_weight=0.02 \
    --global_change_weight=1.0 \
    --global_saliency_weight=0.01 \
    --negative_change_weight=0.05 \
    --change_loss='focal' \
    --saliency_loss='dicefocal' \
    --class_loss='cce' \
    --normalize_inputs=2048 \
    --max_epochs=200 \
    --patience=200 \
    --num_workers=16 \
    --gpus=1  \
    --batch_size=16 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 
