# INV SHARED EXPERIMENT
# ~/code/watch/scripts/generate_ta2_features.sh
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
#ARCH=smt_it_joint_p8
ARCH=deit

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
echo "KWCOCO_BUNDLE_DPATH = $KWCOCO_BUNDLE_DPATH"

kwcoco subset --src $KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json \
        --dst $KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json \
        --select_images '.sensor_coarse != "WV"'

kwcoco subset --src $KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json \
        --dst $KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json \
        --select_images '.sensor_coarse != "WV"'

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json


prep_and_inspect(){
    python -m watch.cli.coco_visualize_videos \
       --src $KWCOCO_BUNDLE_DPATH/prop_data.kwcoco.json \
       --channels "red|green|blue" \
       --draw_imgs=False  --animate=True

    BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/prop_data.kwcoco.json
    VIZ_DPATH=$KWCOCO_BUNDLE_DPATH/_viz

    python -m watch stats $KWCOCO_BUNDLE_DPATH/prop_data.kwcoco.json


    kwcoco subset --src $BASE_COCO_FPATH \
            --dst $KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
            --select_images '.sensor_coarse != "WV"'

    # Split out train and validation data (TODO: add test when we can)
    kwcoco subset --src $KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
            --dst $VALI_FPATH \
            --select_videos '.name | startswith("KR_")'

    kwcoco subset --src $KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
            --dst $TRAIN_FPATH \
            --select_videos '.name | startswith("KR_") | not'
}


#python -m kwcoco stats $TRAIN_FPATH $VALI_FPATH
#python -m watch stats $TRAIN_FPATH
CHANNELS="red|green|blue|nir|swir16|swir22"
#CHANNELS="inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24|matseg_25|matseg_26|matseg_27|matseg_28|matseg_29|matseg_30|matseg_31|matseg_32|matseg_33|matseg_34|matseg_35|matseg_36|matseg_37|matseg_38|matseg_39|coastal|blue|green|red|nir|swir16|swir22|cirrus"


EXPERIMENT_NAME=Saliency_${ARCH}_toothbrush_prop_s64_t3_v12
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
    --global_class_weight=0.05 \
    --global_change_weight=0.001 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --change_loss='focal' \
    --saliency_loss='dicefocal' \
    --class_loss='focal' \
    --normalize_inputs=512 \
    --max_epochs=200 \
    --patience=200 \
    --num_workers=16 \
    --gpus=1  \
    --batch_size=16 \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --eval_after_fit=True \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 



##########################
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=deit
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
CHANNELS="red|green|blue|nir|swir16|swir22"
EXPERIMENT_NAME=Saliency_${ARCH}_toothbrush_prop_s64_t3_v14
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
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
    --optimizer=AdamW \
    --match_histograms=False \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=0.000002 \
    --global_change_weight=0.000001 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --saliency_loss='dicefocal' \
    --class_loss='focal' \
    --normalize_inputs=1024 \
    --max_epochs=140 \
    --patience=140 \
    --num_workers=8 \
    --gpus=1  \
    --batch_size=16 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --eval_after_fit=True \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 


##########################
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=deit
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
CHANNELS="red|green|blue|nir|swir16|swir22"
EXPERIMENT_NAME=Saliency_${ARCH}_toothbrush_prop_s64_t3_v14-cont
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
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
    --optimizer=AdamW \
    --match_histograms=False \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=2.0 \
    --global_class_weight=0.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --saliency_loss='focal' \
    --class_loss='focal' \
    --normalize_inputs=1024 \
    --max_epochs=140 \
    --patience=140 \
    --num_workers=5 \
    --gpus=1  \
    --batch_size=16 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --eval_after_fit=True \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --init=/home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1_November2021/runs/Saliency_deit_toothbrush_prop_s64_t3_v14/lightning_logs/version_0/package-interupt/package_epoch89_step66181.pt \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 



##########################
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}

prep_wv_expt(){

    cd $KWCOCO_BUNDLE_DPATH
    python -m kwcoco stats train_data_nowv.kwcoco.json vali_data_nowv.kwcoco.json train_data_wv.kwcoco.json vali_data_wv.kwcoco.json

    kwcoco subset --src $KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json \
            --dst $KWCOCO_BUNDLE_DPATH/train_data_wv.kwcoco.json \
            --select_images '.sensor_coarse == "WV"'

    kwcoco subset --src $KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json \
            --dst $KWCOCO_BUNDLE_DPATH/vali_data_wv.kwcoco.json \
            --select_images '.sensor_coarse == "WV"'

    python -m watch visualize --src $KWCOCO_BUNDLE_DPATH/train_data_wv.kwcoco.json --viz_dpath $KWCOCO_BUNDLE_DPATH/_train_wv_viz
    python -m watch visualize --src $KWCOCO_BUNDLE_DPATH/vali_data_wv.kwcoco.json --viz_dpath $KWCOCO_BUNDLE_DPATH/_vali_wv_viz
}

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_wv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_wv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_wv.kwcoco.json
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=deit
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
CHANNELS="red|green|blue"
EXPERIMENT_NAME=Saliency_${ARCH}_toothbrush_wv_s64_t3_v15
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
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
    --optimizer=AdamW \
    --match_histograms=False \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=2.0 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=0.0001 \
    --negative_change_weight=0.05 \
    --change_loss='focal' \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --normalize_inputs=1024 \
    --max_epochs=140 \
    --patience=140 \
    --num_workers=5 \
    --gpus=1  \
    --batch_size=16 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --eval_after_fit=True \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 


##########################
#
# NAMEK 
#
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_combo11.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_combo11.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_combo11.kwcoco.json
#python -m watch stats $VALI_FPATH
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=deit
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
CHANNELS="blue|green|red|nir|cirrus|swir16|swir22|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24|matseg_25|matseg_26|matseg_27|matseg_28|matseg_29|matseg_30|matseg_31|matseg_32|matseg_33|matseg_34|matseg_35|matseg_36|matseg_37|matseg_38|matseg_39|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
EXPERIMENT_NAME=Activity_${ARCH}_newanns_rutgers_dzyne_v1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=64 \
    --chip_overlap=0.0 \
    --time_steps=3 \
    --time_span=1y \
    --diff_inputs=False \
    --optimizer=Adam \
    --match_histograms=False \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=2.0 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=0.0001 \
    --negative_change_weight=0.05 \
    --change_loss='focal' \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --normalize_inputs=1024 \
    --max_epochs=140 \
    --patience=140 \
    --num_workers=16 \
    --gpus=1  \
    --batch_size=4 \
    --accumulate_grad_batches=4 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --eval_after_fit=True \
    --window_size=4 \
    --window_size=4 \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  \
         --dump $WORKDIR/configs/namek_common_activity11.yaml

# NAMEK Try1
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
ARCH=deit
CHANNELS="blue|green|red|nir|cirrus|swir16|swir22|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24|matseg_25|matseg_26|matseg_27|matseg_28|matseg_29|matseg_30|matseg_31|matseg_32|matseg_33|matseg_34|matseg_35|matseg_36|matseg_37|matseg_38|matseg_39|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
DATASET_CODE=Drop1_November2021
EXPERIMENT_NAME=Activity_${ARCH}_newanns_rutgers_dzyne_v1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --config $WORKDIR/configs/namek_common_activity11.yaml 


# NAMEK Try2
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
ARCH=smt_it_joint_s12
CHANNELS="blue|green|red|nir|cirrus|swir16|swir22|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24|matseg_25|matseg_26|matseg_27|matseg_28|matseg_29|matseg_30|matseg_31|matseg_32|matseg_33|matseg_34|matseg_35|matseg_36|matseg_37|matseg_38|matseg_39|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
DATASET_CODE=Drop1_November2021
EXPERIMENT_NAME=Activity_${ARCH}_newanns_rutgers_dzyne_v2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --chip_size=64 \
    --time_steps=5 \
    --global_saliency_weight=0.0 \
    --arch_name=$ARCH \
    --optimizer=RAdam \
    --config $WORKDIR/configs/namek_common_activity11.yaml 



##########################
#
# Toothbrush Try3
#
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
#python -m watch stats $VALI_FPATH
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=deit
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
CHANNELS="blue|green|red|nir|cirrus|swir16|swir22"
EXPERIMENT_NAME=Activity_${ARCH}_newanns_rgb_v3
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
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
    --optimizer=AdamW \
    --match_histograms=False \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=2.0 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=0.0001 \
    --negative_change_weight=0.05 \
    --change_loss='focal' \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --normalize_inputs=1024 \
    --max_epochs=140 \
    --patience=140 \
    --num_workers=6 \
    --gpus=1  \
    --batch_size=16 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --eval_after_fit=True \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 


# Toothbrush Try4
#
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
#python -m watch stats $VALI_FPATH
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=smt_it_joint_m24
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
CHANNELS="blue|green|red|nir|cirrus|swir16|swir22"
EXPERIMENT_NAME=Activity_${ARCH}_newanns_rgb_v4
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.0 \
    --time_steps=5 \
    --time_span=1y \
    --diff_inputs=False \
    --optimizer=AdamW \
    --match_histograms=False \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=2.0 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=0.0001 \
    --negative_change_weight=0.05 \
    --change_loss='focal' \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --normalize_inputs=1024 \
    --max_epochs=140 \
    --patience=140 \
    --num_workers=6 \
    --gpus=1  \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --eval_after_fit=True \
    --window_size=4 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 



####----
# Horologic Try5
#
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_combo11.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_combo11.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_combo11.kwcoco.json
#python -m watch stats $VALI_FPATH
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=smt_it_joint_p8
CHANNELS="blue|green|red|nir|cirrus|swir16|swir22"
EXPERIMENT_NAME=ActivityTemplate
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=64 \
    --chip_overlap=0.0 \
    --time_steps=3 \
    --time_span=1y \
    --diff_inputs=False \
    --optimizer=AdamW \
    --match_histograms=False \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=2.0 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=0.0001 \
    --negative_change_weight=0.05 \
    --change_loss='focal' \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --normalize_inputs=1024 \
    --max_epochs=140 \
    --patience=140 \
    --num_workers=6 \
    --gpus=1  \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --eval_after_fit=True \
    --window_size=4 \
    --num_draw=8 \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  
         --dump $WORKDIR/configs/horologic_common_activity11.yaml


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=smt_it_joint_s24
CHANNELS="matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24|matseg_25|matseg_26|matseg_27|matseg_28|matseg_29|matseg_30|matseg_31|matseg_32|matseg_33|matseg_34|matseg_35|matseg_36|matseg_37|matseg_38|matseg_39|blue|green|red|nir|cirrus|swir16|swir22"
EXPERIMENT_NAME=Activity_${ARCH}_newanns_rutgers_v5
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/horologic_common_activity11.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=smt_it_joint_s24
CHANNELS="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|blue|green|red|nir|cirrus|swir16|swir22"
EXPERIMENT_NAME=Activity_${ARCH}_newanns_dzyne_v6
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/horologic_common_activity11.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=smt_it_joint_s24
CHANNELS="brush|bare_ground|built_up|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|blue|green|red|nir|cirrus|swir16|swir22"
EXPERIMENT_NAME=Activity_${ARCH}_newanns_hybrid_v7
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/horologic_common_activity11.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=11 \
    --batch_size=1 \
    --optimizer=AdamW \
    --accumulate_grad_batches=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=smt_it_st_s12
CHANNELS="brush|bare_ground|built_up|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|blue|green|red|nir|cirrus|swir16|swir22"
EXPERIMENT_NAME=Activity_${ARCH}_newanns_hybrid_v8
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/horologic_common_activity11.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=17 \
    --neg_to_pos_ratio=1.0 \
    --batch_size=1 \
    --optimizer=RAdam \
    --accumulate_grad_batches=16 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH 
