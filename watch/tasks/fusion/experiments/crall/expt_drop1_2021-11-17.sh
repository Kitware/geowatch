hack_rsync_dataset(){
    rsync -avprLR horologic:data/dvc-repos/smart_watch_dvc/./Drop1-Aligned-L1  $HOME/data/dvc-repos/smart_watch_dvc
}

#Activity_smt_it_joint_m24_newanns_rgb_v4_epoch
prep_and_inspect(){
    __doc__="
    SeeAlso:
        ~/code/watch/scripts/prepare_drop1_level1.sh
    "
    python -m watch.cli.coco_visualize_videos \
       --src $KWCOCO_BUNDLE_DPATH/prop_data.kwcoco.json \
       --channels "red|green|blue" \
       --draw_imgs=False  --animate=True

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1

    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-TA1-2022-01
    VIZ_DPATH=$KWCOCO_BUNDLE_DPATH/_viz
    TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_nowv.kwcoco.json
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_nowv.kwcoco.json
    TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_nowv.kwcoco.json


    kwcoco subset --src $KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --dst $KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
            --select_images '.sensor_coarse != "WV"'
    RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth"
    DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH/models/landcover/visnav_remap_s2_subset.pt"

    export CUDA_VISIBLE_DEVICES="1"
    python -m watch.tasks.rutgers_material_seg.predict \
        --test_dataset=$KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
        --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
        --default_config_key=iarpa \
        --pred_dataset=$KWCOCO_BUNDLE_DPATH/data_nowv_rutgers_mat_seg.kwcoco.json \
        --num_workers="16" \
        --batch_size=4 --gpus "1" \
        --compress=RAW --blocksize=64

    export CUDA_VISIBLE_DEVICES="1"
    python -m watch.tasks.landcover.predict \
        --dataset=$KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
        --deployed=$DZYNE_LANDCOVER_MODEL_FPATH  \
        --device=0 \
        --num_workers="16" \
        --output=$KWCOCO_BUNDLE_DPATH/data_nowv_dzyne_landcover.kwcoco.json

    python ~/code/watch/watch/cli/coco_combine_features.py \
        --src $KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
              $KWCOCO_BUNDLE_DPATH/data_nowv_rutgers_mat_seg.kwcoco.json \
              $KWCOCO_BUNDLE_DPATH/data_nowv_dzyne_landcover.kwcoco.json \
        --dst $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json

    python -m watch project \
        --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
        --src $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json \
        --dst $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json

    smartwatch stats $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json

    # Split out train and validation data (TODO: add test when we can)
    kwcoco subset --src $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json \
            --dst $VALI_FPATH \
            --select_videos '.name | startswith("KR_")'

    kwcoco subset --src $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json \
            --dst $TRAIN_FPATH \
            --select_videos '.name | startswith("KR_") | not'

    kwcoco stats $TRAIN_FPATH $VALI_FPATH
}


####----
# Common Root - 2021-11-17
#
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_nowv.kwcoco.json
#python -m watch stats $VALI_FPATH
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1_November2021
ARCH=smt_it_joint_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
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
    --chip_overlap=0.0 \
    --chip_size=64 \
    --time_steps=3 \
    --time_span=1y \
    --diff_inputs=False \
    --optimizer=AdamW \
    --match_histograms=False \
    --normalize_perframe=False \
    --time_sampling=soft+distribute \
    --attention_impl=exact \
    --squash_modes=True \
    --neg_to_pos_ratio=0.25 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=0.00 \
    --negative_change_weight=0.05 \
    --change_loss='focal' \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --normalize_inputs=1024 \
    --max_epochs=140 \
    --patience=140 \
    --num_workers=4 \
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
         --num_sanity_val_steps=0 \
         --dump $WORKDIR/configs/common_20201117.yaml


####----
# Horologic - Try6
# 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_st_s12
CHANNELS="brush|bare_ground|built_up|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_hybrid_v20
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=deit
CHANNELS="brush|bare_ground|built_up|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_hybrid_v21
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_st_s12
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_rgb_v22
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=deit
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_rgb_v23
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH 


####----
# Toothbrush - Try6
# 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_st_s12
CHANNELS="brush|bare_ground|built_up|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_pfnorm_hybrid_v24
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --normalize_perframe=True \
    --num_workers=10 \
    --arch_name=$ARCH 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_st_s12
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_pfnorm_rgb_v25
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --normalize_perframe=True \
    --arch_name=$ARCH 


####----
# Extension Horologic - 2021-12-03
#


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_rgb_v26
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="brush|bare_ground|built_up|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_hybrid_v27
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 

# Extension Horologic - 2021-12-11
#


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_cs96_t3_perframe_rgb_v32
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=96 \
    --time_steps=3 \
    --normalize_perframe=True \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="brush|bare_ground|built_up|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_cs96_t3_hybrid_v33
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=96 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --normalize_perframe=True \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_cs64_t5_perframe_rgb_v34
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=5 \
    --normalize_perframe=True \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="brush|bare_ground|built_up|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_cs64_t5_perframe_hybrid_v35
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=5 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --normalize_perframe=True \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 


#
# Followup Namek - 2021-12-13

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_rgb_v36
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --normalize_inputs=False \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 

# Followup Yardrat - 2021-12-13
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_rgb_v37
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --normalize_inputs=False \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 



#
# New True-MultiModal Horologic - 2021-12-26

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_raw_v38
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=3 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_raw_v39
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=11 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_materials24_v40
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=11 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 



DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6,blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_newanns_weighted_mat6raw6_v41
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=15 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH 

#
# TA1 With Positive Centers Horologic - 2022-01-10

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/raw_train_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/raw_vali_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/raw_vali_nowv.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

# Split out train and validation data (TODO: add test when we can)
kwcoco subset --src $KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
        --dst "$VALI_FPATH" \
        --select_videos '.name | startswith("KR_")'

kwcoco subset --src $KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
        --dst "$TRAIN_FPATH" \
        --select_videos '.name | startswith("KR_") | not'

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=SC_${ARCH}_centerannot_raw_v42
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=11 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --use_grid_positives=False \
    --use_centered_positives=True \
    --arch_name=$ARCH 


# L1 With Invariants + Positive Horologic - 2022-01-11
# Invariants got broke
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_nowv_invariants.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_nowv_invariants.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_nowv_invariants.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

# Split out train and validation data (TODO: add test when we can)
kwcoco subset --src "$KWCOCO_BUNDLE_DPATH/invariants_nowv.kwcoco.json" \
        --dst "$VALI_FPATH" \
        --select_videos '.name | startswith("KR_")'

kwcoco subset --src "$KWCOCO_BUNDLE_DPATH/invariants_nowv.kwcoco.json" \
        --dst "$TRAIN_FPATH" \
        --select_videos '.name | startswith("KR_") | not'

smartwatch stats "$TRAIN_FPATH"

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,before_after_heatmap,segmentation_heatmap"
EXPERIMENT_NAME=SC_${ARCH}_centerannot_uky2_v43
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=7 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --use_grid_positives=False \
    --use_centered_positives=True \
    --arch_name=$ARCH 


# L1 With Invariants + Positive Horologic - 2022-01-11
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,invariants:6,before_after_heatmap,segmentation_heatmap"
EXPERIMENT_NAME=SC_${ARCH}_centerannot_raw_v44
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=7 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --use_grid_positives=False \
    --use_centered_positives=True \
    --arch_name=$ARCH 


# L1 With Invariants + DZYNE + Positive Horologic - 2022-01-17
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_nowv_du_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_nowv_du_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_nowv_du_data.kwcoco.json

_prep_feats_for_2022_01_17(){
    # Rutgers mats did not finish, so combine what we have
    python -m watch.cli.coco_combine_features \
        --src data.kwcoco.json \
              uky_invariants.kwcoco.json \
              dzyne_landcover.kwcoco.json \
        --dst combo_du_data.kwcoco.json

    kwcoco subset --src "$KWCOCO_BUNDLE_DPATH/combo_du_data.kwcoco.json" \
            --dst "vali_nowv_du_data.kwcoco.json" \
            --select_images '.sensor_coarse != "WV"' \
            --select_videos '.name | startswith("KR_")'

    kwcoco subset --src "$KWCOCO_BUNDLE_DPATH/combo_du_data.kwcoco.json" \
            --dst "train_nowv_du_data.kwcoco.json" \
            --select_images '.sensor_coarse != "WV"' \
            --select_videos '.name | startswith("KR_") | not'

    smartwatch stats train_nowv_du_data.kwcoco.json vali_nowv_du_data.kwcoco.json
    kwcoco stats train_nowv_du_data.kwcoco.json vali_nowv_du_data.kwcoco.json
}


WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,invariants:6,before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
EXPERIMENT_NAME=SC_${ARCH}_centerannot_du_v45
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml" \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=7 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --use_grid_positives=False \
    --attention_impl=exact \
    --use_centered_positives=True \
    --arch_name=$ARCH 


# L1 With All Features + Positive Toothbrush - 2022-01-11
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,invariants:6,before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
EXPERIMENT_NAME=SC_${ARCH}_centerannot_du_v45
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=7 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --use_grid_positives=False \
    --attention_impl=exact \
    --use_centered_positives=True \
    --num_workers=avail \
    --arch_name=$ARCH 


# L1 With All Features + Positive Toothbrush Conv7 - 2022-01-18
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,invariants:6,before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|matseg_20|matseg_21|matseg_22|matseg_23|matseg_24|matseg_25|matseg_26|matseg_27|matseg_28|matseg_29|matseg_30|matseg_31|matseg_32|matseg_33|matseg_34|matseg_35|matseg_36|matseg_37|matseg_38|matseg_39"

smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=SC_${ARCH}_centerannot_IL_v47
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=7 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --use_grid_positives=False \
    --attention_impl=exact \
    --tokenizer=conv7 \
    --use_centered_positives=True \
    --num_workers=avail \
    --arch_name=$ARCH 


# L1 BAS with raw features Namek - 2022-01-19
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=BAS_${ARCH}_L1_raw_v48
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=96 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=rearrange \
    --use_grid_positives=True \
    --use_centered_positives=True \
    --neg_to_pos_ratio=0.25 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.0 \
    --num_workers=avail/2 \
    --arch_name=$ARCH


# TA1 BAS with raw features Namek - 2022-01-19
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=BAS_${ARCH}_TA1_raw_v49
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=96 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=rearrange \
    --use_grid_positives=True \
    --use_centered_positives=True \
    --neg_to_pos_ratio=0.25 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.0 \
    --num_workers=avail/2 \
    --arch_name=$ARCH


# BAS+SC L1 With Many Features + Positive Toothbrush LinConv - 2022-01-19
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15"

smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BOTH_${ARCH}_centerannot_ILM_v50
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=5 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --global_saliency_weight=1.00 \
    --global_class_weight=1.00 \
    --arch_name=$ARCH


# BAS+SC L1 With Many Features + Positive Toothbrush LinConv - 2022-01-19
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"

smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BOTH_${ARCH}_L1_IL_v51
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=5 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --global_saliency_weight=1.00 \
    --global_class_weight=1.00 \
    --time_sampling=soft2 \
    --batch_size=8 \
    --arch_name=$ARCH


# BAS+SC WV+L1 With Many Features + Positive Toothbrush LinConv - 2022-01-19
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,depth,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"

smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BOTH_${ARCH}_L1_DIL_v52
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=64 \
    --time_steps=5 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --global_saliency_weight=1.00 \
    --global_class_weight=1.00 \
    --time_sampling=soft2 \
    --batch_size=16 \
    --arch_name=$ARCH


# L1 BAS with raw features Namek - 2022-01-21
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=BAS_${ARCH}_L1_raw_v53
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=224 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=rearrange \
    --use_grid_positives=True \
    --use_centered_positives=True \
    --neg_to_pos_ratio=0.25 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.0 \
    --time_sampling=hardish \
    --num_workers=avail/2 \
    --arch_name=$ARCH

# TA1 BAS with raw features Namek - 2022-01-21
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=BAS_${ARCH}_TA1_raw_v54
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=224 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=rearrange \
    --use_grid_positives=True \
    --use_centered_positives=True \
    --neg_to_pos_ratio=0.25 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.0 \
    --time_sampling=hardish \
    --num_workers=avail/2 \
    --arch_name=$ARCH


# Transfer BAS+SC WV+L1 With Many Features + Positive Toothbrush LinConv - 2022-01-21
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,depth,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"

smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BOTH_${ARCH}_L1_DIL_v55
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=160 \
    --time_steps=9 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --time_span=1y \
    --global_saliency_weight=1.00 \
    --global_class_weight=1.00 \
    --time_sampling=hardish \
    --batch_size=4 \
    --arch_name=$ARCH \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1-20201117/runs/BOTH_smt_it_stm_p8_L1_DIL_v52/lightning_logs/version_0/checkpoints/epoch=13-step=55215-c.ckpt"



# Transfer BAS+SC WV+L1 With Many Features + Positive Toothbrush LinConv - 2022-01-21
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,depth,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
#smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BOTH_${ARCH}_L1_DIL_v56
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=464 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --time_span=1y \
    --global_saliency_weight=1.00 \
    --global_class_weight=1.00 \
    --time_sampling=hardish \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1-20201117/runs/BOTH_smt_it_stm_p8_L1_DIL_v52/lightning_logs/version_0/checkpoints/epoch=13-step=55215-c.ckpt"




# Transfer BOTH model to SC only Many Features + Positive Toothbrush LinConv - 2022-01-25
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,depth,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
#smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BAS_${ARCH}_TUNE_L1_DIL_v57
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=432 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --time_span=1y \
    --global_saliency_weight=1.00 \
    --global_class_weight=0.00 \
    --time_sampling=soft2 \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


# Transfer BOTH model to SC only RAW + Positive Toothbrush LinConv - 2022-01-25
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
#smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BAS_${ARCH}_TUNE_L1_RAW_v58
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=432 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --time_span=1y \
    --global_saliency_weight=1.00 \
    --global_class_weight=0.00 \
    --time_sampling=soft2 \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


# Continue fine tuning of BOTH model to SC only Many Features + Positive Toothbrush LinConv - 2022-01-26
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,depth,before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
#smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BAS_${ARCH}_TUNE_L1_I2L8_v59
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=432 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --time_span=1y \
    --global_saliency_weight=1.00 \
    --global_class_weight=0.00 \
    --time_sampling=soft2 \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BAS_smt_it_stm_p8_TUNE_L1_DIL_v57/BAS_smt_it_stm_p8_TUNE_L1_DIL_v57_epoch=3-step=81135.pt"


# Continue fine tuning of BOTH model to SC only Many Features + Positive Toothbrush LinConv - 2022-01-26
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22,invariants:6|before_after_heatmap|segmentation_heatmap,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
#smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BAS_${ARCH}_TUNE_L1_I8L8_v60
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=432 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="avail/2" \
    --neg_to_pos_ratio=1.0 \
    --time_span=1y \
    --global_saliency_weight=1.00 \
    --global_class_weight=0.00 \
    --time_sampling=soft2 \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BAS_smt_it_stm_p8_TUNE_L1_DIL_v57/BAS_smt_it_stm_p8_TUNE_L1_DIL_v57_epoch=3-step=81135.pt"


# Transfer BAS+SC WV+L1 With Few Features Toothbrush LinConv - 2022-01-27
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
#CHANNELS="depth,before_after_heatmap|segmentation_heatmap,brush|bare_ground|built_up,blue|green|red|nir|swir16|swir22"
CHANNELS="blue|green|red|nir|swir16|swir22"
#smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BOTH_${ARCH}_TA1_RAW_v61
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=416 \
    --time_steps=3 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="8" \
    --time_span=1y \
    --neg_to_pos_ratio=1.0 \
    --global_saliency_weight=1.00 \
    --global_class_weight=2.00 \
    --time_sampling=soft2 \
    --batch_size=1 \
    --normalize_inputs=1024 \
    --temporal_dropout=True \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


# Transfer BAS+SC WV+L1 With Few Features Toothbrush LinConv - 2022-01-27
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
#CHANNELS="depth,before_after_heatmap|segmentation_heatmap,brush|bare_ground|built_up,blue|green|red|nir|swir16|swir22"
CHANNELS="blue|green|red|nir|swir16|swir22"
#smartwatch stats "$TRAIN_FPATH" "$VALI_FPATH"
EXPERIMENT_NAME=BOTH_${ARCH}_TA1_RAW_v62
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=416 \
    --time_steps=9 \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_centered_positives=True \
    --use_grid_positives=True \
    --num_workers="8" \
    --time_span=1y \
    --neg_to_pos_ratio=1.0 \
    --global_saliency_weight=1.00 \
    --global_class_weight=1.00 \
    --time_sampling=soft2 \
    --batch_size=1 \
    --normalize_inputs=512 \
    --arch_name=$ARCH \
    --temporal_dropout=True \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"
