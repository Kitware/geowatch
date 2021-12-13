
#Activity_smt_it_joint_m24_newanns_rgb_v4_epoch
prep_and_inspect(){
    python -m watch.cli.coco_visualize_videos \
       --src $KWCOCO_BUNDLE_DPATH/prop_data.kwcoco.json \
       --channels "red|green|blue" \
       --draw_imgs=False  --animate=True

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1
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
        --pred_dataset=$KWCOCO_BUNDLE_DPATH/rutgers_nowv.kwcoco.json \
        --num_workers="16" \
        --batch_size=4 --gpus "0" \
        --compress=RAW --blocksize=64

    export CUDA_VISIBLE_DEVICES="1"
    python -m watch.tasks.landcover.predict \
        --dataset=$KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
        --deployed=$DZYNE_LANDCOVER_MODEL_FPATH  \
        --device=0 \
        --num_workers="16" \
        --output=$KWCOCO_BUNDLE_DPATH/landcover_nowv.kwcoco.json

    python ~/code/watch/watch/cli/coco_combine_features.py \
        --src $KWCOCO_BUNDLE_DPATH/data_nowv.kwcoco.json \
              $KWCOCO_BUNDLE_DPATH/rutgers_nowv.kwcoco.json \
              $KWCOCO_BUNDLE_DPATH/landcover_nowv.kwcoco.json \
        --dst $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json

    python -m watch project \
        --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
        --src $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json \
        --dst $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json.proj

    kwcoco stats $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json.proj
    mv $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json.proj $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json

    python -m watch stats $KWCOCO_BUNDLE_DPATH/combo_nowv.kwcoco.json

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
