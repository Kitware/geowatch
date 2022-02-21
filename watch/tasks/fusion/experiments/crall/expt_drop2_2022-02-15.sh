#!/bin/bash

prep_teamfeat_drop2(){
    # Team Features on Drop2
    DVC_DPATH=$(python -m watch.cli.find_dvc)
    WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
    DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
    python -m watch.cli.prepare_teamfeats \
        --base_fpath="$DVC_DPATH/$DATASET_CODE/data.kwcoco.json" \
        --gres=0,1 \
        --with_landcover=1 \
        --with_depth=1 \
        --with_materials=1 \
        --with_invariants=1 \
        --do_splits=1 \
        --depth_workers='auto' \
        --run=0 --cache=1
    #python -m watch.cli.prepare_splits --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L.kwcoco.json --run=False

}


aggregate_multiple_evaluations(){
    __doc__="
    This script will aggregate results over all packaged checkpoints with
    computed metrics. You can run this while the schedule_evaluation script is
    running. It will dump aggregate stats into the 'out_dpath' folder.
    "

    smartwatch stats "$VALI_FPATH"

    #DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    #DVC_DPATH=$HOME/flash1/smart_watch_dvc
    DVC_DPATH=$(python -m watch.cli.find_dvc)

    EXPT_NAME_PAT="*"
    MODEL_EPOCH_PAT="*"
    PRED_DSET_PAT="*"
    MEASURE_GLOBSTR=$DVC_DPATH/models/fusion/SC-20201117/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/eval/curves/measures2.json
    python -m watch.tasks.fusion.gather_results \
        --measure_globstr="$MEASURE_GLOBSTR" \
        --out_dpath="$DVC_DPATH/agg_results/baseline" \
        --dset_group_key="*_vali.kwcoco" --show=True
}

[[ -f '/home/joncrall/data/dvc-repos/smart_watch_dvc/Drop2-Aligned-TA1-2022-01/rutgers_material_seg_v3.kwcoco.json' ]] || python -m watch.tasks.rutgers_material_seg.predict --test_dataset="/home/joncrall/data/dvc-repos/smart_watch_dvc/Drop2-Aligned-TA1-2022-01/data.kwcoco.json" --checkpoint_fpath="/home/joncrall/data/dvc-repos/smart_watch_dvc/models/rutgers/rutgers_peri_materials_v3/experiments_epoch_18_loss_59.014100193977356_valmF1_0.18694573888313187_valChangeF1_0.0_time_2022-02-01-01:53:20.pth" --pred_dataset="/home/joncrall/data/dvc-repos/smart_watch_dvc/Drop2-Aligned-TA1-2022-01/rutgers_material_seg_v3.kwcoco.json" --default_config_key=iarpa --num_workers="2" --batch_size=32 --gpus "0" --compress=DEFLATE --blocksize=128



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
    --temporal_dropout=0.5 \
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
    --temporal_dropout=0.5 \
    --init="$HOME/remote/toothbrush/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


# Fresh Toothbrush - 2022-01-29
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
EXPERIMENT_NAME=BOTH_${ARCH}_TA1_RAW_scratch_v63
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
    --temporal_dropout=0.5  


# Fresh Toothbrush - 2022-01-29
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
EXPERIMENT_NAME=BOTH_${ARCH}_TA1_RAW_scratch_v64
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
    --normalize_inputs=1024 \
    --arch_name=$ARCH \
    --temporal_dropout=0.5 



# Fine Tune For BAS TA-1 Transfer Learning - 2022-02-02
BAS_PRETRAINED_MODEL_FPATH="$DVC_DPATH/models/fusion/SC-20201117/BAS_smt_it_stm_p8_L1_raw_v53/BAS_smt_it_stm_p8_L1_raw_v53_epoch=3-step=85011.pt"
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
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=BAS_${ARCH}_TA1_xfer53_v65
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=416 \
    --time_steps=9 \
    --learning_rate=1e-4 \
    --optimizer=SGD \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --amp_backend=apex \
    --attention_impl=exact \
    --tokenizer=linconv \
    --use_grid_positives=True \
    --use_centered_positives=True \
    --neg_to_pos_ratio=0.25 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.0 \
    --time_span=1y \
    --time_sampling=hardish \
    --num_workers=8 \
    --arch_name=$ARCH \
    --init="$BAS_PRETRAINED_MODEL_FPATH"


# Fine Tune For SC TA-1 Transfer Learning - 2022-02-02
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="blue|green|red|nir|swir16|swir22"
SC_PRETRAINED_MODEL_FPATH="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"
EXPERIMENT_NAME=SC_${ARCH}_TA1_xfer55_v66
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=320 \
    --time_steps=21 \
    --learning_rate=1e-4 \
    --optimizer=SGD \
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
    --num_workers=8 \
    --global_saliency_weight=0.00 \
    --global_class_weight=1.00 \
    --time_span=1y \
    --time_sampling=hardish \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="$SC_PRETRAINED_MODEL_FPATH"


# Fine Tune For SC TA-1 Transfer Learning (with some team features) - 2022-02-04
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22"
SC_PRETRAINED_MODEL_FPATH="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"
EXPERIMENT_NAME=SC_${ARCH}_TA1_xfer55_v67
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=320 \
    --time_steps=16 \
    --learning_rate=1e-4 \
    --optimizer=SGD \
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
    --num_workers=8 \
    --global_saliency_weight=0.00 \
    --global_class_weight=1.00 \
    --time_span=1y \
    --time_sampling=hardish \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="$SC_PRETRAINED_MODEL_FPATH"


# Fine Tune For SC TA-1 Transfer Learning (with some team features) - 2022-02-04
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22"
SC_PRETRAINED_MODEL_FPATH="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"
EXPERIMENT_NAME=SC_${ARCH}_TA1_xfer55_v68
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=320 \
    --time_steps=16 \
    --learning_rate=3e-4 \
    --optimizer=SGD \
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
    --num_workers=8 \
    --global_saliency_weight=0.00 \
    --global_class_weight=1.00 \
    --time_span=1y \
    --time_sampling=hardish \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="$SC_PRETRAINED_MODEL_FPATH"




# Fine Tune For SC TA-1 Transfer Learning (with some team features) - 2022-02-04
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22"
SC_PRETRAINED_MODEL_FPATH="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"
EXPERIMENT_NAME=SC_${ARCH}_TA1_xfer55_v69
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=320 \
    --time_steps=16 \
    --learning_rate=1e-3 \
    --optimizer=SGD \
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
    --num_workers=8 \
    --global_saliency_weight=0.00 \
    --global_class_weight=1.00 \
    --time_span=1y \
    --time_sampling=hardish \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v68/SC_smt_it_stm_p8_TA1_xfer55_v68_epoch=19-step=40959.pt"


# Fine Tune For SC TA-1 Transfer Learning (with some team features) - 2022-02-04
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22"
SC_PRETRAINED_MODEL_FPATH="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"
EXPERIMENT_NAME=SC_${ARCH}_TA1_xfer55_v70
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=320 \
    --time_steps=16 \
    --learning_rate=1e-3 \
    --optimizer=SGD \
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
    --num_workers=8 \
    --global_saliency_weight=0.00 \
    --global_class_weight=1.00 \
    --time_span=1y \
    --time_sampling=hardish \
    --batch_size=1 \
    --arch_name=$ARCH \
    --init="/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v68/SC_smt_it_stm_p8_TA1_xfer55_v68_epoch=19-step=40959.pt"




# --- horologic ---
#
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
ARCH=smt_it_stm_p8
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
CHANNELS="blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
python -m watch.tasks.fusion.fit \
    --config $WORKDIR/configs/common_20201117.yaml  \
    --channels=${CHANNELS} \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --batch_size=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --name="BAS-Template" \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --draw_interval=5000m \
    --num_draw=0 \
    --chip_size=256 \
    --time_steps=5 \
    --tokenizer=dwcnn \
    --optim=AdamW \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --amp_backend=apex \
    --arch_name=$ARCH \
     --dump $WORKDIR/configs/BAS_20220205.yaml

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v071
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --name=$EXPERIMENT_NAME \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=AdamW

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v072
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --name=$EXPERIMENT_NAME \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=SGD

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v073
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --name=$EXPERIMENT_NAME \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=AdamW \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v074
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --name=$EXPERIMENT_NAME \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=SGD \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


# On Namek
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v075
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --saliency_loss='focal' \
    --name=$EXPERIMENT_NAME \
    --class_loss='dicefocal' \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=AdamW 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v076
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --name=$EXPERIMENT_NAME \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=AdamW \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"

# --- toothbrush ---

#/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v70/pred_SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=42-step=88063/Drop2-Aligned-TA1-2022-01_combo_L_nowv_vali.kwcoco/pred.kwcoco.json


kwcoco subset --src "$KWCOCO_BUNDLE_DPATH/combo_L.kwcoco.json" \
        --dst "$KWCOCO_BUNDLE_DPATH/combo_L_nowv.kwcoco.json" \
        --select_images '.sensor_coarse != "WV"' 

# Train + Fine Tune on Korea SUBMISSION CANDIDATE
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
CHANNELS="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22"
SC_PRETRAINED_MODEL_FPATH="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"
EXPERIMENT_NAME=SC_TA1_ALL_REGIONS_c002_v077
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=384 \
    --time_steps=5 \
    --learning_rate=1e-3 \
    --optimizer=RAdam \
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
    --use_grid_positives=False \
    --num_workers=8 \
    --global_saliency_weight=0.00 \
    --global_class_weight=1.00 \
    --time_span=1y \
    --time_sampling=soft2+distribute \
    --batch_size=1 \
    --arch_name=$ARCH \
    --num_draw=1 \
    --init="/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v70/SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=42-step=88063.pt"


# Train + Fine Tune on Korea SUBMISSION CANDIDATE
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
DATASET_CODE=Drop1-20201117
ARCH=smt_it_stm_p8
CHANNELS="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22"
SC_PRETRAINED_MODEL_FPATH="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"
EXPERIMENT_NAME=SC_TA1_ALL_REGIONS_c002_v078
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --channels=${CHANNELS} \
    --name=$EXPERIMENT_NAME \
    --chip_size=384 \
    --time_steps=5 \
    --learning_rate=1e-3 \
    --optimizer=RAdam \
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
    --num_workers=8 \
    --global_saliency_weight=0.10 \
    --global_class_weight=1.00 \
    --time_span=1y \
    --time_sampling=soft2+distribute \
    --batch_size=1 \
    --arch_name=$ARCH \
    --num_draw=1 \
    --init="/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v70/SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=42-step=88063.pt"


# Horologic linconv

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v079
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --name=$EXPERIMENT_NAME \
    --tokenizer=linconv \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=AdamW

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v080
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --name=$EXPERIMENT_NAME \
    --tokenizer=linconv \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=SGD

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v081
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="2"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --name=$EXPERIMENT_NAME \
    --tokenizer=linconv \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=AdamW \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_c001_v082
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="3"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --name=$EXPERIMENT_NAME \
    --tokenizer=linconv \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --optim=SGD \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


# toothbrush - fine-tune on korea

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_KOREA_v083
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --test_dataset=$TEST_FPATH \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --learning_rate=3e-4 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --name=$EXPERIMENT_NAME \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --num_draw=8 \
    --optim=AdamW \
    --normalize_inputs='transfer' \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_c001_v076/BAS_TA1_c001_v076_epoch=90-step=186367.pt"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_nowv_vali.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_ALL_REGIONS_v084
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --default_root_dir=$DEFAULT_ROOT_DIR \
    --train_dataset=$TRAIN_FPATH \
    --vali_dataset=$VALI_FPATH \
    --test_dataset=$TEST_FPATH \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --learning_rate=3e-4 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --name=$EXPERIMENT_NAME \
    --config $WORKDIR/configs/BAS_20220205.yaml  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --num_draw=8 \
    --optim=AdamW \
    --normalize_inputs='transfer' \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_KOREA_v083/BAS_TA1_KOREA_v083_epoch=5-step=11189.pt"

# ------

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_v085
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
export CUDA_VISIBLE_DEVICES="1"
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --channels="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22" \
    --global_class_weight=0.01 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --batch_size=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_sampling=soft2 \
    --attention_impl=exact \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --draw_interval=5000m \
    --num_draw=1 \
    --chip_size=448 \
    --time_steps=5 \
    --time_span=7m \
    --tokenizer=linconv \
    --num_workers=8 \
    --optim=AdamW \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --chip_overlap=0.0 \
    --amp_backend=apex \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_v086
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
export CUDA_VISIBLE_DEVICES="0"
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --channels="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22" \
    --global_class_weight=0.01 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --batch_size=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_sampling=soft2 \
    --attention_impl=exact \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --draw_interval=5000m \
    --num_draw=1 \
    --chip_size=448 \
    --time_steps=5 \
    --time_span=7m \
    --tokenizer=linconv \
    --optim=AdamW \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --num_workers=8 \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --chip_overlap=0.0 \
    --amp_backend=apex \
    --init="noop"

# ------ horologic


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_v087
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
export CUDA_VISIBLE_DEVICES="0"
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --channels="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22" \
    --global_class_weight=0.01 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --batch_size=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_sampling=soft2 \
    --attention_impl=exact \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --draw_interval=5000m \
    --num_draw=1 \
    --chip_size=384 \
    --time_steps=5 \
    --time_span=7m \
    --tokenizer=linconv \
    --num_workers=5 \
    --optim=AdamW \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --chip_overlap=0.0 \
    --amp_backend=apex \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_v088
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
export CUDA_VISIBLE_DEVICES="1"
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --channels="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22" \
    --global_class_weight=0.01 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --batch_size=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_sampling=soft2 \
    --attention_impl=exact \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --draw_interval=5000m \
    --num_draw=1 \
    --chip_size=384 \
    --time_steps=5 \
    --time_span=7m \
    --tokenizer=linconv \
    --optim=AdamW \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --num_workers=5 \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --chip_overlap=0.0 \
    --amp_backend=apex \
    --init="noop"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BAS_TA1_v089
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
export CUDA_VISIBLE_DEVICES="2"
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --channels="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22" \
    --global_class_weight=0.01 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --batch_size=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_sampling=hardish \
    --attention_impl=exact \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --draw_interval=5000m \
    --num_draw=1 \
    --chip_size=384 \
    --time_steps=5 \
    --time_span=7m \
    --tokenizer=linconv \
    --num_workers=5 \
    --optim=AdamW \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --chip_overlap=0.0 \
    --amp_backend=apex \
    --init="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_L_vali.kwcoco.json
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
EXPERIMENT_NAME=BAS_TA1_v090
DATASET_CODE=Drop1-20201117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
export CUDA_VISIBLE_DEVICES="3"
__check__='
smartwatch stats $VALI_FPATH $TRAIN_FPATH
'
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20201117.yaml"  \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --channels="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,blue|green|red|nir|swir16|swir22" \
    --global_class_weight=0.01 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --batch_size=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_sampling=hardish \
    --attention_impl=exact \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --draw_interval=5000m \
    --num_draw=1 \
    --chip_size=384 \
    --time_steps=5 \
    --time_span=7m \
    --tokenizer=linconv \
    --optim=AdamW \
    --method="MultimodalTransformer" \
    --gpus "1" \
    --num_workers=5 \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --chip_overlap=0.0 \
    --amp_backend=apex \
    --init="noop"


# ------ toothbrush -2020-02-17


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,invariants.0:7,invariants.7,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p1_v0100
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p1 \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,invariants.0:8,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p2w_v0101
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p2w \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,invariants.0:8,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p2w_v0102
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p2w \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


# ------ horologic


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p2w \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v0104
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p2w \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p2w \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v2_v0106
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p2w \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"

# toothbrush

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,invariants.0:7,invariants.7,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p1_scratch_v0107
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p1 \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,invariants.0:8,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BOTH_TA1_COMBO_TINY_p2w_scratch_v0108
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p2w \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"

