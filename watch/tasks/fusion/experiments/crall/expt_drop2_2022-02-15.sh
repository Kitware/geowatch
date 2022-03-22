#!/bin/bash
__notes__="
https://docs.google.com/spreadsheets/d/1kYseTFyLb-_7BzILtSOWuimRVyLuefaninfNwkg45r4/edit#gid=0
"

prep_teamfeat_drop2(){
    # Team Features on Drop2
    DVC_DPATH=$(python -m watch.cli.find_dvc --hardware=ssd)
    DVC_DPATH=$(python -m watch.cli.find_dvc --hardware=hdd)
    WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
    DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
    python -m watch.cli.prepare_teamfeats \
        --base_fpath="$DVC_DPATH/$DATASET_CODE/data.kwcoco.json" \
        --gres="0,1" \
        --with_landcover=1 \
        --with_depth=0 \
        --with_materials=1 \
        --with_invariants=0 \
        --do_splits=1 \
        --depth_workers=0 \
        --cache=0 --run=1 --serial=1
    #python -m watch.cli.prepare_splits --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L.kwcoco.json --run=False

}


repackage_checkpoints_and_evaluate(){
    __doc__='
    Prepare existing checkpoints for DVC storage and evaluation
    '

    DVC_DPATH=$(python -m watch.cli.find_dvc)
    DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
    EXPT_GROUP_CODE=eval3_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    python -m watch.tasks.fusion.repackage gather_checkpoints \
        --dvc_dpath="$DVC_DPATH" \
        --storage_dpath="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages" \
        --train_dpath="$DVC_DPATH/training/$HOSTNAME/$USER/$DATASET_CODE/runs/*/lightning_logs" \
        --mode=copy

        #--mode=commit

    # Note: change backend to tmux if slurm is not installed
    DVC_DPATH=$(python -m watch.cli.find_dvc)
    DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
    EXPT_GROUP_CODE=eval3_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*.pt" \
            --test_dataset="$VALI_FPATH" \
            --run=0 --skip_existing=True --backend=slurm 

    #####
    # Alternative invocations : only schedule prediction, then evaluate independently
    #####

    DVC_DPATH=$(python -m watch.cli.find_dvc)
    DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
    EXPT_GROUP_CODE=eval3_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*.pt" \
            --test_dataset="$VALI_FPATH" \
            --run=1 --skip_existing=0 --backend=slurm --enable_pred=False

    # As metrics are reported add them to dvc via the following
    DVC_DPATH=$(python -m watch.cli.find_dvc)
    ls "$DVC_DPATH"/models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json




    ##
    ##
    # How to package metrics
    #
    # After eval, adding the measures2.json file to DVC will prevent other
    # machines from needing to rerun prediction to compare against past results

    # paths of interest
    __doc__="
    ls models/fusion/eval3_candidates/packages/*
    ls models/fusion/eval3_candidates/pred/*/*/*/*/pred.kwcoco.json
    ls models/fusion/eval3_candidates/pred/*/*/*/*/_assets
    models/fusion/eval3_candidates/pred/*/*/*/*/eval

    ls models/fusion/eval3_candidates/eval/*/*/*/*/eval
    ls models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/*.png
    ls models/fusion/eval3_candidates/eval/*/*/*/*/eval/heatmaps
    "

    DVC_DPATH=$(python -m watch.cli.find_dvc)
    (cd "$DVC_DPATH" && dvc add models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json)
    
}


aggregate_multiple_evaluations(){
    __doc__="
    This script will aggregate results over all packaged checkpoints with
    computed metrics. You can run this while the schedule_evaluation script is
    running. It will dump aggregate stats into the 'out_dpath' folder.
    "

    DVC_DPATH=$(python -m watch.cli.find_dvc)
    EXPT_GROUP_CODE=eval3_candidates
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="BOTH_TA1_COMBO_TINY_p2w_raw*"
    MODEL_EPOCH_PAT="*"
    PRED_DSET_PAT="*"
    PRED_CFG_PAT="*"
    MEASURE_GLOBSTR=${DVC_DPATH}/models/fusion/${EXPT_GROUP_CODE}/eval/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/${PRED_CFG_PAT}/eval/curves/measures2.json

    python -m watch.tasks.fusion.gather_results \
        --measure_globstr="$MEASURE_GLOBSTR" \
        --out_dpath="$DVC_DPATH/agg_results/$EXPT_GROUP_CODE" \
        --dset_group_key="*" --show=True \
        --classes_of_interest "Site Preparation" "Active Construction"
}


#### Baseline BAS+SC config
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
EXPERIMENT_NAME=BaselineTemplate2022-03-03
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="blue|green|red" \
    --global_change_weight=0.00 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.5 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --optimizer=AdamW \
    --learning_rate=3e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=256 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=120 \
    --patience=120 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="noop" \
       --package_fpath="$PACKAGE_FPATH" \
        --train_dataset="$TRAIN_FPATH" \
         --vali_dataset="$VALI_FPATH" \
         --test_dataset="$TEST_FPATH" \
         --num_sanity_val_steps=0 \
         --dump "$WORKDIR/configs/common_20220303.yaml"


    #--use_centered_positives=True \ # Should have been true
    #--multimodal_reduce=max \
    #--modulate_class_weights="positive*0,negative*0,background*0.001,No Activity*0.0,Post Construction*0.0001" \
    #--dist_weight=True \
    #--stream_channels=8



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


# namek


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="matseg_0|matseg_1|matseg_2|matseg_3|invariants.0:8|forest|built_up|water|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BOTH_TA1_p8_scratch_fused_norgb_v0109
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=7e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=256 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=none \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3|invariants.0:8|forest|built_up|water"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BOTH_TA1_p8_scratch_2stream_v0110
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=7e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=256 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=none \
    --draw_interval=5000m \
    --num_draw=1 \
    --amp_backend=apex \
    --init="$INITIAL_STATE"


# yardrat

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BASELINE_EXPERIMENT_V111
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=7e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=256 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=none \
    --draw_interval=5000m \
    --num_draw=1 \
    --amp_backend=apex \
    --init="$INITIAL_STATE"



# yardrat

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BASELINE_EXPERIMENT_nowv_V111
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=7e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=256 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=none \
    --draw_interval=5000m \
    --num_draw=1 \
    --amp_backend=apex \
    --init="$INITIAL_STATE"



### horologic --- 2022-02-27


# ------ horologic


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22|matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matearly_nowv_p2w_V112
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-5 \
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
    --arch_name=smt_it_stm_p8 \
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
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matlate_nowv_p8_V113
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-5 \
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
    --arch_name=smt_it_stm_p8 \
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
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matlate_nowv_p2w_V114
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-5 \
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
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22|matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matearly_nowv_p2w_V115
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-5 \
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


### Toothbrush 2022-02-27


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22|matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matearly_nowv_p2w_V116
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=3e-5 \
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
    --arch_name=smt_it_stm_p8 \
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
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matlate_nowv_p8_V117
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=3e-5 \
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
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


### Toothbrush 2022-02-28


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22|matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matearly_nowv_p2w_V118
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=3e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=SGD \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
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
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matlate_nowv_p8_V119
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=3e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=SGD \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


# ooo

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BASELINE_EXPERIMENT_V120
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=7e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=256 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=none \
    --draw_interval=5000m \
    --num_draw=1 \
    --amp_backend=apex \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16"
INITIAL_STATE="noop"
EXPERIMENT_NAME=BASELINE_EXPERIMENT_V121
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.1 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=7e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=256 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=none \
    --draw_interval=5000m \
    --num_draw=1 \
    --amp_backend=apex \
    --init="$INITIAL_STATE"


### Toothbrush 2022-02-28


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22|matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matearly_nowv_p2w_V122
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=3e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=SGD \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
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
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_matlate_nowv_p8_V123
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=3e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=SGD \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE"


#### GENERAL 2022-03-01


DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
EXPERIMENT_NAME=BaselineTemplate2022-03-01
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package_$EXPERIMENT_NAME.pt 
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="blue|green|red|nir|swir16|swir22" \
    --global_change_weight=0.00 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --optimizer=AdamW \
    --learning_rate=3e-5 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.5 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="noop" \
       --package_fpath="$PACKAGE_FPATH" \
        --train_dataset="$TRAIN_FPATH" \
         --vali_dataset="$VALI_FPATH" \
         --test_dataset="$TEST_FPATH" \
         --num_sanity_val_steps=0 \
         --dump "$WORKDIR/configs/common_20220301.yaml"

### Horologic 2022-03-01

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_M_late_nowv_p8_shorter_V124
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220301.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-5 \
    --attention_impl=exact \
    --chip_overlap=0.5 \
    --optimizer=AdamW \
    --max_epoch_length=256 \
    --arch_name=smt_it_stm_p8 \
    --init="$INITIAL_STATE"

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_M_late_nowv_p8_scratch_shorter_V125
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220301.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-5 \
    --attention_impl=exact \
    --chip_overlap=0.3 \
    --optimizer=AdamW \
    --max_epoch_length=256 \
    --arch_name=smt_it_stm_p8 \
    --init="noop"

export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_M_late_nowv_p8_shorter_sgd_V126
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220301.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=1e-3 \
    --attention_impl=exact \
    --chip_overlap=0.3 \
    --optimizer=SGD \
    --max_epoch_length=256 \
    --arch_name=smt_it_stm_p8 \
    --init="$INITIAL_STATE"

export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_M_late_nowv_p8_scratch_shorter_sgd_V127
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220301.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=1e-3 \
    --attention_impl=exact \
    --chip_overlap=0.3 \
    --optimizer=SGD \
    --max_epoch_length=256 \
    --arch_name=smt_it_stm_p8 \
    --init="noop"


### Toothbrush 2022-03-01


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_late_nowv_p8_shorter_V128
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220301.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-5 \
    --attention_impl=exact \
    --chip_overlap=0.3 \
    --optimizer=AdamW \
    --max_epoch_length=256 \
    --arch_name=smt_it_stm_p8 \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_late_nowv_p24_shorter_V129
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220301.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --chip_size=256 \
    --time_steps=3 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-5 \
    --attention_impl=exact \
    --chip_overlap=0.3 \
    --optimizer=AdamW \
    --max_epoch_length=256 \
    --arch_name=smt_it_stm_s24 \
    --init="$INITIAL_STATE"

### Toothbrush 2022-03-02


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_only_nowv_p8_V130
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220301.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=0.0003 \
    --attention_impl=exact \
    --chip_overlap=0.3 \
    --optimizer=AdamW \
    --max_epoch_length=none \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=10m \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_only_nowv_p12_V131
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220301.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --chip_size=256 \
    --time_steps=5 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=0.003 \
    --attention_impl=exact \
    --chip_overlap=0.3 \
    --optimizer=SGD \
    --max_epoch_length=none \
    --arch_name=smt_it_stm_n12 \
    --num_draw=8 \
    --draw_interval=10m \
    --init="$INITIAL_STATE" \
    --auto_lr_find=True


### Toothbrush 2022-03-03

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,depth,panchromatic"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_SC_DM_wv_p8_V132
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.0 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --neg_to_pos_ratio=1.0 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=3e-5 \
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
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=4 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --use_conditional_classes=True \
    --init="$INITIAL_STATE"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,depth,panchromatic"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_SC_DM_wv_p8_V133
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=0.00 \
    --neg_to_pos_ratio=0.5 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-5 \
    --weight_decay=1e-8 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=224 \
    --time_steps=11 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=4 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --use_conditional_classes=True \
    --init="$INITIAL_STATE"


### Horologic 2022-03-03

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,depth,panchromatic"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_BOTH_DM_wv_p8_V134
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.0 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.5 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-5 \
    --weight_decay=1e-8 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=224 \
    --time_steps=11 \
    --chip_overlap=0.0 \
    --time_sampling=hardish \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=120 \
    --patience=120 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=0 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --use_conditional_classes=True \
    --min_spacetime_weight=0.5 \
    --init="$INITIAL_STATE"

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,depth,panchromatic"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_BOTH_DM_wv_p8_V135
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.0 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.5 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --learning_rate=1e-5 \
    --weight_decay=1e-8 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=224 \
    --time_steps=11 \
    --chip_overlap=0.0 \
    --time_sampling=hardish \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=120 \
    --patience=120 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=0 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --use_conditional_classes=True \
    --min_spacetime_weight=0.5 \
    --init="$INITIAL_STATE"

export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,depth,panchromatic"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_DM_wv_p8_V136
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.0 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.5 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --learning_rate=1e-4 \
    --weight_decay=1e-8 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=224 \
    --time_steps=11 \
    --chip_overlap=0.0 \
    --time_sampling=hardish \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=120 \
    --patience=120 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=0 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --use_conditional_classes=True \
    --min_spacetime_weight=0.5 \
    --init="$INITIAL_STATE"

export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,depth,panchromatic"
INITIAL_STATE="$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v52/BOTH_smt_it_stm_p8_L1_DIL_v52_epoch=13-step=55215.pt"
EXPERIMENT_NAME=FUSION_EXPERIMENT_DM_wv_p8_V137
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.0 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --neg_to_pos_ratio=0.5 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=16 \
    --learning_rate=1e-4 \
    --weight_decay=1e-8 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=224 \
    --time_steps=11 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=120 \
    --patience=120 \
    --max_epoch_length=1024 \
    --draw_interval=5000m \
    --num_draw=2 \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --use_conditional_classes=True \
    --min_spacetime_weight=0.5 \
    --init="$INITIAL_STATE"


# -------------------------------------

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V138
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=48 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-3 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=1024 \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=5m \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_SC_ML_V139
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=32 \
    --chip_size=128 \
    --time_steps=32 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=1e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=1024 \
    --arch_name=smt_it_stm_p16 \
    --num_draw=8 \
    --draw_interval=5m \
    --init="$INITIAL_STATE" 


# ------------------------------------- horologic 2022-03-08

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V140
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-3 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=1024 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=5m \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=/flash/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V141
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=14 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-3 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=1024 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=2 \
    --init="$INITIAL_STATE" 

export CUDA_VISIBLE_DEVICES=2
#DVC_DPATH=$(python -m watch.cli.find_dvc)
DVC_DPATH=/flash/smart_watch_dvc
#$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V142
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=1024 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=/flash/smart_watch_dvc
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V143
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=14 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=1024 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=2 \
    --init="$INITIAL_STATE" 

# ------------------------------------- toothbrush 2022-03-08

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V144
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=1024 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=5m \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V145
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=1e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=1024 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=5m \
    --init="$INITIAL_STATE" 


# ------------------------------------- toothbrush 2022-03-10

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_ILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_ILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_ILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V146
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=1m \
    --dist_weight=True \
    --modulate_class_weights="positive*0,negative*0,background*0.001,No Activity*0.0,Post Construction*0.0001" \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_ILM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_ILM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_ILM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V147
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=1e-3 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=SGD \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --multimodal_reduce=mean \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --modulate_class_weights="positive*0,negative*0,background*1e-3,No Activity*1e-9,Post Construction*1e-2" \
    --stream_channels=32 \
    --init="$INITIAL_STATE" 


# ------------------------------------- toothbrush 2022-03-10

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V148
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --decoder=segmenter \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=1m \
    --dist_weight=True \
    --modulate_class_weights="positive*0,negative*0,background*0.001,No Activity*0.0,Post Construction*0.0001" \
    --init="$INITIAL_STATE" 


# ------------------------------------- horologic 2022-03-12

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V149
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --decoder=segmenter \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --weight_decay=1e-8 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --modulate_class_weights="positive*0,negative*0,background*0.001,No Activity*0.0,Post Construction*0.0001" \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V150
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --decoder=segmenter \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --weight_decay=1e-8 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --modulate_class_weights="" \
    --init="$INITIAL_STATE" 

export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V151
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --decoder=segmenter \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --weight_decay=1e-7 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --modulate_class_weights="" \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V152
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=256 \
    --decoder=segmenter \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --weight_decay=1e-7 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p8 \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --modulate_class_weights="positive*0,negative*0,background*0.001,No Activity*0.0,Post Construction*0.0001" \
    --init="$INITIAL_STATE" 

# ------------------------------------- toothbrush 2022-03-14

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,depth,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V153
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --decoder=segmenter \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=RAdam \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p4 \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --modulate_class_weights="positive*0,negative*0,background*0.001,No Activity*0.0,Post Construction*0.0001" \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,depth,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V154
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=8 \
    --chip_size=128 \
    --decoder=segmenter \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=0.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=RAdam \
    --max_epoch_length=2048 \
    --time_sampling=hardish \
    --arch_name=smt_it_stm_p2 \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --modulate_class_weights="positive*0,negative*0,background*0.001,No Activity*0.0,Post Construction*0.0001" \
    --init="$INITIAL_STATE" 


# ------------------------------------- toothbrush 2022-03-17


#TODO:

# CHANNELS="(S2,L8,WV):blue|green|red,(S2,L8):nir|swir16|swir22,(WV):depth|pan,(WV):depth|red|green|blue,(S2,L8):matseg_0|matseg_1|matseg_2|matseg_3"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,depth,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V155
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=24 \
    --chip_size=224 \
    --decoder=segmenter \
    --tokenizer=dwcnn \
    --time_steps=5 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=RAdam \
    --max_epoch_length=None \
    --time_sampling=hardish \
    --arch_name=smt_it_sm_p2 \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --stream_channels=64 \
    --modulate_class_weights="positive*0,negative*0,background*0.1,No Activity*0.0,Post Construction*0.0,Site Preparation*2.0" \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,depth,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/FUSION_EXPERIMENT_ML_V146/FUSION_EXPERIMENT_ML_V146_epoch=67-step=17407.pt
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V156
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --neg_to_pos_ratio=0.5 \
    --accumulate_grad_batches=16 \
    --chip_size=224 \
    --decoder=segmenter \
    --tokenizer=dwcnn \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=3e-4 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=RAdam \
    --max_epoch_length=None \
    --time_sampling=hardish \
    --arch_name=smt_it_sm_m24 \
    --max_epoch_length=4096 \
    --num_draw=8 \
    --draw_interval=100m \
    --dist_weight=True \
    --stream_channels=64 \
    --modulate_class_weights="positive*0,negative*0,background*0.1,No Activity*0.0,Post Construction*0.0,Site Preparation*2.0" \
    --init="$INITIAL_STATE" 


# ------------------------------------- toothbrush 2022-03-19


#TODO:

# CHANNELS="(S2,L8,WV):blue|green|red,(S2,L8):nir|swir16|swir22,(WV):depth|pan,(WV):depth|red|green|blue,(S2,L8):matseg_0|matseg_1|matseg_2|matseg_3"


#export CUDA_VISIBLE_DEVICES=1
#DVC_DPATH=$(python -m watch.cli.find_dvc)
#WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
#DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
#KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
#TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
#TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
#CHANNELS="blue|green|red|nir|swir16|swir22,depth,matseg_0|matseg_1|matseg_2|matseg_3"
#INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/FUSION_EXPERIMENT_ML_V156/FUSION_EXPERIMENT_ML_V156_epoch=39-step=10239.pt \
#EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V156-cont1
#DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
#python -m watch.tasks.fusion.fit \
#    --config "$WORKDIR/configs/common_20220303.yaml" \
#    --default_root_dir="$DEFAULT_ROOT_DIR" \
#    --name=$EXPERIMENT_NAME \
#    --train_dataset="$TRAIN_FPATH" \
#    --vali_dataset="$VALI_FPATH" \
#    --test_dataset="$TEST_FPATH" \
#    --use_centered_positives=True \
#    --channels="$CHANNELS" \
#    --neg_to_pos_ratio=0.5 \
#    --accumulate_grad_batches=16 \
#    --chip_size=224 \
#    --decoder=segmenter \
#    --tokenizer=dwcnn \
#    --time_steps=7 \
#    --global_class_weight=1.0 \
#    --global_saliency_weight=1.00 \
#    --num_workers=8 \
#    --gpus "1" \
#    --learning_rate=1e-3 \
#    --attention_impl=exact \
#    --chip_overlap=0.0 \
#    --optimizer=AdamW \
#    --time_sampling=hardish \
#    --arch_name=smt_it_sm_m24 \
#    --max_epoch_length=4096 \
#    --num_draw=8 \
#    --draw_interval=100m \
#    --dist_weight=True \
#    --stream_channels=64 \
#    --modulate_class_weights="positive*0,negative*0,background*0.2,No Activity*0.0,Post Construction*0.0,Site Preparation*2.0" \
#    --init="$INITIAL_STATE" 

#DVC_DPATH=$(python -m watch.cli.find_dvc)
#DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
#EXPT_GROUP_CODE=eval3_candidates
#KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
#python -m watch.tasks.fusion.repackage gather_checkpoints \
#    --dvc_dpath="$DVC_DPATH" \
#    --storage_dpath="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages" \
#    --train_dpath="$DVC_DPATH/training/*/*/*/runs/FUSION_EXPERIMENT_ML_V156-cont1/lightning_logs/version_2/checkpoints/epoch=3-step=1023-v2.ckpt" \
#    --mode=copy


#DVC_DPATH=$(python -m watch.cli.find_dvc)
#DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
#EXPT_GROUP_CODE=eval3_candidates
#KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
#ls $DVC_DPATH/training/*/*/*/runs/FUSION_EXPERIMENT_ML_V155-cont1/lightning_logs/*/checkpoints
#python -m watch.tasks.fusion.repackage gather_checkpoints \
#    --dvc_dpath="$DVC_DPATH" \
#    --storage_dpath="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages" \
#    --train_dpath="$DVC_DPATH/training/*/*/*/runs/FUSION_EXPERIMENT_ML_V155-cont1/lightning_logs/version_2/checkpoints/epoch=3-step=1023-v2.ckpt" \
#    --mode=copy


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,depth,matseg_0|matseg_1|matseg_2|matseg_3"
INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/FUSION_EXPERIMENT_ML_V155/FUSION_EXPERIMENT_ML_V155_epoch=18-step=41628.pt \
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V155-cont1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=24 \
    --chip_size=224 \
    --decoder=segmenter \
    --tokenizer=linconv \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --gpus "1" \
    --learning_rate=1e-2 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --max_epoch_length=None \
    --time_sampling=hardish \
    --arch_name=smt_it_sm_p2w \
    --num_draw=8 \
    --draw_interval=1m \
    --max_epoch_length=16384 \
    --dist_weight=True \
    --stream_channels=64 \
    --temporal_dropout=0.22 \
    --modulate_class_weights="positive*0,negative*0,background*0.2,No Activity*0.0,Post Construction*0.0,Site Preparation*2.0" \
    --init="$INITIAL_STATE" 



export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22,blue|green|red,depth,panchromatic,matseg_0|matseg_1|matseg_2|matseg_3"
#INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/FUSION_EXPERIMENT_ML_V155/FUSION_EXPERIMENT_ML_V155_epoch=18-step=41628.pt 
INITIAL_STATE="noop"
EXPERIMENT_NAME=FUSION_EXPERIMENT_ML_V157
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config "$WORKDIR/configs/common_20220303.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --use_centered_positives=True \
    --channels="$CHANNELS" \
    --neg_to_pos_ratio=0.3 \
    --accumulate_grad_batches=16 \
    --chip_size=196 \
    --decoder=segmenter \
    --tokenizer=linconv \
    --time_steps=7 \
    --global_class_weight=1.0 \
    --global_saliency_weight=1.00 \
    --num_workers=0 \
    --gpus "1" \
    --learning_rate=1e-3 \
    --attention_impl=exact \
    --chip_overlap=0.0 \
    --optimizer=AdamW \
    --time_sampling=hardish \
    --arch_name=smt_it_sm_p1 \
    --max_epoch_length=4096 \
    --num_draw=8 \
    --draw_interval=1m \
    --dist_weight=True \
    --stream_channels=4 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*0.2,No Activity*0.0,Post Construction*0.0,Site Preparation*2.0" \
    --init="$INITIAL_STATE" 
