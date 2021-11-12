__doc__='
Script for generating TA2 features on aligned kwcoco videos

Set the appropraite environment variables for your machine:

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
KWCOCO_BUNDLE_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned

Sourcing this script just defines the environment variables and prediction
functions. To do the prediction first source the script then run:

Example:
    source ~/code/watch/scripts/generate_ta2_features.sh
    predict_all_ta2_features
'

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Aligned-L1}

BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
#BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/propogated.kwcoco.json


# Models

# Older
#RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_30_loss_0.05691597167379317_valmIoU_0.5694727912477856_time_2021-08-07-09:01:01.pth"
#DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH/models/landcover/visnav_osm.pt"
#UKY_S2_MODEL_FPATH=${UKY_L8_MODEL_FPATH:-$DVC_DPATH/models/uky_invariants/sort_augment_overlap/S2_drop1-S2-L8-aligned-old.0.ckpt}
#UKY_L8_MODEL_FPATH=${UKY_L8_MODEL_FPATH:-$DVC_DPATH/models/uky_invariants/sort_augment_overlap/L8_drop1-S2-L8-aligned-old.0.ckpt}
#DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH/models/landcover/visnav_sentinel2.pt"

# Current
UKY_S2_MODEL_FPATH=${UKY_S2_MODEL_FPATH:-$DVC_DPATH/models/uky_features_21-10-01/S2_model/drop1-S2-L8-aligned/checkpoints/S2_drop1-S2-L8-aligned.cpkt}
UKY_L8_MODEL_FPATH=${UKY_L8_MODEL_FPATH:-$DVC_DPATH/models/uky_features_21-10-01/L8_model/drop1-S2-L8-aligned/checkpoints/L8_drop1-S2-L8-aligned.cpkt}
RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_20211001T162707.pth"
DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH/models/landcover/visnav_remap_s2_subset.pt"


UKY_S2_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/_partial_uky_pred_S2.kwcoco.json
UKY_L8_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/_partial_uky_pred_L8.kwcoco.json

UKY_INVARIANTS_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/uky_pred_S2_L8.kwcoco.json
RUTGERS_MATERIAL_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/rutgers_material_seg.kwcoco.json
DZYNE_LANDCOVER_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/landcover.kwcoco.json

COMBO_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_data.kwcoco.json

COMBO_TRAIN_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
COMBO_VALI_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json

COMBO_TRAIN_S2_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_s2_data.kwcoco.json
COMBO_VALI_S2_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_s2_data.kwcoco.json

BASE_TRAIN_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/base_train_data.kwcoco.json
BASE_VALI_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/base_vali_data.kwcoco.json


echo  "
=====================================================

DVC_DPATH                  = $DVC_DPATH
KWCOCO_BUNDLE_DPATH        = $KWCOCO_BUNDLE_DPATH

# Model files
UKY_S2_MODEL_FPATH           = $UKY_S2_MODEL_FPATH
UKY_L8_MODEL_FPATH           = $UKY_L8_MODEL_FPATH
RUTGERS_MATERIAL_MODEL_FPATH = $RUTGERS_MATERIAL_MODEL_FPATH
DZYNE_LANDCOVER_MODEL_FPATH  = $DZYNE_LANDCOVER_MODEL_FPATH

# Intermediate files
UKY_S2_COCO_FPATH          = $UKY_S2_COCO_FPATH
UKY_L8_COCO_FPATH          = $UKY_L8_COCO_FPATH

# Final feature files
BASE_COCO_FPATH             = $BASE_COCO_FPATH
UKY_INVARIANTS_COCO_FPATH   = $UKY_INVARIANTS_COCO_FPATH
RUTGERS_MATERIAL_COCO_FPATH = $RUTGERS_MATERIAL_COCO_FPATH
DZYNE_LANDCOVER_COCO_FPATH  = $DZYNE_LANDCOVER_COCO_FPATH

# Final output file
COMBO_COCO_FPATH           = $COMBO_COCO_FPATH

=====================================================
"

base_split(){
    __doc__='
    source ~/code/watch/scripts/generate_ta2_features.sh
    '
    python -m watch stats $BASE_COCO_FPATH

    # Split out train and validation data (TODO: add test when we can)
    kwcoco subset --src $BASE_COCO_FPATH \
            --dst $BASE_VALI_COCO_FPATH \
            --select_videos '.name | startswith("KR_")'

    kwcoco subset --src $COMBO_COCO_FPATH \
            --dst $BASE_TRAIN_COCO_FPATH \
            --select_videos '.name | startswith("KR_") | not'
}



uky_prediction(){
    __doc__='
    source ~/code/watch/scripts/generate_ta2_features.sh
    '
    # --------------
    # UKY Prediction
    # --------------

    # Predict with UKY Invariants (one model for S2 and L8)
    export CUDA_VISIBLE_DEVICES="0"
    python -m watch.tasks.invariants.predict \
        --sensor="S2" \
        --input_kwcoco $BASE_COCO_FPATH \
        --output_kwcoco $UKY_S2_COCO_FPATH \
        --ckpt_path $UKY_S2_MODEL_FPATH  \
        --device=cuda \
        --num_workers="8"

        #--gpus 1 \

    export CUDA_VISIBLE_DEVICES="0"
    python -m watch.tasks.invariants.predict \
        --sensor L8 \
        --input_kwcoco $BASE_COCO_FPATH \
        --output_kwcoco $UKY_L8_COCO_FPATH \
        --ckpt_path $UKY_L8_MODEL_FPATH \
        --device=cuda \
        --num_workers="8"

        #--gpus 1 \

    kwcoco stats $UKY_S2_COCO_FPATH $UKY_L8_COCO_FPATH

    # Combine S2 and L8 outputs into a single UKY file
    python -m watch.cli.coco_combine_features \
        --src $UKY_S2_COCO_FPATH $UKY_L8_COCO_FPATH \
        --dst $UKY_INVARIANTS_COCO_FPATH
}


HACKED_S2_MOEL_uky_prediction(){
    __doc__='
    source ~/code/watch/scripts/generate_ta2_features.sh
    '
    # --------------
    # UKY Prediction
    # --------------

    # Hack to use the same model to predict on all bands
    export CUDA_VISIBLE_DEVICES="0"
    python -m watch.tasks.invariants.predict \
        --sensor="all" \
        --bands="S2" \
        --input_kwcoco $BASE_COCO_FPATH \
        --output_kwcoco $UKY_INVARIANTS_COCO_FPATH \
        --ckpt_path $UKY_S2_MODEL_FPATH  \
        --device=cuda \
        --num_workers="16"
}


rutgers_prediction(){
    __doc__='
    source ~/code/watch/scripts/generate_ta2_features.sh
    '
    # Generate Rutgers Features
    export CUDA_VISIBLE_DEVICES="0"
    python -m watch.tasks.rutgers_material_seg.predict \
        --test_dataset=$BASE_COCO_FPATH \
        --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
        --default_config_key=iarpa \
        --pred_dataset=$RUTGERS_MATERIAL_COCO_FPATH \
        --num_workers="8" \
        --batch_size=32 --gpus "0" \
        --compress=RAW --blocksize=64
}



dzyne_prediction(){
    __doc__='
    source ~/code/watch/scripts/generate_ta2_features.sh
    '
    # ----------------
    # DZYNE Prediction
    # ----------------

    # 88 in 40 seconds
    # 44 in 40 seconds
    #export CUDA_VISIBLE_DEVICES="1"
    python -m watch.tasks.landcover.predict \
        --dataset=$BASE_COCO_FPATH \
        --deployed=$DZYNE_LANDCOVER_MODEL_FPATH  \
        --device=0 \
        --num_workers="16" \
        --output=$DZYNE_LANDCOVER_COCO_FPATH
          
    #\
    #    --num_workers=12 \
    #    --batch_size=4 
    ##--gpus "0"
}


predict_all_ta2_features(){
    __doc__="
    Spot check auxiliary features alignment

    Example:
        source ~/code/watch/scripts/generate_ta2_features.sh
    "
    # Run checks
    if [ ! -d "$KWCOCO_BUNDLE_DPATH" ]; then
        echo "MISSING DIRECTORY KWCOCO_BUNDLE_DPATH=$KWCOCO_BUNDLE_DPATH"
        exit 1
    fi

    if [ ! -f "$BASE_COCO_FPATH" ]; then
        echo "MISSING FILE BASE_COCO_FPATH=$BASE_COCO_FPATH"
        exit 1
    fi

    uky_prediction
    rutgers_prediction
    dzyne_prediction

    # Final Combination
    # Combine TA2 Team Features into a single file
    kwcoco stats $BASE_COCO_FPATH $UKY_INVARIANTS_COCO_FPATH $RUTGERS_MATERIAL_COCO_FPATH $DZYNE_LANDCOVER_COCO_FPATH

    kwcoco validate $BASE_COCO_FPATH
    kwcoco validate $UKY_INVARIANTS_COCO_FPATH
    kwcoco validate $RUTGERS_MATERIAL_COCO_FPATH
    kwcoco validate $DZYNE_LANDCOVER_COCO_FPATH

    python ~/code/watch/watch/cli/coco_combine_features.py \
        --src $BASE_COCO_FPATH $UKY_INVARIANTS_COCO_FPATH $RUTGERS_MATERIAL_COCO_FPATH $DZYNE_LANDCOVER_COCO_FPATH \
        --dst $COMBO_COCO_FPATH

    kwcoco validate $COMBO_COCO_FPATH

    # Ensure "Video Space" is 10 GSD
    # Might not need to do that?
    #python -m watch.cli.coco_add_watch_fields \
    #    --src $COMBO_COCO_FPATH \
    #    --dst $COMBO_COCO_FPATH \
    #    --target_gsd 10
    
    # Propogate labels (should no longer be needed)
    # python -m watch.cli.propagate_labels \
    #         --src $COMBO_COCO_FPATH --dst $COMBO_PROPOGATED_COCO_FPATH \
    #         --viz_dpath=$KWCOCO_BUNDLE_DPATH/_prop_viz

    #LEFT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_data_left.kwcoco.json
    #RIGHT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_data_right.kwcoco.json
    #python -m watch.cli.coco_spatial_crop \
    #        --src $COMBO_PROPOGATED_COCO_FPATH --dst $LEFT_COCO_FPATH \
    #        --suffix=_left
    #python -m watch.cli.coco_spatial_crop \
    #        --src $COMBO_PROPOGATED_COCO_FPATH --dst $RIGHT_COCO_FPATH \
    #        --suffix=_right

    python -m watch stats $COMBO_COCO_FPATH

    # Split out train and validation data (TODO: add test when we can)
    kwcoco subset --src $COMBO_COCO_FPATH \
            --dst $COMBO_VALI_COCO_FPATH \
            --select_videos '.name | startswith("KR_")'

    kwcoco subset --src $COMBO_COCO_FPATH \
            --dst $COMBO_TRAIN_COCO_FPATH \
            --select_videos '.name | startswith("KR_") | not'

    kwcoco validate $COMBO_VALI_COCO_FPATH
    kwcoco validate $COMBO_TRAIN_COCO_FPATH


    # Also split out S2
    kwcoco subset --src $COMBO_TRAIN_COCO_FPATH \
            --dst $COMBO_TRAIN_S2_COCO_FPATH \
            --select_images '.sensor_coarse == "S2"'

    kwcoco subset --src $COMBO_VALI_COCO_FPATH \
            --dst $COMBO_VALI_S2_COCO_FPATH \
            --select_images '.sensor_coarse == "S2"'
}


viz_check(){
    source ~/code/watch/scripts/generate_ta2_features.sh
    echo "COMBO_COCO_FPATH = $COMBO_COCO_FPATH"
    echo "KWCOCO_BUNDLE_DPATH = $KWCOCO_BUNDLE_DPATH"

    python -m watch stats $COMBO_COCO_FPATH

    # Optional: visualize the combo data before and after propogation
    python -m watch.cli.coco_visualize_videos \
        --src $COMBO_COCO_FPATH --space=video --num_workers=6 \
        --viz_dpath $KWCOCO_BUNDLE_DPATH/_viz_preprop \
        --channels "red|green|blue,inv_sort1|inv_augment1|inv_shared1"

    CHANNELS=matseg_0|matseg_1|matseg_2
    python -m watch.cli.coco_visualize_videos \
        --src $COMBO_COCO_FPATH --space=video --num_workers=6 \
        --viz_dpath $KWCOCO_BUNDLE_DPATH/_viz_preprop \
        --channels "$CHANNELS"

    # Optional: visualize the combo data before and after propogation
    python -m watch.cli.coco_visualize_videos \
        --src $COMBO_PROPOGATED_COCO_FPATH --space=video --num_workers=6 \
        --viz_dpath $KWCOCO_BUNDLE_DPATH/_viz_postprop \
        --channels "red|green|blue,inv_sort1|inv_augment1|inv_shared1"

    # Optional: visualize the combo data before and after propogation
    python -m watch.cli.coco_visualize_videos \
        --src $COMBO_PROPOGATED_COCO_FPATH --space=video --num_workers=6 \
        --viz_dpath $KWCOCO_BUNDLE_DPATH/_viz_postprop \
        --channels "red|green|blue"

    items=$(jq -r '.videos[] | .name' $COMBO_PROPOGATED_COCO_FPATH)
    for item in ${items[@]}; do
        echo "item = $item"
        python -m watch.cli.gifify  --frames_per_second .7 \
            --input "$KWCOCO_BUNDLE_DPATH/_viz_preprop/$item/_anns/red|green|blue/" \
            --output "$KWCOCO_BUNDLE_DPATH/_viz_preprop/${item}_rgb.gif"
        python -m watch.cli.gifify  --frames_per_second .7 \
            --input "$KWCOCO_BUNDLE_DPATH/_viz_preprop/$item/_anns/inv_sort1|inv_augment1|inv_shared1/" \
            --output "$KWCOCO_BUNDLE_DPATH/_viz_preprop/${item}_invariants.gif"
    done

    items=$(jq -r '.videos[] | .name' $COMBO_PROPOGATED_COCO_FPATH)
    for item in ${items[@]}; do
        echo "item = $item"
        python -m watch.cli.gifify --frames_per_second .7 \
            --input "$KWCOCO_BUNDLE_DPATH/_viz_postprop/$item/_anns/red|green|blue/" \
            --output "$KWCOCO_BUNDLE_DPATH/_viz_postprop/${item}_rgb.gif"
        python -m watch.cli.gifify  --frames_per_second .7 \
            --input "$KWCOCO_BUNDLE_DPATH/_viz_postprop/$item/_anns/inv_sort1|inv_augment1|inv_shared1/" \
            --output "$KWCOCO_BUNDLE_DPATH/_viz_postprop/${item}_invariants.gif"
    done
    

    python -m watch.cli.gifify \
        --input "$KWCOCO_BUNDLE_DPATH/_viz_preprop/US_Jacksonville_R01/_anns/red|green|blue/" \
        --output "$KWCOCO_BUNDLE_DPATH/_viz_preprop/US_Jacksonville_R01_rgb.gif"

    python -m watch.cli.gifify \
        --input "$KWCOCO_BUNDLE_DPATH/_viz_postprop/US_Jacksonville_R01/_anns/red|green|blue/" \
        --output "$KWCOCO_BUNDLE_DPATH/_viz_postprop/US_Jacksonville_R01_rgb.gif"

    python -m watch.cli.gifify \
        --input "$KWCOCO_BUNDLE_DPATH/_viz_postprop/US_Jacksonville_R01/_anns/red|green|blue/" \
        --output "$KWCOCO_BUNDLE_DPATH/_viz_postprop/US_Jacksonville_R01_rgb.gif"
    

}


spot_check(){
    __doc__="
    Spot check auxiliary features alignment

    Example:
        source ~/code/watch/scripts/generate_ta2_features.sh
    "

    python -m kwcoco stats $COMBO_COCO_FPATH

    python -m watch stats $COMBO_COCO_FPATH

    python -m watch coco_show_auxiliary --src $RUTGERS_MATERIAL_COCO_FPATH --channels2 matseg_4

    python -m watch coco_show_auxiliary --src $COMBO_COCO_FPATH --channels2 matseg_4

    kwcoco validate $COMBO_COCO_FPATH

    # Optional: visualize the combo data before and after propogation
    #CHANNELS="inv_shared2|inv_shared3|inv_shared4"
    #--num_frames=10 \

    CHANNELS="red|green|blue,inv_sort1|inv_augment1|inv_shared1,matseg_0|matseg_1|matseg_2,grassland|built_up|bare_ground,matseg_3|matseg_4|matseg_5,inv_shared2|inv_shared3|inv_shared4"
    VIZ_DPATH=$KWCOCO_BUNDLE_DPATH/_viz_teamfeats
    python -m watch.cli.coco_visualize_videos \
        --src $COMBO_COCO_FPATH --space=video --num_workers=6 \
        --viz_dpath $VIZ_DPATH \
        --channels $CHANNELS

    # Split bands up into a bash array
    mapfile -td \, _BANDS < <(printf "%s\0" "$CHANNELS")

    items=$(jq -r '.videos[] | .name' $COMBO_COCO_FPATH)
    for item in ${items[@]}; do
        for bandname in ${_BANDS[@]}; do
            echo "_BANDS = $_BANDS"
            BAND_DPATH="$VIZ_DPATH/${item}/_anns/${bandname}/"
            GIF_FPATH="$VIZ_DPATH/${item}_anns_${bandname}.gif"
            python -m watch.cli.gifify --frames_per_second .7 \
                --input "$BAND_DPATH" --output "$GIF_FPATH"
        done
    done

    # Print stats
    kwcoco stats \
        $COMBO_TRAIN_COCO_FPATH \
        $COMBO_VALI_COCO_FPATH 

}


split_demo(){
    DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
    KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
    BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json

    TRAIN_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
    VALI_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json

    # Split out train and validation data 
    # (TODO: add test when we get enough data)
    kwcoco subset --src $BASE_COCO_FPATH \
            --dst $TRAIN_COCO_FPATH \
            --select_videos '.name | startswith("KR_")'

    kwcoco subset --src $COMBO_COCO_FPATH \
            --dst $VALI_COCO_FPATH \
            --select_videos '.name | startswith("KR_") | not'

}


train_model(){
    DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
    KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
    TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_data_right.kwcoco.json
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_data_left.kwcoco.json
    TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_data_left.kwcoco.json

    WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

    ARCH=smt_it_stm_p8
    #ARCH=smt_it_joint_p8

    CHANNELS="blue|green|red|nir|inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|rice_field|cropland|water|inland_water|river_or_stream|sebkha|snow_or_ice_field|bare_ground|sand_dune|built_up|grassland|brush|forest|wetland|road"

    EXPERIMENT_NAME=DirectCD_${ARCH}_teamfeat_v013
    DATASET_CODE=Drop1_RightLeft_V1

    DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
    PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
    PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
    EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

    TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
    PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

    python -m watch stats $TRAIN_FPATH

    kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH

    # Write train and prediction configs
    python -m watch.tasks.fusion.fit \
        --channels=${CHANNELS} \
        --method="MultimodalTransformer" \
        --arch_name=$ARCH \
        --time_steps=9 \
        --chip_size=96 \
        --batch_size=1 \
        --accumulate_grad_batches=32 \
        --window_overlap=0.5 \
        --time_overlap=0.6 \
        --num_workers=6 \
        --max_epochs=400 \
        --patience=400 \
        --gpus=1  \
        --learning_rate=1e-3 \
        --weight_decay=1e-4 \
        --dropout=0.1 \
        --attention_impl=exact \
        --window_size=8 \
        --dump=$TRAIN_CONFIG_FPATH 

    python -m watch.tasks.fusion.predict \
        --gpus=1 \
        --write_preds=True \
        --write_probs=False \
        --dump=$PRED_CONFIG_FPATH

    ## TODO: predict and eval steps should be called after training.
    # But perhaps it should be a different invocation of the fit script?
    # So the simple route is still available?

    # Execute train -> predict -> evaluate
    python -m watch.tasks.fusion.fit \
               --config=$TRAIN_CONFIG_FPATH \
        --default_root_dir=$DEFAULT_ROOT_DIR \
           --package_fpath=$PACKAGE_FPATH \
            --train_dataset=$TRAIN_FPATH \
             --vali_dataset=$VALI_FPATH \
             --test_dataset=$TEST_FPATH \
             --num_sanity_val_steps=0  && \
    python -m watch.tasks.fusion.predict \
            --config=$PRED_CONFIG_FPATH \
            --test_dataset=$TEST_FPATH \
           --package_fpath=$PACKAGE_FPATH \
            --pred_dataset=$PRED_FPATH && \
    python -m watch.tasks.fusion.evaluate \
            --true_dataset=$TEST_FPATH \
            --pred_dataset=$PRED_FPATH \
              --eval_dpath=$EVAL_DPATH
        
}

basic_left_right_split(){
    # This is just the basic data for the teams
    LEFT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_data_left.kwcoco.json
    RIGHT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/combo_data_right.kwcoco.json

    python -m watch.cli.coco_spatial_crop \
            --src $COMBO_PROPOGATED_COCO_FPATH --dst $LEFT_COCO_FPATH \
            --suffix=_left

    python -m watch.cli.coco_spatial_crop \
            --src $COMBO_PROPOGATED_COCO_FPATH --dst $RIGHT_COCO_FPATH \
            --suffix=_right

}
