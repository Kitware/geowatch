__doc__="
This script prepares the Drop1 Level1 (i.e. Raw Data) dataset

It crops the unaligned data in the 'drop1' folder on DVC to regions and
projects the annotations onto the data
"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

ALIGNED_BUNDLE_NAME=Drop1-Aligned-L1
UNCROPPED_BUNDLE_NAME=drop1

UNCROPPED_DPATH=$DVC_DPATH/$UNCROPPED_BUNDLE_NAME
ALIGNED_KWCOCO_BUNDLE=$DVC_DPATH/$ALIGNED_BUNDLE_NAME


BASE_COCO_FPATH=$ALIGNED_KWCOCO_BUNDLE/data_nowv.kwcoco.json
RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth"
DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH/models/landcover/visnav_remap_s2_subset.pt"


echo "
DVC_DPATH                    = $DVC_DPATH
ALIGNED_BUNDLE_NAME          = $ALIGNED_BUNDLE_NAME
UNCROPPED_BUNDLE_NAME        = $UNCROPPED_BUNDLE_NAME
UNCROPPED_DPATH              = $UNCROPPED_DPATH
ALIGNED_KWCOCO_BUNDLE        = $ALIGNED_KWCOCO_BUNDLE
BASE_COCO_FPATH              = $BASE_COCO_FPATH
RUTGERS_MATERIAL_MODEL_FPATH = $RUTGERS_MATERIAL_MODEL_FPATH
DZYNE_LANDCOVER_MODEL_FPATH  = $DZYNE_LANDCOVER_MODEL_FPATH
"

unprotect_dvc(){
    # Unprotect DVC files that will get updated
    dvc unprotect $UNCROPPED_DPATH/data.kwcoco.json
    dvc unprotect $ALIGNED_KWCOCO_BUNDLE/*/*.json
    dvc unprotect $ALIGNED_KWCOCO_BUNDLE/*.json
}


_debug_extract_aligned(){
    __doc__="
        source ~/code/watch/scripts/prepare_drop1_level1.sh
    "

    kwcoco subset data.kwcoco.json
    kwcoco stats $UNCROPPED_DPATH/data.kwcoco.json
    smartwatch stats $UNCROPPED_DPATH/data.kwcoco.json

    kwcoco subset --src $UNCROPPED_DPATH/data.kwcoco.json \
            --dst $UNCROPPED_DPATH/data_nowv.kwcoco.json \
            --select_images '.sensor_coarse != "WV"'

    echo "UNCROPPED_DPATH = $UNCROPPED_DPATH"
    smartwatch add_fields \
        --src $UNCROPPED_DPATH/data_nowv.kwcoco.json \
        --dst $UNCROPPED_DPATH/test_fielded.kwcoco.json \
        --overwrite=warp --workers=12 --mode=process --profile

    smartwatch add_fields \
        --src $UNCROPPED_DPATH/test_fielded.kwcoco.json \
        --dst $UNCROPPED_DPATH/test_fielded.kwcoco.json \
        --overwrite=warp --workers=15 --mode=process 

    smartwatch stats $UNCROPPED_DPATH/test_fielded.kwcoco.json

    TEST_REGION=NZ_R001
    TEST_REGION=LT_R001

    WATCH_HACK_IMPORT_ORDER=geopandas,pyproj,gdal python -X faulthandler -m watch align \
        --src $UNCROPPED_DPATH/test_fielded.kwcoco.json \
        --dst $ALIGNED_KWCOCO_BUNDLE/_test/$TEST_REGION.kwcoco.json \
        --regions $UNCROPPED_DPATH/all_regions.geojson \
        --keep none \
        --exclude_sensors=WV \
        --workers="0" \
        --aux_workers="0" 

    #    --workers="avail/2" \
    #    --aux_workers="2" \

    #jq ".videos[] | .name" $ALIGNED_KWCOCO_BUNDLE/_test/$TEST_REGION.kwcoco.json

    smartwatch visualize \
        --src $ALIGNED_KWCOCO_BUNDLE/_test/$TEST_REGION.kwcoco.json \
        --channels "red|green|blue" \
        --select_images '.sensor_coarse != "WV"'

}


prepare_uncropped_data(){
    # Ensure unstructure drop1 data has geo-info in the kwcoco file 
    # (makes running the align script faster)
    echo "UNCROPPED_DPATH = $UNCROPPED_DPATH"
    smartwatch add_fields \
        --src $UNCROPPED_DPATH/data.kwcoco.json \
        --dst $UNCROPPED_DPATH/data.kwcoco.json \
        --overwrite=warp --workers=avail --mode=process 
}


extract_aligned_cropped_regions(){
    __doc__="
    Extract relevant data from 

    source ~/code/watch/scripts/prepare_drop1_level1.sh
    "
    # Align and orthorectify the data to the chosen regions 
    # TODO: FIXME: When I pass in a glob string of region files this doesnt work
    # not sure why. It should work. I need to use the merged region script. Very strange.
    smartwatch align \
        --src $UNCROPPED_DPATH/data.kwcoco.json \
        --dst $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
        --regions "$UNCROPPED_DPATH/all_regions.geojson" \
        --keep img \
        --workers="avail/2" \
        --aux_workers="2" 

    smartwatch project \
        --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
        --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
        --dst $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json

    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/data_nowv.kwcoco.json \
            --select_images '.sensor_coarse != "WV"'

    # Split out train and validation data 
    # (TODO: add test when we get enough data)
    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/train_data.kwcoco.json \
            --select_videos '.name | startswith("KR_") | not'

    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/vali_data.kwcoco.json \
            --select_videos '.name | startswith("KR_")'

    # 
    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/train_data.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/train_data_nowv.kwcoco.json \
            --select_images '.sensor_coarse != "WV"'

    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/vali_data.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/vali_data_nowv.kwcoco.json \
            --select_images '.sensor_coarse != "WV"'

    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/train_data.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/train_data_wv.kwcoco.json \
            --select_images '.sensor_coarse == "WV"'

    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/vali_data.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/vali_data_wv.kwcoco.json \
            --select_images '.sensor_coarse == "WV"'
}


clean(){
    # Remove old non-necessary files
    ls
}



update_dvc_files(){
    cd $ALIGNED_KWCOCO_BUNDLE
    dvc add \
        data.kwcoco.json \
        data_nowv.kwcoco.json \
        train_data.kwcoco.json \
        vali_data.kwcoco.json \
        train_data_wv.kwcoco.json \
        vali_data_wv.kwcoco.json \
        train_data_nowv.kwcoco.json \
        vali_data_nowv.kwcoco.json \
        */S2 \
        */L8 \
        */WV \
        */subdata.kwcoco.json
}



inspect(){
    __doc__="
    Get stats and visualize data produced by this script 

    source ~/code/watch/scripts/prepare_drop1_level1.sh
    "

    # Make sure everything looks good
    jq ".videos[] | .name" $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json
    jq ".videos[] | .name" $ALIGNED_KWCOCO_BUNDLE/train_data.kwcoco.json
    jq ".videos[] | .name" $ALIGNED_KWCOCO_BUNDLE/vali_data.kwcoco.json

    kwcoco stats $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json
    smartwatch stats $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json

    smartwatch visualize --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json --channels "red|green|blue"
    smartwatch visualize --src $ALIGNED_KWCOCO_BUNDLE/vali_data_wv.kwcoco.json --channels "red|green|blue" --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz_wv_vali
    smartwatch visualize --src $ALIGNED_KWCOCO_BUNDLE/vali_data_nowv.kwcoco.json --channels "red|green|blue" --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz_nowv_vali
    #smartwatch.cli.animate_visualizations --channels "red|green|blue" --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz_nowv_vali
    #smartwatch visualize --src subdata.kwcoco.json --channels "red|green|blue" --viz_dpath=./_viz --animate=True --workers=8
}

teamfeatures(){
    export CUDA_VISIBLE_DEVICES="1"
    smartwatch.tasks.rutgers_material_seg.predict \
        --test_dataset=$BASE_COCO_FPATH \
        --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
        --default_config_key=iarpa \
        --pred_dataset=$ALIGNED_KWCOCO_BUNDLE/data_nowv_rutgers_mat_seg.kwcoco.json \
        --num_workers="8" \
        --batch_size=32 --gpus "0" \
        --compress=RAW --blocksize=64

    export CUDA_VISIBLE_DEVICES="2"
    smartwatch.tasks.landcover.predict \
        --dataset=$BASE_COCO_FPATH \
        --deployed=$DZYNE_LANDCOVER_MODEL_FPATH  \
        --device=0 \
        --num_workers="16" \
        --output=$ALIGNED_KWCOCO_BUNDLE/data_nowv_dzyne_landcover.kwcoco.json

    
    python ~/code/watch/watch/cli/coco_combine_features.py \
        --src $BASE_COCO_FPATH \
              $ALIGNED_KWCOCO_BUNDLE/data_nowv_rutgers_mat_seg.kwcoco.json \
              $ALIGNED_KWCOCO_BUNDLE/data_nowv_dzyne_landcover.kwcoco.json \
        --dst $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json

    smartwatch stats $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json

    smartwatch visualize --src $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json --channels="matseg_0|matseg_1|matseg_2,matseg_3|matseg_4|matseg_5" --workers=8
    smartwatch visualize --src $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json --channels="bare_ground|forest|brush,built_up|cropland|wetland,snow_or_ice_field|forest|water" --workers=8

    # Split out train and validation data 
    # (TODO: add test when we get enough data)
    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json \
            --select_videos '.name | startswith("KR_R002") | not'

    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json \
            --select_videos '.name | startswith("KR_R002")'

    echo "ALIGNED_KWCOCO_BUNDLE = $ALIGNED_KWCOCO_BUNDLE"
    smartwatch add_fields --src $ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json --dst=$ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json --overwrite=warp
    smartwatch add_fields --src $ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json --dst=$ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json --overwrite=warp

    smartwatch project \
        --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
        --src $ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json \
        --dst $ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json.prop 

    smartwatch project \
        --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
        --src $ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json \
        --dst $ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json.prop 
}


main_drop1_level1(){
    prepare_uncropped_data
    extract_aligned_cropped_regions
}
