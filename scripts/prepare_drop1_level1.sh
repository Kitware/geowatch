__doc__="
This script prepares the Drop1 Level1 (i.e. Raw Data) dataset

It crops the unaligned data in the 'drop1' folder on DVC to regions and
projects the annotations onto the data
"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

UNSTRUCTURED_KWCOCO_BUNDLE=$DVC_DPATH/drop1
ALIGNED_KWCOCO_BUNDLE=$DVC_DPATH/Drop1-Aligned-L1

REGION_FPATH="$UNSTRUCTURED_KWCOCO_BUNDLE/all_regions.geojson" 
#REGION_FPATH="$UNSTRUCTURED_KWCOCO_BUNDLE/region_models/NZ_R001.geojson" 

# Unprotect DVC files that will get updated
dvc unprotect $UNSTRUCTURED_KWCOCO_BUNDLE/data.kwcoco.json
dvc unprotect $ALIGNED_KWCOCO_BUNDLE/*/*.json
dvc unprotect $ALIGNED_KWCOCO_BUNDLE/*.json

# Combine all regions into a single geojson file (exists in DVC)
#python -m watch.cli.merge_region_models \
#    --src $UNSTRUCTURED_KWCOCO_BUNDLE/region_models/*.geojson \
#    --dst $UNSTRUCTURED_KWCOCO_BUNDLE/all_regions.geojson

# Ensure unstructure drop1 data has geo-info in the kwcoco file 
# (makes running the align script faster)
python -m watch add_fields \
    --src $UNSTRUCTURED_KWCOCO_BUNDLE/data.kwcoco.json \
    --dst $UNSTRUCTURED_KWCOCO_BUNDLE/data.fielded.kwcoco.json --overwrite=warp --workers=avail

# Align and orthorectify the data to the chosen regions 
# TODO: FIXME: I dont understand why this doesnt work
# when I pass the glob path to all the regions
# I need to use the merged region script. Very strange.
WATCH_HACK_IMPORT_ORDER=geopandas,pyproj,gdal python -X faulthandler -m watch align \
    --src $UNSTRUCTURED_KWCOCO_BUNDLE/data.fielded.kwcoco.json \
    --dst $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
    --regions "$REGION_FPATH" \
    --keep img \
    --geo_preprop=False \
    --workers="avail/2" \
    --aux_workers="2" 

python -m watch project \
    --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
    --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
    --dst $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json.prop 
mv $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json.prop $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json


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



inspect(){

    # Make sure everything looks good
    jq ".videos[] | .name" $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json
    jq ".videos[] | .name" $ALIGNED_KWCOCO_BUNDLE/train_data.kwcoco.json
    jq ".videos[] | .name" $ALIGNED_KWCOCO_BUNDLE/vali_data.kwcoco.json

    kwcoco stats $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json
    smartwatch stats $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json

    python -m watch visualize --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json --channels "red|green|blue"
    python -m watch visualize --src $ALIGNED_KWCOCO_BUNDLE/vali_data_wv.kwcoco.json --channels "red|green|blue" --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz_wv_vali
    python -m watch visualize --src $ALIGNED_KWCOCO_BUNDLE/vali_data_nowv.kwcoco.json --channels "red|green|blue" --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz_nowv_vali
    python -m watch.cli.animate_visualizations --channels "red|green|blue" --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz_nowv_vali


    python -m watch visualize --src subdata.kwcoco.json --channels "red|green|blue" --viz_dpath=./_viz --animate --workers=8

}

teamfeatures(){
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    ALIGNED_KWCOCO_BUNDLE=$DVC_DPATH/Drop1-Aligned-L1
    BASE_COCO_FPATH=$ALIGNED_KWCOCO_BUNDLE/data_nowv.kwcoco.json
    RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth"
    DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH/models/landcover/visnav_remap_s2_subset.pt"

    export CUDA_VISIBLE_DEVICES="1"
    python -m watch.tasks.rutgers_material_seg.predict \
        --test_dataset=$BASE_COCO_FPATH \
        --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
        --default_config_key=iarpa \
        --pred_dataset=$ALIGNED_KWCOCO_BUNDLE/data_nowv_rutgers_mat_seg.kwcoco.json \
        --num_workers="8" \
        --batch_size=32 --gpus "0" \
        --compress=RAW --blocksize=64

    export CUDA_VISIBLE_DEVICES="2"
    python -m watch.tasks.landcover.predict \
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

    python -m watch stats $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json

    python -m watch visualize --src $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json --channels="matseg_0|matseg_1|matseg_2,matseg_3|matseg_4|matseg_5" --workers=8
    python -m watch visualize --src $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json --channels="bare_ground|forest|brush,built_up|cropland|wetland,snow_or_ice_field|forest|water" --workers=8

    # Split out train and validation data 
    # (TODO: add test when we get enough data)
    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json \
            --select_videos '.name | startswith("KR_R002") | not'

    kwcoco subset --src $ALIGNED_KWCOCO_BUNDLE/combo_nowv.kwcoco.json \
            --dst $ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json \
            --select_videos '.name | startswith("KR_R002")'

    echo "ALIGNED_KWCOCO_BUNDLE = $ALIGNED_KWCOCO_BUNDLE"
    python -m watch add_fields --src $ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json --dst=$ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json --overwrite=warp
    python -m watch add_fields --src $ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json --dst=$ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json --overwrite=warp

    python -m watch project \
        --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
        --src $ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json \
        --dst $ALIGNED_KWCOCO_BUNDLE/vali_combo11.kwcoco.json.prop 

    python -m watch project \
        --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
        --src $ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json \
        --dst $ALIGNED_KWCOCO_BUNDLE/train_combo11.kwcoco.json.prop 
}
