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
python -m watch align \
    --src $UNSTRUCTURED_KWCOCO_BUNDLE/data.fielded.kwcoco.json \
    --dst $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
    --regions "$REGION_FPATH" \
    --keep img \
    --exclude_sensors=WV \
    --workers="avail/2" \
    --aux_workers="2" 

python -m watch project \
    --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
    --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
    --dst $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json.prop 
mv $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json.prop $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json


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

    python -m watch visualize --src $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json --channels "red|green|blue"
    python -m watch visualize --src $ALIGNED_KWCOCO_BUNDLE/vali_data_wv.kwcoco.json --channels "red|green|blue" --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz_wv_vali
    python -m watch visualize --src $ALIGNED_KWCOCO_BUNDLE/vali_data_nowv.kwcoco.json --channels "red|green|blue" --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz_nowv_vali

}
