#!/bin/bash

source "$HOME"/code/watch/secrets/secrets

DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=TA1-S2-L8-WV-PD-ACC-3
DATASET_SUFFIX=Drop7
REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/drop6_hard_v1/region_models/*.geojson"
SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"

export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=20 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DATA_DVC_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --requester_pays=False \
    --fields_workers=8 \
    --convert_workers=8 \
    --align_workers=4 \
    --align_aux_workers=0 \
    --backend=tmux \
    --tmux_workers=10 \
    --ignore_duplicates=1 \
    --separate_region_queues=1 \
    --separate_align_jobs=1 \
    --visualize=0 \
    --target_gsd=10 \
    --cache=1 \
    --verbose=100 \
    --skip_existing=1 \
    --force_min_gsd=2.0 \
    --run=1

# ~/code/watch/dev/poc/prepare_time_combined_dataset.py

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python ~/code/watch/dev/poc/prepare_time_combined_dataset.py \
    --regions=all \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-MedianSummer10GSD \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/site_models \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/region_models \
    --spatial_tile_size=256 \
    --merge_method=median \
    --remove_seasons=spring,fall,winter \
    --tmux_workers=2 \
    --time_window=1y \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --run=1


    #--regions="[
    #        # T&E Regions
    #        AE_R001, BH_R001, BR_R001, BR_R002, BR_R004, BR_R005, CH_R001,
    #        KR_R001,
    #        KR_R002, LT_R001, NZ_R001, US_R001, US_R004, US_R005,
    #        US_R006, US_R007,
    #        # # iMerit Regions
    #        AE_C001,
    #        AE_C002,
    #        AE_C003, PE_C001, QA_C001, SA_C005, US_C000, US_C010,
    #        US_C011, US_C012,
    #]" \
