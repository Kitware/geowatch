#!/bin/bash

source "$HOME"/code/watch/secrets/secrets

DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=TA1-S2-L8-WV-PD-ACC-3
DATASET_SUFFIX=Drop7
REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/drop6_hard_v1/*.geojson"
SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/drop6_hard_v1/*.geojson"

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
    --max_queue_size=4 \
    --ignore_duplicates=1 \
    --separate_region_queues=1 \
    --separate_align_jobs=1 \
    --visualize=0 \
    --target_gsd=10 \
    --cache=0 \
    --verbose=100 \
    --skip_existing=0 \
    --force_min_gsd=2.0 \
    --backend=tmux --run=0

    #--exclude_channels=pan \
