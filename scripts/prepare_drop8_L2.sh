#!/bin/bash

source "$HOME"/code/watch-smartflow-dags/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
#SENSORS=TA1-S2-L8-WV-PD-ACC-3

#'ta1-datacube-ara-4',
#'ta1-dsm-ara-4',

#'ta1-ls-ara-4',
#'ta1-pd-ara-4',
#'ta1-s2-ara-4',
#'ta1-wv-ara-4',

#'ta1-wv-ara-4t1',
#'ta1-wv-ara-4t2',
#'ta1-dsm-ara-4t1',
#'ta1-dsm-ara-4t2',


#SENSORS="ta1-ls-ara-4,ta1-pd-ara-4,ta1-s2-ara-4,ta1-wv-ara-4"
SENSORS="sentinel-s2-l2a-cogs,landsat-c2l2-sr,planet-dove,worldview-nitf"
#SENSORS="sentinel-s2-l2a-cogs,landsat-c2l2-sr,planet-dove,worldview-nitf"

DATASET_SUFFIX=Drop8-L2

# Just KR1
#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models/KR_R001.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models/KR_R001*.geojson"

# All Regions
#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*.geojson"

# T&E Regions Only
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*_R*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*_R*_*.geojson"


export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export REQUESTER_PAYS=True
#export SMART_STAC_API_KEY=""

# Construct the TA2-ready dataset
python -m geowatch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=20 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DVC_DATA_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --requester_pays=$REQUESTER_PAYS \
    --fields_workers=8 \
    --convert_workers=0 \
    --align_workers=4 \
    --align_aux_workers=0 \
    --ignore_duplicates=1 \
    --visualize=0 \
    --target_gsd="10GSD" \
    --cache=0 \
    --verbose=100 \
    --skip_existing=0 \
    --force_min_gsd=2.0 \
    --force_nodata=-9999 \
    --hack_lazy=False \
    --backend=tmux \
    --tmux_workers=8 \
    --sensor_to_time_window='
        S2: 2 weeks
        L8: 2 weeks
        PD: 2 weeks
    ' \
    --run=1


export AWS_REQUEST_PAYER=requester
python -m geowatch.cli.coco_add_watch_fields \
    --src=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Uncropped-Drop8-L2/data_KR_R001.kwcoco.zip \
    --dst=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Uncropped-Drop8-L2/data_KR_R001_fielded.kwcoco.zip \
    --overwrite=False \
    --workers=8 \
    --enable_video_stats=False \
    --target_gsd=10GSD \
    --remove_broken=True \
    --skip_populate_errors=False


AWS_DEFAULT_PROFILE=iarpa AWS_REQUEST_PAYER='requester' python -m geowatch.cli.coco_align \
   --regions "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/drop6_hard_v1/region_models/KR_R001.geojson" \
   --context_factor=1 \
   --geo_preprop=auto \
   --keep=img \
   --force_nodata=-9999 \
   --include_channels="None" \
   --exclude_channels="None" \
   --visualize=False \
   --debug_valid_regions=False \
   --rpc_align_method orthorectify \
   --sensor_to_time_window  "
        S2: 2 weeks
        L8: 2 weeks
        PD: 2 weeks " \
   --verbose=100 \
   --aux_workers=0 \
   --target_gsd=10GSD \
   --force_min_gsd=2.0 \
   --workers=4 \
   --tries=2 \
   --hack_lazy=False \
   --src=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Uncropped-Drop8-L2/data_KR_R001_fielded.kwcoco.zip \
   --dst=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop8-L2/KR_R001/imgonly-KR_R001-rawbands.kwcoco.zip \
   --dst_bundle_dpath=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop8-L2

