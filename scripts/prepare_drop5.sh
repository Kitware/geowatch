#!/bin/bash

source "$HOME"/code/watch/secrets/secrets

DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=TA1-S2-L8-WV-PD-ACC-1
DATASET_SUFFIX=Drop5-2022-11-07-c30-$SENSORS
REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*.geojson"
SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=30 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DATA_DVC_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --exclude_channels=pan \
    --requester_pays=False \
    --fields_workers=8 \
    --convert_workers=8 \
    --align_workers=2 \
    --max_queue_size=1 \
    --ignore_duplicates=1 \
    --separate_region_queues=1 \
    --separate_align_jobs=1 \
    --visualize=0 \
    --target_gsd=10 \
    --cache=1 \
    --skip_existing=1 \
    --warp_tries=2 \
    --asset_timeout="1hour" \
    --image_timeout="1hour" \
    --backend=tmux --run=1


#--include_channels="blue|green|red|nir|swir16|swir22" \


smartwatch stats "$DATA_DVC_DPATH"/Aligned-Drop5-2022-10-11-c30-TA1-S2-L8-WV-PD-ACC/data.kwcoco.json "$DATA_DVC_DPATH"/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data.kwcoco.json


cd "$DATA_DVC_DPATH"/Aligned-Drop5-2022-10-11-c30-TA1-S2-L8-WV-PD-ACC-1

codeblock  "

# CHECK SIZES:
sensors = ['WV', 'PD', 'S2', 'L8']
sensor_to_sizes = ub.ddict(list)
for s in sensors:
    for p in list(bundle_dpath.glob('*/' + s)):
        nbytes = ub.cmd(f'du -sL {p}')['out'].split('\t')[0]
        sensor_to_sizes[s].append(nbytes)

sensor_to_bytes = ub.udict(sensor_to_sizes).map_values(lambda x: sum(int(b) * 1024 for b in x))
import xdev as xd
sensor_to_size = sensor_to_bytes.map_values(xd.byte_str)
print('sensor_to_size = {}'.format(ub.repr2(sensor_to_size, nl=1)))

total_size = sum(sensor_to_bytes.values())
print('total = {}'.format(xd.byte_str(total_size)))

"




#DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
#SENSORS=TA1-S2-L8-WV-PD-ACC
#DATASET_SUFFIX=TestRegionKW4
#REGION_GLOBSTR="/home/joncrall/Downloads/KW_test4.geojson"
#SITE_GLOBSTR=""

#export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

## Construct the TA2-ready dataset
#python -m watch.cli.prepare_ta2_dataset \
#    --dataset_suffix=$DATASET_SUFFIX \
#    --stac_query_mode=auto \
#    --cloud_cover=30 \
#    --sensors="$SENSORS" \
#    --api_key=env:SMART_STAC_API_KEY \
#    --collated True \
#    --dvc_dpath="$DATA_DVC_DPATH" \
#    --aws_profile=iarpa \
#    --region_globstr="$REGION_GLOBSTR" \
#    --site_globstr="$SITE_GLOBSTR" \
#    --requester_pays=False \
#    --fields_workers=20 \
#    --convert_workers=8 \
#    --max_queue_size=2 \
#    --align_workers=12 \
#    --cache=1 \
#    --ignore_duplicates=1 \
#    --separate_region_queues=1 \
#    --separate_align_jobs=1 \
#    --visualize=0 \
#    --target_gsd=10 \
#    --backend=tmux --run=1
